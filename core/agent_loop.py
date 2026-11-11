import re
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from .prompt_builder import PromptBuilder, classify_effort, estimate_tokens, ReasoningEffort
from .state_machine import StateMachine, AgentState
from .user_context import UserContext, CLI_USER, current_user_context
from .tenant_context import (
    TenantContext, current_tenant, set_tenant_context, reset_tenant_context,
)
from state.checkpoint_manager import CheckpointManager
from state.compaction import CompactionEngine

logger = logging.getLogger(__name__)


_CORRECTION_SIGNALS = frozenset([
    "actually", "don't", "instead", "wrong", "not that",
    "i prefer", "i want", "please use", "use this", "never do",
])
# Too ambiguous mid-sentence ("there is no error…") — only count as a
# correction when the message STARTS with them.
_CORRECTION_PREFIX_ONLY = ("no,", "no ")

_PASSIVITY_SIGNALS = [
    "i don't have access", "i'm unable to access",
    "is not accessible", "is outside",
    "outside the scope", "outside my capabilities",
    "i don't have the ability", "beyond my capabilities",
]
_MAX_PASSIVITY_CORRECTIONS = 2

_CONTINUATION_SIGNALS = [
    "let me ", "let's ", "now i'll", "now i will", "now let me", "i'll now",
    "i will now", "next, i", "next i'll", "next i will", "moving on",
    "let me start", "let me proceed", "let me continue", "i'll start",
    "i'll proceed", "i'll go ahead", "i'm going to", "going to ",
]
# Common CLOSING phrases that contain a continuation signal ("let me know…")
# — stripped before signal matching so a finished answer is never re-prompted.
_CONTINUATION_NEGATIVE = (
    "let me know", "let us know", "feel free to", "don't hesitate to",
)
_MAX_CONTINUATION_NUDGES = 4


_TURN_SEMAPHORES: Dict[int, asyncio.Semaphore] = {}


def _turn_semaphore() -> Optional[asyncio.Semaphore]:
    try:
        from core.config import app_config
        cap = app_config.get_int("agent.max_concurrent_turns", 8)
    except Exception:
        cap = 8
    if cap <= 0:
        return None
    key = id(asyncio.get_running_loop())
    sem = _TURN_SEMAPHORES.get(key)
    if sem is None:
        sem = asyncio.Semaphore(cap)
        _TURN_SEMAPHORES[key] = sem
    return sem


class AgentLoop:
    """Core execution engine — Observe-Reason-Act with native function calling.
    
    Architecture (identical to Agent):
    
    1. TOOL CALLING: Native function calling via OpenAI-compatible API.
       Tools via `tools` parameter. Model returns structured `tool_calls`.
       Results via `role: "tool"` with `tool_call_id` linkage.
    
    2. CONTEXT MANAGEMENT:
       - Per-message token caching (_tokens field)
       - Tool result compression: old results shrunk to ~20 tokens
       - Smart priority-based pruning: tool results → empty assistants → user msgs
       - Checkpoint injection during pruning
    
    3. LOOP CONTROL:
       - finish_reason-driven: "stop" → respond, "tool_calls" → execute
       - 200-iteration safety cap
       - Parallel tool execution via asyncio.gather
       - Repeated failure detection with auto-correction
    
    4. AUTO-MEMORY:
       - User corrections auto-stored as preference memories
       - Conversation summary generated at session end
    """
    
    DEFAULT_HISTORY_TOKEN_BUDGET = 32_000
    # One high safety cap for every turn (the Claude Code model): the real
    # limiters are the model's own stop decision, the conversation token
    # budget, interruption, and the loop guards — NOT a guess about task
    # complexity derived from keywords in the user's phrasing.
    MAX_TOOL_ITERATIONS = 150
    TOOL_RESULT_COMPRESS_THRESHOLD = 500
    MAX_CONSECUTIVE_FAILURES = 3
    COMPACTION_THRESHOLD = 0.8
    REFLECTION_INTERVAL = 12
    
    def __init__(self, workspace_path: str, llm_client: Any, tool_registry: Any,
                 event_bus: Any, transcript_logger: Any = None, memory_manager: Any = None,
                 user_context: UserContext = None, tool_executor: Any = None,
                 db: Any = None, prompt_overrides: dict = None,
                 guardrail_engine: Any = None, approval_manager: Any = None,
                 audit_sink: Any = None, tenant_context: TenantContext = None,
                 conversation_id: str = None, dynamic_picker: Any = None,
                 usage_sink: Any = None, budget_check: Any = None):
        self.workspace_path = workspace_path
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self._dynamic_picker = dynamic_picker
        self.event_bus = event_bus
        self.transcript_logger = transcript_logger
        self.memory_manager = memory_manager
        self.user_context = user_context or CLI_USER
        self.tenant_context = tenant_context or current_tenant()
        self._prompt_overrides = prompt_overrides or {}
        self.guardrail_engine = guardrail_engine
        self.approval_manager = approval_manager
        self.prompt_builder = PromptBuilder(workspace_path)
        self.state = StateMachine()
        self._processing_lock = asyncio.Lock()
        
        from core.tool_executor import LocalToolExecutor
        self._local_executor = LocalToolExecutor(
            tool_registry, guardrail_engine=guardrail_engine,
            approval_manager=approval_manager, audit_sink=audit_sink,
        )
        self.bind_tool_executor(
            tool_executor, guardrail_engine=guardrail_engine,
            approval_manager=approval_manager, audit_sink=audit_sink,
        )
        
        self.conversation_id = conversation_id
        self.checkpoint_mgr = CheckpointManager(
            db or workspace_path,
            session_id=(conversation_id
                        or getattr(transcript_logger, 'conversation_id', None)
                        or 'default'),
            user_id=self.user_context.user_id,
        )
        
        from state.history_store import HistoryStore
        self.history_store = HistoryStore(
            db or workspace_path,
            session_id=(conversation_id
                        or getattr(transcript_logger, 'conversation_id', None)
                        or 'default'),
            user_id=self.user_context.user_id,
        )

        self.conversation_history: List[Dict[str, Any]] = []
        self._total_history_tokens = 0

        if self._resume_enabled() and self.history_store.available:
            restored = self.history_store.load()
            if restored:
                self.restore_history(restored)

        self._last_prompt_tokens = 0
        self._history_tokens_at_last_call = 0
        self._conversation_tokens_used = 0
        self._conversation_cost_usd = 0.0
        # usage_sink(record: dict) — host-injected persistence hook (twin of
        # audit_sink): receives every LLM call's real usage for the tenant
        # ledger. budget_check() -> str|None — host-injected limit probe; a
        # returned message stops the turn gracefully at the next iteration.
        # Both best-effort: engine never fails a turn on hook errors.
        self._usage_sink = usage_sink
        self._budget_check = budget_check
        self._consecutive_failures: Dict[str, int] = {}
        self._interaction_fail_cycles = 0
        self._passivity_corrections = 0
        self._continuation_nudges = 0
        self._cached_system_prompt: Optional[str] = None
        self._effort_prompts: Dict[Any, str] = {}
        self._openai_tools_cache: Optional[list] = None
        # Session-sticky tool subset, in FIRST-SEEN order. Serialization must
        # append newly selected tools at the END: providers cache the longest
        # byte-stable prefix, so interleaving a new tool at its registry
        # position would shift every schema after it and re-bill the whole
        # prefix, while appending re-bills only the tail.
        self._session_tool_order: list = []
        self._session_tool_names: set = set()
        self._last_memory_context: Optional[str] = None

        from state.todos import TodoStore
        self.todos = TodoStore()
        self._todo_snapshot = ""
        self._todo_stale_iters = 0

        self._cancel_requested = False
        self._steering_queue: List[str] = []
        self._compaction_count = 0

        # Learning-review observation (host-driven loop; engine only watches).
        # _skills_in_play is filled by TWO paths:
        #   1. skill_view tool calls during a turn (via current_skill_tracker
        #      ContextVar — active usage)
        #   2. seed_skills_in_play() called by the factory at creation time
        #      when skills are injected into the system prompt (passive usage)
        # A correction later in the same conversation counts against all
        # skills in play (the effectiveness signal the hermes design lacks).
        self._skills_in_play: set = set()
        self._turn_review_signals: Dict[str, Any] = {}
        self._last_turn_iterations = 0

        from core.config import app_config
        budget = app_config.get("agent.history_token_budget")
        if budget:
            self.history_token_budget = int(budget)
        else:
            self.history_token_budget = self._derive_budget_from_window(app_config)

        self.event_bus.subscribe('user_input', self._process_input)
        self.event_bus.subscribe('cancel_turn', self._on_cancel)
        self.event_bus.subscribe('subagent_completed', self._on_subagent_completed)

    async def _on_cancel(self, payload: dict):
        self.request_cancel()

    async def _on_subagent_completed(self, payload: dict):
        """Background sub-agent report → this conversation's loop.

        _process_input does the right thing in both states: mid-turn it
        becomes a steering message (injected at the next iteration); idle
        it starts a new turn that reacts to the report.
        """
        report = (f"[SUBAGENT COMPLETED] Task '{payload.get('task_id', '?')}' "
                  f"(agent: {payload.get('agent_name', '?')}) finished. "
                  f"Report:\n{str(payload.get('result', ''))[:6000]}")
        await self._process_input({"user_input": report})

    def request_cancel(self):
        """Stop the in-flight turn at the next iteration boundary.

        The Claude Code Esc contract: cancellation is a first-class loop
        input, not an exception — the turn ends cleanly with a
        user-facing stop message, history stays consistent, and the next
        user message continues the conversation.
        """
        self._cancel_requested = True

    def queue_steering_message(self, text: str):
        """Inject a user message into the CURRENT turn (drained each
        iteration, suffix-only so the prompt cache is unaffected)."""
        if text:
            self._steering_queue.append(text)

    def bind_tool_executor(self, tool_executor=None, *, guardrail_engine=None,
                           approval_manager=None, audit_sink=None) -> None:
        """Resolve the ACTIVE tool executor for the current request/transport.

        Called once at construction and again on every reused turn (by the
        session pool). Pass a live RemoteToolExecutor to delegate tools to a
        connected client (WebSocket / Local Server mode); pass None to run tools
        on the VPS sandbox via the agent's durable LocalToolExecutor. This makes
        executor binding a per-request decision — a transport switch can never
        inherit a stale executor bound to a closed socket.
        """
        if tool_executor is None:
            self.tool_executor = self._local_executor
            return
        ge = guardrail_engine if guardrail_engine is not None else self.guardrail_engine
        if ge is not None and getattr(tool_executor, "_guardrail", None) is None:
            tool_executor._guardrail = ge
        am = approval_manager if approval_manager is not None else self.approval_manager
        if am is not None and getattr(tool_executor, "_approvals", None) is None:
            tool_executor._approvals = am
        if audit_sink is not None and getattr(tool_executor, "_audit", None) is None:
            tool_executor._audit = audit_sink
        self.tool_executor = tool_executor

    def _derive_budget_from_window(self, app_config) -> int:
        """History budget from the model context window, or the flat default.

        budget = context_window − max_tokens(reserved for output) − safety
        """
        window = app_config.get_int("llm.context_window", 0)
        if window <= 0:
            return self.DEFAULT_HISTORY_TOKEN_BUDGET
        reserved_output = getattr(self.llm_client, "max_tokens", 16000) or 16000
        safety = 2000
        budget = window - int(reserved_output) - safety
        return max(budget, 8000)


    def _append_message(self, msg: Dict[str, Any]):
        """Append a message to history with cached token count.
        
        Every message gets a `_tokens` field computed once at insertion.
        This avoids re-scanning the entire history for token estimation.
        """
        tokens = self._compute_msg_tokens(msg)
        msg["_tokens"] = tokens
        self.conversation_history.append(msg)
        self._total_history_tokens += tokens

    def _compute_msg_tokens(self, msg: Dict[str, Any]) -> int:
        """Compute token count for a single message."""
        content = msg.get("content") or ""
        tokens = estimate_tokens(content)
        if "tool_calls" in msg:
            tokens += estimate_tokens(json.dumps(msg["tool_calls"]))
        if msg.get("_thinking"):
            try:
                tokens += estimate_tokens(json.dumps(msg["_thinking"]))
            except (TypeError, ValueError):
                pass
        return tokens

    def _estimate_history_tokens(self) -> int:
        """Fast token count using cached per-message values (char heuristic)."""
        return self._total_history_tokens

    def _effective_context_tokens(self) -> int:
        """Best estimate of the REAL prompt size that will be sent next.

        Anchored to the last API-reported usage.prompt_tokens (which already
        includes the system prompt + tool schemas + history), plus the
        char-estimated delta of messages appended since that measurement.
        Falls back to the raw history estimate before the first call.

        This is what pruning decisions should use — the previous code compared
        history-only estimates against the budget and ignored the (often
        10k+ token) fixed system+tool overhead, risking context overflow.
        """
        if self._last_prompt_tokens > 0:
            delta = self._total_history_tokens - self._history_tokens_at_last_call
            return max(self._last_prompt_tokens + delta, self._total_history_tokens)
        return self._total_history_tokens

    def _record_usage(self, usage: Dict[str, Any], served_model: str = None):
        """Anchor real-token accounting to the API's reported usage.

        `served_model` is the model that actually answered (fallback-aware);
        without it a fallback-served call would be priced and ledgered as the
        configured primary — e.g. a misconfigured claude-* definition served
        by deepseek-v4-pro was reported at Sonnet pricing (8.7× the real cost).
        """
        if not usage:
            return
        pt = usage.get("prompt_tokens") or 0
        if pt > 0:
            self._last_prompt_tokens = pt
            self._history_tokens_at_last_call = self._total_history_tokens
        ct = usage.get("completion_tokens") or 0
        self._conversation_tokens_used += pt + ct
        # Cached prompt tokens bill at the provider's cache-read rate, not
        # the full input rate — pricing the whole prompt at in_p over-charged
        # cache-heavy turns 10-30×.
        from core.telemetry import cached_prompt_tokens
        cached = min(cached_prompt_tokens(usage), pt)
        model = served_model or getattr(self.llm_client, "model", "")
        cost = 0.0
        try:
            from core.telemetry import telemetry
            # The client says who bills it: the same model costs one thing
            # through a router and another — by time of day — from DeepSeek.
            in_p, out_p, cr_p = telemetry._resolve_price(
                model, provider=getattr(self.llm_client, "provider", None),
                url=getattr(self.llm_client, "url", None))
            cost = ((pt - cached) * in_p + cached * cr_p
                    + ct * out_p) / 1_000_000.0
            self._conversation_cost_usd += cost
        except Exception:
            pass
        if self._usage_sink is not None:
            try:
                self._usage_sink({
                    "model": model,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "cached_tokens": cached,
                    "cost_usd": cost,
                    "conversation_id": self.conversation_id,
                })
            except Exception:
                logging.getLogger(__name__).debug(
                    "usage_sink failed (ignored)", exc_info=True)

    def _conversation_budget_exceeded(self) -> bool:
        """True when cumulative conversation tokens pass the configured soft cap.

        Soft = checked between iterations, not mid-call: the current turn is
        allowed to finish its in-flight LLM call, then the loop stops with a
        partial answer instead of silently spending unboundedly.
        """
        try:
            from core.config import app_config
            cap = app_config.get_int("agent.conversation_token_budget", 0)
        except Exception:
            cap = 0
        return cap > 0 and self._conversation_tokens_used >= cap

    def _recalculate_tokens(self):
        """Full recalculation — called after compression/pruning modifies messages."""
        total = 0
        for msg in self.conversation_history:
            if "_tokens" not in msg:
                msg["_tokens"] = self._compute_msg_tokens(msg)
            total += msg["_tokens"]
        self._total_history_tokens = total


    def _compress_old_tool_results(self, keep_recent: int = 3):
        """Compress tool results from previous iterations to save tokens.
        
        This is HOW Agent manages context without losing information:
        - Results the model has ALREADY SEEN get compressed to a summary
        - Recent results (last `keep_recent` iterations) stay full
        - Compression is in-place: the message content is replaced
        
        A 4000-token file read becomes a 20-token summary:
        "[Compressed: 16234 chars, 245 lines] File auth.py shown with line numbers."
        """
        protected_tail = keep_recent * 4
        cutoff = len(self.conversation_history) - protected_tail
        
        if cutoff <= 1:
            return
        
        compressed_count = 0
        saved_tokens = 0
        
        for i in range(1, cutoff):
            msg = self.conversation_history[i]
            if msg.get("role") != "tool":
                continue
            
            content = msg.get("content", "")
            if len(content) <= self.TOOL_RESULT_COMPRESS_THRESHOLD:
                continue
            
            lines = content.split('\n')
            line_count = len(lines)
            char_count = len(content)

            head = content[:160].strip()
            tail = content[-160:].strip() if char_count > 360 else ""

            old_tokens = msg.get("_tokens", estimate_tokens(content))

            if tail and tail != head:
                compressed = (
                    f"[Compressed: {char_count} chars, {line_count} lines]\n"
                    f"{head}\n…\n{tail}"
                )
            else:
                compressed = (
                    f"[Compressed: {char_count} chars, {line_count} lines] "
                    f"{head}…"
                )
            
            msg["content"] = compressed
            new_tokens = estimate_tokens(compressed)
            msg["_tokens"] = new_tokens
            
            saved_tokens += (old_tokens - new_tokens)
            compressed_count += 1
        
        if compressed_count > 0:
            self._recalculate_tokens()
            self._log("SYSTEM", "COMPRESSION",
                      f"Compressed {compressed_count} tool results, "
                      f"saved ~{saved_tokens} tokens")


    @staticmethod
    def _context_strategy() -> str:
        """`agent.context_strategy`: cache_first (default) | legacy.

        cache_first — the Claude Code model: history is APPEND-ONLY between
        rare, one-shot compaction events. Providers cache the byte-stable
        prefix (Anthropic cache_control, OpenAI/DeepSeek/Gemini implicit
        prefix caches) and serve prior tokens at ~10% price. Any in-place
        edit of an already-sent message invalidates the cache from that
        point on — so the legacy per-turn compress/prune cascade, which
        mutates old messages on EVERY turn (and every few iterations
        mid-turn), silently re-bills the whole conversation at full price.
        """
        try:
            from core.config import app_config
            return (app_config.get("agent.context_strategy", "cache_first")
                    or "cache_first").strip().lower()
        except Exception:
            return "cache_first"

    def _context_trigger_tokens(self) -> int:
        """Token level at which context management kicks in."""
        if self._context_strategy() == "legacy":
            return self.history_token_budget
        return int(self.history_token_budget * self.COMPACTION_THRESHOLD)

    async def _prune_history(self):
        """Manage context size, then repair tool-call pairing.

        cache_first: NO-OP below the compaction threshold (append-only —
        never touch sent messages). Above it, ONE compaction event rewrites
        old messages into a summary + recent tail; the prompt cache is
        invalidated exactly once and re-warms on the next call. An
        emergency oldest-first drop only runs if a single oversized tail
        still exceeds the hard budget (rare — tool results are capped at
        insertion).

        legacy: the original per-turn compress → compact → phased-prune
        cascade (config escape hatch).

        Compaction and pruning can split an assistant `tool_calls` message
        from its `role: "tool"` results; the API rejects such histories, so
        every mutation is followed by a pairing repair.
        """
        if self._context_strategy() == "legacy":
            await self._prune_history_inner()
            self._repair_tool_pairing()
            return

        if self._effective_context_tokens() <= self._context_trigger_tokens():
            return

        await self._compact_conversation()
        self._repair_tool_pairing()

        if self._effective_context_tokens() > self.history_token_budget:
            min_tail = 6
            dropped = 0
            while (self._effective_context_tokens() > self.history_token_budget
                   and len(self.conversation_history) > min_tail + 1):
                removed = self.conversation_history.pop(1)
                self._total_history_tokens -= removed.get("_tokens", 0)
                dropped += 1
            if dropped:
                self._repair_tool_pairing()
                self._log("SYSTEM", "PRUNING",
                          f"Emergency drop of {dropped} messages after compaction "
                          f"(oversized tail)")

    def _repair_tool_pairing(self):
        """Remove orphan tool messages and unanswered tool_calls.

        Invariants enforced (required by the OpenAI chat protocol):
        - Every `role: "tool"` message must answer a tool_call id from the
          most recent assistant message.
        - Every tool_call id on an assistant message must have a tool result.
        Violations are repaired by dropping the orphan side.
        """
        repaired: List[Dict[str, Any]] = []
        pending_ids: set = set()
        dropped = 0

        def _close_group():
            """Strip unanswered tool_call ids from the group's assistant msg."""
            nonlocal dropped
            if not pending_ids:
                return
            for prev in reversed(repaired):
                if prev.get("role") == "assistant" and prev.get("tool_calls"):
                    kept = [tc for tc in prev["tool_calls"]
                            if tc.get("id") not in pending_ids]
                    if kept:
                        prev["tool_calls"] = kept
                    else:
                        del prev["tool_calls"]
                        if not prev.get("content"):
                            prev["content"] = "(tool calls pruned from context)"
                    prev["_tokens"] = self._compute_msg_tokens(prev)
                    dropped += 1
                    break

        for msg in self.conversation_history:
            role = msg.get("role")
            if role == "tool":
                call_id = msg.get("tool_call_id")
                if call_id in pending_ids:
                    pending_ids.discard(call_id)
                    repaired.append(msg)
                else:
                    dropped += 1
                continue

            _close_group()
            pending_ids = set()
            if role == "assistant" and msg.get("tool_calls"):
                pending_ids = {tc.get("id") for tc in msg["tool_calls"]}
            repaired.append(msg)

        _close_group()

        if dropped > 0:
            self.conversation_history = repaired
            self._recalculate_tokens()
            self._log("SYSTEM", "PRUNING",
                      f"Repaired tool pairing: fixed {dropped} orphaned entries")

    async def _prune_history_inner(self):
        """Compaction-first pruning matching Agent's strategy.

        Phase 0: Compress old tool results (in-place, cheapest)
        Phase 0.5: COMPACT old conversation into structured summary (NEW)
        Phase 1-4: Priority-based pruning (fallback if compaction isn't enough)

        Compaction vs Pruning:
        - Compaction: summarizes messages → preserves decisions/findings
        - Pruning: deletes messages → loses context permanently

        Always preserve:
        - First user message (task anchor)
        - Compaction summary (if any)
        - Last 6 messages (immediate context)
        - Checkpoint injected after pruning
        """
        self._compress_old_tool_results(keep_recent=3)
        
        if self._effective_context_tokens() <= self.history_token_budget:
            return
        
        compaction_budget = int(self.history_token_budget * self.COMPACTION_THRESHOLD)
        if self._effective_context_tokens() > compaction_budget:
            await self._compact_conversation()
        
        if self._effective_context_tokens() <= self.history_token_budget:
            return
        
        min_tail = 6
        checkpoint_summary = self.checkpoint_mgr.get_checkpoint_summary()
        
        self._drop_by_role_filter(
            min_tail,
            lambda msg: msg.get("role") == "tool" and "[Compressed:" in msg.get("content", "")
        )
        if self._effective_context_tokens() <= self.history_token_budget:
            self._inject_checkpoint(checkpoint_summary)
            return
        
        self._drop_by_role_filter(
            min_tail,
            lambda msg: msg.get("role") == "tool"
        )
        if self._effective_context_tokens() <= self.history_token_budget:
            self._inject_checkpoint(checkpoint_summary)
            return
        
        self._drop_by_role_filter(
            min_tail,
            lambda msg: (msg.get("role") == "assistant" and 
                        "tool_calls" in msg and 
                        not msg.get("content"))
        )
        if self._effective_context_tokens() <= self.history_token_budget:
            self._inject_checkpoint(checkpoint_summary)
            return
        
        while (self._effective_context_tokens() > self.history_token_budget and
               len(self.conversation_history) > min_tail + 1):
            removed = self.conversation_history.pop(1)
            self._total_history_tokens -= removed.get("_tokens", 0)
        
        self._inject_checkpoint(checkpoint_summary)

    async def _compact_conversation(self):
        """Compact old messages into a structured summary.

        This is Agent's key context management innovation:
        - Old messages are replaced by a compact summary (~200 tokens)
        - Summary preserves: objective, decisions, findings, unresolved items
        - The agent never loses critical context, even in 200-iteration conversations

        Tiered (E.1): heuristic extraction by default; one cheap LLM call
        when `agent.compaction_llm` is enabled (degrades to heuristic on
        any failure).
        """
        min_tail = 8
        compact_up_to = len(self.conversation_history) - min_tail
        
        if compact_up_to <= 2:
            return
        
        if (len(self.conversation_history) > 1 and
                (self.conversation_history[1].get("content") or "").startswith("[COMPACTED")):
            keep_first = 2
        else:
            keep_first = 1
        
        if compact_up_to <= keep_first:
            return
        
        from core.config import app_config
        if app_config.get_bool("agent.compaction_llm", True):
            summary = await CompactionEngine.compact_llm(
                self.conversation_history, compact_up_to, keep_first,
                self.llm_client,
                on_usage=self._record_usage,
            )
        else:
            summary = CompactionEngine.compact(
                self.conversation_history, compact_up_to, keep_first
            )

        if not summary:
            return

        summary += ("\n[Note: older file contents and tool outputs were "
                    "compacted away — re-read a file before editing it if "
                    "its content is not in recent context.]")

        if self._context_strategy() != "legacy":
            # cache_first never injects checkpoints separately (that would
            # mutate the history head) — fold the latest checkpoint into the
            # one compaction message instead.
            cp = self.checkpoint_mgr.get_checkpoint_summary()
            if cp:
                summary = f"{summary}\n{cp}"

        removed_tokens = sum(
            m.get("_tokens", 0)
            for m in self.conversation_history[keep_first:compact_up_to]
        )
        del self.conversation_history[keep_first:compact_up_to]
        self._total_history_tokens -= removed_tokens
        
        compaction_msg = {
            "role": "user",
            "content": summary,
        }
        compaction_msg["_tokens"] = estimate_tokens(summary)
        self.conversation_history.insert(1, compaction_msg)
        self._total_history_tokens += compaction_msg["_tokens"]
        
        self._compaction_count += 1
        self._log("SYSTEM", "COMPACTION",
                  f"Compaction #{self._compaction_count}: replaced {compact_up_to - keep_first} messages "
                  f"(saved ~{removed_tokens - compaction_msg['_tokens']} tokens)")

    def _drop_by_role_filter(self, min_tail: int, should_drop):
        """Drop messages matching a filter, preserving first and last N messages."""
        i = 1
        boundary = len(self.conversation_history) - min_tail
        
        while i < boundary and self._effective_context_tokens() > self.history_token_budget:
            if should_drop(self.conversation_history[i]):
                removed = self.conversation_history.pop(i)
                self._total_history_tokens -= removed.get("_tokens", 0)
                boundary -= 1
            else:
                i += 1

    def _inject_checkpoint(self, checkpoint_summary: str):
        """Inject checkpoint context after task anchor if available."""
        if checkpoint_summary and len(self.conversation_history) > 1:
            inject_at = 1
            if (self.conversation_history[1].get("content") or "").startswith("[COMPACTED"):
                inject_at = 2

            if (inject_at < len(self.conversation_history) and
                    (self.conversation_history[inject_at].get("content") or "").startswith("[CHECKPOINT")):
                old_tokens = self.conversation_history[inject_at].get("_tokens", 0)
                self.conversation_history[inject_at]["content"] = f"[CHECKPOINT CONTEXT]\n{checkpoint_summary}"
                new_tokens = estimate_tokens(self.conversation_history[inject_at]["content"])
                self.conversation_history[inject_at]["_tokens"] = new_tokens
                self._total_history_tokens += (new_tokens - old_tokens)
            else:
                msg = {
                    "role": "user",
                    "content": f"[CHECKPOINT CONTEXT]\n{checkpoint_summary}"
                }
                msg["_tokens"] = estimate_tokens(msg["content"])
                self.conversation_history.insert(inject_at, msg)
                self._total_history_tokens += msg["_tokens"]
        
        self._log("SYSTEM", "PRUNING",
                  f"History pruned to {len(self.conversation_history)} messages "
                  f"(~{self._estimate_history_tokens()} tokens)")


    def _log(self, source: str, step_type: str, content: str, tool_calls: list = None):
        """Log to transcript if logger is available."""
        if self.transcript_logger:
            self.transcript_logger.log_step(source, step_type, content, tool_calls)


    async def _check_auto_memory(self, user_input: str):
        """Auto-store user corrections and preferences as memories.

        Two tiers (roadmap B.3):
        - Substring signals decide WHETHER this turn is a memory candidate
          (cheap pre-filter, unchanged).
        - When `agent.memory_llm_salience` is on, one cheap-model call
          distills the candidate into a clean, self-contained fact (or
          rejects it) instead of storing the raw truncated message.
        """
        if not self.memory_manager:
            return

        lower = user_input.lower().strip()

        # Machine-originated turns (cron missions, agent task prompts, system
        # kicks) are not user speech: their imperative wording ("never do X
        # twice") false-positives the correction signals, firing a salience
        # LLM call per cron run and storing prompt fragments as fake "user
        # corrections". Claude Code has no per-message memory LLM call at
        # all — memory writes are model-initiated.
        if lower.startswith(("[bot context", "[agent:", "[system]", "[cron")):
            return
        # Synthetic identities are dunder-prefixed (__cli__, __system__,
        # __cron__, __agent__<name>) — verified live: the prefix gate alone
        # missed factory agents because the runner strips "[AGENT:...]" from
        # the task text before the loop sees it.
        _uid = str(getattr(self.user_context, "user_id", "") or "")
        if _uid.startswith(("__", "agent_", "cron", "bot_")):
            return

        matched = (
            any(lower.startswith(p) for p in _CORRECTION_PREFIX_ONLY)
            or any(lower.startswith(s) or f" {s}" in lower
                   for s in _CORRECTION_SIGNALS)
        )
        if not matched:
            return
        try:
            value = await self._distill_memory(user_input)
            if value is None:
                return
            import hashlib
            content_hash = hashlib.md5(value.encode()).hexdigest()[:8]
            self.memory_manager.remember(
                key=f"correction.{content_hash}",
                value=value,
                user_id=self.user_context.user_id,
                category="correction",
                source="auto_memory",
            )
            self._log("SYSTEM", "AUTO_MEMORY",
                      f"Auto-stored: {value[:100]}")
        except Exception as e:
            self._log("SYSTEM", "AUTO_MEMORY_ERROR", str(e))

    async def _distill_memory(self, user_input: str) -> Optional[str]:
        """Return the memory value to store, or None to skip.

        Default (salience off): the legacy raw form. With salience on, a
        cheap-model call extracts one durable preference/fact — failures
        degrade to the legacy form, never to a lost memory.
        """
        legacy = f"User preference: {user_input[:200]}"
        try:
            from core.config import app_config
            if not app_config.get_bool("agent.memory_llm_salience", False):
                return legacy
        except Exception:
            return legacy
        try:
            from core.model_routing import resolve_model
            resp = await self.llm_client.generate(
                ("Extract the single durable user preference, correction, or fact "
                 "from this message as ONE self-contained sentence (e.g. 'User "
                 "prefers tabs over spaces'). It must still be true and useful "
                 "weeks from now. If the message is only situational/contextual "
                 "(a one-off instruction for the current task), reply exactly: "
                 "NOT_MEMORABLE"),
                [{"role": "user", "content": user_input[:1000]}],
                # Profile small_model wins over the config-routed cheap
                # model: a session pinned to a GLM profile must not route
                # aux calls to another tenant-level provider.
                model=(getattr(self.llm_client, "profile_small_model", None)
                       or resolve_model("salience")),
                stream=False,
            )
            self._record_usage(resp.get("usage"), resp.get("served_model"))
            text = ((resp or {}).get("content") or "").strip()
            if not text or "NOT_MEMORABLE" in text.upper():
                return None
            return text[:300]
        except Exception:
            return legacy

    @staticmethod
    def _resume_enabled() -> bool:
        try:
            from core.config import app_config
            return app_config.get_bool("agent.session_resume", True)
        except Exception:
            return True

    def export_history(self) -> List[Dict[str, Any]]:
        """The exact replayable message list (volatile token cache stripped;
        `_thinking` / `_is_error` kept — they're needed for faithful replay).

        `_attachments` (base64 image/doc bytes) is EXCLUDED from the snapshot: it
        would bloat the resume store by megabytes per turn, and it is re-hydrated
        from ag_system.files on load (web) or is transient (channels)."""
        return [{k: v for k, v in m.items() if k not in ("_tokens", "_attachments")}
                for m in self.conversation_history]

    def restore_history(self, messages: List[Dict[str, Any]]):
        """Rehydrate a persisted conversation (process restart / session
        eviction). Token counts are recomputed; pairing is repaired in case
        the snapshot was taken mid-mutation."""
        self.conversation_history = []
        self._total_history_tokens = 0
        for m in messages or []:
            if not isinstance(m, dict) or not m.get("role"):
                continue
            m = dict(m)
            m.pop("_tokens", None)
            self._append_message(m)
        self._repair_tool_pairing()
        self._log("SYSTEM", "RESUME",
                  f"Restored {len(self.conversation_history)} messages")

    def _persist_history_async(self):
        """Fire-and-forget end-of-turn snapshot (fail-soft, off-loop)."""
        if not (self._resume_enabled() and self.history_store.available):
            return
        snapshot = self.export_history()
        try:
            asyncio.get_running_loop().create_task(
                asyncio.to_thread(self.history_store.save, snapshot))
        except RuntimeError:
            try:
                self.history_store.save(snapshot)
            except Exception:
                pass

    @staticmethod
    def _nudges_enabled() -> bool:
        """`agent.behavior_nudges` (default ON).

        The passivity corrector and continuation nudge are SELF-LIMITING:
        on frontier models they essentially never fire (the doctrine keeps
        behavior correct, false-positive closers are filtered), so their
        cost is ~zero; on mid-tier models (e.g. deepseek-flash) they are
        the difference between finishing a task and narrating then
        stalling. Same design as the todo staleness reminder. Config
        exists to disable per tenant.
        """
        try:
            from core.config import app_config
            return app_config.get_bool("agent.behavior_nudges", True)
        except Exception:
            return True

    def _detect_passivity(self, message: str) -> bool:
        """Detect if the model is being passive — refusing without trying tools.

        Returns True if the message contains refusal language, meaning
        the agent should be nudged to use tools instead of giving up.
        """
        lower = message.lower()
        return any(signal in lower for signal in _PASSIVITY_SIGNALS)

    def _looks_like_intent_to_continue(self, message: str) -> bool:
        """True if the assistant ANNOUNCED a next action but emitted no tool call
        — e.g. "Let me read the runner file:", "Now I'll clean the gateway:".

        Used only when finish_reason=stop with no tool calls, to distinguish a
        mid-task narration (the model should have acted) from a genuine final
        answer or a clarifying question. Conservative: requires a forward-looking
        signal AND that the message isn't a question, so we never re-prompt a
        real answer into a loop (and the nudge itself offers a clean exit)."""
        if not message:
            return False
        text = message.strip()
        ends_with_colon = text.endswith(":")
        lower = text.lower()
        for neg in _CONTINUATION_NEGATIVE:
            lower = lower.replace(neg, "")
        has_signal = any(sig in lower for sig in _CONTINUATION_SIGNALS)
        if not (ends_with_colon or has_signal):
            return False
        if text.endswith("?"):
            return False
        return True

    def _available_tools_hint(self) -> str:
        """List the ACTUAL registered tool names for the passivity nudge.

        The old nudge hardcoded 'filesystem, email, Jira, calendar, shell…'.
        For persona-filtered bots that only expose a few tools, that lied to
        the model — it would try to call tools it doesn't have, burning
        iterations on 'tool not registered' errors. Deriving the real names
        keeps the nudge accurate to THIS agent's toolset.
        """
        try:
            names = list(getattr(self.tool_registry, "_tools", {}).keys())
        except Exception:
            names = []
        if not names:
            return "You have tools available in this session."
        shown = ", ".join(names[:25])
        if len(names) > 25:
            shown += f", … (+{len(names) - 25} more)"
        return f"You have access to these tools: {shown}."


    def _parse_xml_tool_calls(self, text: str):
        """Fallback: parse XML-style tool calls (for non-FC models only)."""
        tool_calls = []
        errors = []
        
        matches = re.finditer(r'<call:([\w]+)>(.*?)</call:\1>', text, re.DOTALL)
        for match in matches:
            tool_name = match.group(1)
            raw_json = match.group(2).strip()
            raw_json = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_json, flags=re.MULTILINE).strip()
            try:
                args = json.loads(raw_json)
                tool_calls.append({"name": tool_name, "args": args})
            except json.JSONDecodeError as e:
                errors.append(f"JSON error in '{tool_name}': {e}")
        
        clean = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
        clean = re.sub(r'<call:[\w]+>.*?</call:[\w]+>', '', clean, flags=re.DOTALL)
        return clean.strip(), tool_calls, errors


    async def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Execute a single tool via the ToolExecutor.

        The ToolExecutor decides WHERE to run it:
        - LocalToolExecutor: runs in a thread on the VPS (Web Chat, CLI, Cloud mode)
        - RemoteToolExecutor: routes LOCAL tools to VS Code via WebSocket
        """
        try:
            result = await self.tool_executor.execute(tool_name, tool_args)
            result = str(result) if result is not None else "Done."
            if self._is_failure_result(tool_name, result):
                self._consecutive_failures[tool_name] = \
                    self._consecutive_failures.get(tool_name, 0) + 1
            else:
                self._consecutive_failures[tool_name] = 0
            return result
        except Exception as e:
            self._consecutive_failures[tool_name] = \
                self._consecutive_failures.get(tool_name, 0) + 1
            args_summary = str(tool_args)[:200]
            return f"Error in {tool_name}({args_summary}): {e}"

    _GATE_BLOCK_PREFIXES = (
        "[GUARDRAIL BLOCKED]", "[APPROVAL DENIED]", "[ACCESS DENIED]",
        "[DUPLICATE BLOCKED]",
    )

    _PAGE_INTERACTION_TOOLS = frozenset({"page_act", "page_navigate"})
    _SOFT_FAILURE_MARKERS = (
        "not found", "no element", "could not find", "couldn't find",
        "unable to", "no such element", "timed out", "timeout",
        "does not exist", "no matching",
    )

    def _is_failure_result(self, tool_name: str, result: str) -> bool:
        """True if a tool result should count as a failure for loop-breaking.

        Catches the executor's "Error …" strings, the "[… BLOCKED/DENIED]" gate
        messages, and the soft failures element-interaction page tools return as
        ordinary text.
        """
        if result.startswith(("Error executing", "Error:", "Error in")):
            return True
        if result.startswith(self._GATE_BLOCK_PREFIXES):
            return True
        if tool_name in self._PAGE_INTERACTION_TOOLS:
            low = result.lower()
            return any(m in low for m in self._SOFT_FAILURE_MARKERS)
        return False

    @staticmethod
    def _parse_tool_args(tc: dict):
        """Parse tool-call args defensively.

        The model can emit invalid JSON in `arguments` — that must become a
        tool-level error string, never an exception that aborts the turn.
        Returns (args_dict, error_or_None).
        """
        args = tc.get("args")
        if isinstance(args, dict):
            return args, None
        try:
            parsed = json.loads(args) if args else {}
            if not isinstance(parsed, dict):
                return {}, f"Error: tool arguments must be a JSON object, got {type(parsed).__name__}"
            return parsed, None
        except (json.JSONDecodeError, TypeError) as e:
            return {}, f"Error: invalid JSON in tool arguments for '{tc.get('name')}': {e}"

    def _is_parallel_safe(self, tool_name: str) -> bool:
        """Read-class tools may run concurrently; anything that mutates state
        must not race other calls in the same batch."""
        try:
            pol = self.tool_registry.get_policy(tool_name)
            return getattr(pol, "side_effect", "") == "read"
        except Exception:
            return False

    async def _execute_tools_parallel(self, tool_calls: list) -> list:
        """Execute a batch of tool calls (the Claude Code concurrency rule).

        Read-only calls run CONCURRENTLY via asyncio.gather — reading three
        files takes one round-trip. Mutating/irreversible calls run
        SEQUENTIALLY in emission order: two parallel edits to the same file
        (or two sends) are a race the model cannot reason about. Results are
        returned in the original call order regardless of scheduling.
        """
        async def _run(tc):
            args, err = self._parse_tool_args(tc)
            if err:
                return err
            return await self._execute_tool(tc["name"], args)

        if len(tool_calls) == 1:
            return [await _run(tool_calls[0])]

        results: list = [None] * len(tool_calls)
        read_idx = [i for i, tc in enumerate(tool_calls)
                    if self._is_parallel_safe(tc.get("name", ""))]
        read_set = set(read_idx)

        if read_idx:
            gathered = await asyncio.gather(
                *(_run(tool_calls[i]) for i in read_idx),
                return_exceptions=True,
            )
            for i, r in zip(read_idx, gathered):
                results[i] = (str(r) if not isinstance(r, Exception)
                              else f"Error: {r}")

        for i in range(len(tool_calls)):
            if i in read_set:
                continue
            try:
                results[i] = await _run(tool_calls[i])
            except Exception as e:
                results[i] = f"Error: {e}"
        return results


    async def _process_input(self, payload: dict):
        """Execute a single turn of the agent.
        
        Flow (identical to Agent):
        1. OBSERVING → receive input, check auto-memory triggers
        2. PLANNING → classify effort, inject memories
        3. REASONING → call LLM with native function calling
        4. VALIDATING → check finish_reason
        5. EXECUTING → run tool calls in parallel, feed results back
        6. Loop 3-5 until finish_reason == "stop"
        7. RESPONDING → yield final text to user
        """
        if self._processing_lock.locked():
            # A turn is in flight — this message STEERS it (Claude Code
            # behavior: typing mid-turn redirects the agent) instead of
            # waiting to start a second turn afterwards.
            text = (payload or {}).get("user_input", "")
            if text:
                self.queue_steering_message(text)
                self._log("USER", "STEERING_QUEUED", text[:200])
            return

        sem = _turn_semaphore()
        if sem is not None and sem.locked():
            try:
                from core.telemetry import telemetry
                telemetry.incr("agent_turns_queued_total")
            except Exception:
                pass

        async def _guarded():
            async with self._processing_lock:
                from state.todos import current_todo_store
                from agents.task_manager import current_conversation_bus
                from tools.skills_mcp import current_skill_tracker
                _uctx_token = current_user_context.set(self.user_context)
                _todo_token = current_todo_store.set(self.todos)
                _bus_token = current_conversation_bus.set(self.event_bus)
                _skill_token = current_skill_tracker.set(self._skills_in_play)
                _tenant_token = (set_tenant_context(self.tenant_context)
                                 if self.tenant_context is not None else None)
                try:
                    await self._process_input_inner(payload)
                except Exception as e:
                    self._log("SYSTEM", "AGENT_ERROR", f"{type(e).__name__}: {e}")
                    logger.exception("agent_loop turn failed: %s", e)
                    try:
                        await self.event_bus.publish('agent_response', {
                            "message": ("⚠ Something went wrong while processing your "
                                        "request. Please try again in a moment."),
                            "error": True,
                        })
                    except Exception:
                        pass
                    self.state.current_state = AgentState.IDLE
                finally:
                    self._persist_history_async()
                    current_user_context.reset(_uctx_token)
                    current_todo_store.reset(_todo_token)
                    current_conversation_bus.reset(_bus_token)
                    current_skill_tracker.reset(_skill_token)
                    if _tenant_token is not None:
                        reset_tenant_context(_tenant_token)

        if sem is None:
            await _guarded()
        else:
            async with sem:
                await _guarded()

        if self._steering_queue:
            # Steering that arrived too late for the finished turn becomes
            # ordinary follow-up turns instead of being silently dropped.
            leftovers, self._steering_queue = self._steering_queue[:], []
            for text in leftovers:
                await self.event_bus.publish('user_input', {"user_input": text})

    def _recent_context_for_scope(self, turns: int = 4, cap: int = 800) -> str:
        """A compact tail of the conversation for the input scope rail, so a
        legitimate follow-up ("yes", "the red truck") is not misread as an
        off-topic request. Excludes the just-appended current user message."""
        msgs = [m for m in self.conversation_history[:-1]
                if m.get("role") in ("user", "assistant")
                and isinstance(m.get("content"), str) and m.get("content").strip()]
        lines = [f"{m['role']}: {m['content']}" for m in msgs[-turns:]]
        ctx = "\n".join(lines)
        return ctx[-cap:] if len(ctx) > cap else ctx

    async def _process_input_inner(self, payload: dict):
        """Inner implementation of _process_input (called under lock)."""
        user_input = payload.get("user_input", "")
        # Multimodal attachments (Part 1: user → model). Each is a hydrated
        # {kind, media_type, data_b64, filename} dict; they ride on the message's
        # `_attachments` sibling field so `content` stays a str for the scope
        # rail / titling / memory / compaction. An image-only turn (no text) is
        # valid, so attachments alone are enough to proceed.
        attachments = payload.get("attachments") or []
        if not user_input and not attachments:
            return

        if self.state.current_state != AgentState.IDLE:
            self.state.current_state = AgentState.IDLE
        self.state.transition_to(AgentState.OBSERVING)
        self._log("USER", "USER_INPUT", user_input or f"[{len(attachments)} attachment(s)]")
        _user_msg = {"role": "user", "content": user_input}
        if attachments:
            _user_msg["_attachments"] = attachments
        self._append_message(_user_msg)

        self._observe_review_signals(user_input)
        await self._check_auto_memory(user_input)

        # ── INPUT SCOPE RAIL (pre-generation topic gate) ────────────────────
        # The structural control that keeps a scoped persona (e.g. an embedded
        # product copilot) on its domain. Runs BEFORE any model generation, so
        # an out-of-scope request costs one cheap classify call and never
        # reaches the main model — no code written, no essay, no off-brand
        # answer. Active only when the persona carries scope_policy.mode=restrict;
        # unscoped personas pay nothing. See guardrails.GuardrailEngine.
        if self.guardrail_engine and self.guardrail_engine.has_input_scope:
            allowed, refusal = await self.guardrail_engine.evaluate_input(
                user_input, self._recent_context_for_scope())
            if not allowed:
                self._log("GUARDRAIL", "INPUT_SCOPE_BLOCK", user_input[:200])
                # Early pre-generation exit: keep history coherent (user →
                # assistant refusal), emit the reply, and end the turn. We
                # reset() rather than transition_to(RESPONDING) because the
                # legal graph has no OBSERVING→RESPONDING edge — reset() is the
                # state machine's "turn complete → IDLE" contract.
                self._append_message({"role": "assistant", "content": refusal})
                await self.event_bus.publish('agent_response', {"message": refusal})
                try:
                    from core.telemetry import telemetry
                    telemetry.incr("agent_scope_blocks_total")
                except Exception:
                    pass
                self.state.reset()
                return

        self._passivity_corrections = 0
        self._continuation_nudges = 0
        self._interaction_fail_cycles = 0
        self._cancel_requested = False
        self._turn_tool_counts: Dict[str, int] = {}
        self._turn_sig_counts: Dict[str, int] = {}
        
        self.state.transition_to(AgentState.PLANNING)
        effort = classify_effort(user_input)
        
        if self._cached_system_prompt is not None:
            # Explicit full override (agent factory / API layer preset).
            system_prompt = self._cached_system_prompt
        else:
            # cache_first: ONE byte-stable prompt for the whole session (built
            # with the COMPLEX superset of sections) — swapping prompt text
            # between turns would invalidate the provider's cached prefix,
            # which costs far more than the extra planning section.
            # legacy: cache per effort level.
            if self._context_strategy() != "legacy":
                prompt_key, prompt_effort = "stable", ReasoningEffort.COMPLEX
            else:
                prompt_key, prompt_effort = effort, effort
            system_prompt = self._effort_prompts.get(prompt_key)
            if system_prompt is None:
                constraints_override = self._prompt_overrides.get("constraints") or ""
                if self.guardrail_engine and self.guardrail_engine.has_input_guardrail:
                    guardrail_block = self.guardrail_engine.get_input_guardrail_prompt()
                    constraints_override = (
                        f"{constraints_override}\n\n"
                        f"[SECURITY BOUNDARY — NON-NEGOTIABLE]\n{guardrail_block}"
                    ).strip()

                # Note: copilot page-navigation prompt content is NOT injected
                # here — the core engine is surface-agnostic. The copilot
                # session builder (api layer) registers its playbook/routes via
                # prompt_builder.set_section(). See api/services/copilot_prompt.
                _tool_names = getattr(self.tool_registry, "_tools", {})
                self.prompt_builder.set_available_tools(_tool_names.keys())

                system_prompt = self.prompt_builder.build_system_prompt(
                    prompt_effort,
                    custom_identity=self._prompt_overrides.get("identity"),
                    custom_constraints=constraints_override or None,
                    custom_communication=self._prompt_overrides.get("communication"),
                )
                self._effort_prompts[prompt_key] = system_prompt
        
        if self.memory_manager and effort != ReasoningEffort.TRIVIAL:
            try:
                from core.config import app_config
                _mem_k = app_config.get_int("agent.memory_top_k", 5) or 5
                _mem_cap = app_config.get_int("agent.memory_context_max_chars", 2400) or 2400
            except Exception:
                _mem_k, _mem_cap = 5, 2400
            try:
                # Vector/lexical recall is a synchronous ES/pg round-trip —
                # keep it off the event loop (contextvars, incl. the tenant
                # binding, propagate through to_thread).
                # app_id is not optional here even though the parameter has a
                # default. THIS is the injection path — what comes back is
                # pasted into the prompt without anybody choosing it — and it
                # was the actual leak: a service key spends as the human who
                # minted it, so an app-7 turn ran as "user_1" and, with no app
                # filter, read that person's private memories. Observed live:
                # a named individual's profile and a stale "prefers no
                # calculus" correction injected into a Class 12 Integral
                # Calculus brief.
                memory_context = await asyncio.to_thread(
                    self.memory_manager.get_context_memories,
                    user_input, user_id=self.user_context.user_id, top_k=_mem_k,
                    visible_scopes=self.user_context.visible_scopes,
                    app_id=self.user_context.app_id,
                )
            except TypeError:
                # An older store that cannot scope by application. Fine when
                # this turn has no application to scope to; when it HAS one,
                # falling back would silently return everything — the exact
                # leak — so it returns nothing instead. Fail closed.
                if self.user_context.app_id is not None:
                    logger.warning(
                        "memory: store cannot scope by app_id; skipping context "
                        "injection for app %s rather than reading unscoped.",
                        self.user_context.app_id)
                    memory_context = ""
                else:
                    memory_context = await asyncio.to_thread(
                        self.memory_manager.get_context_memories,
                        user_input, top_k=_mem_k,
                    )
            if memory_context and len(memory_context) > _mem_cap:
                memory_context = memory_context[:_mem_cap] + "\n…[memory context truncated]"
            if memory_context:
                if self._context_strategy() == "legacy":
                    # legacy: replace the previous memory block in place.
                    for idx in range(len(self.conversation_history) - 1, -1, -1):
                        old = self.conversation_history[idx]
                        if (old.get("role") == "user" and
                                str(old.get("content", "")).startswith("[CONTEXT — Relevant memories]")):
                            self._total_history_tokens -= old.get("_tokens", 0)
                            self.conversation_history.pop(idx)
                    self._append_message({
                        "role": "user",
                        "content": f"[CONTEXT — Relevant memories]\n{memory_context}"
                    })
                elif memory_context != self._last_memory_context:
                    # cache_first: suffix-only. Deleting the previous block
                    # would mutate mid-history and kill the prompt cache;
                    # duplicates are skipped and compaction absorbs the rest.
                    self._last_memory_context = memory_context
                    self._append_message({
                        "role": "user",
                        "content": f"[CONTEXT — Relevant memories]\n{memory_context}"
                    })
        
        if self._openai_tools_cache is None:
            self._openai_tools_cache = self.tool_registry.get_openai_tools()
        openai_tools = self._select_turn_tools(user_input)

        await self._maybe_plan(effort, user_input)

        self.state.transition_to(AgentState.PRUNING)
        await self._prune_history()

        max_iterations = self.MAX_TOOL_ITERATIONS
        try:
            from core.config import app_config
            override = app_config.get_int("agent.max_tool_iterations", 0)
            if override and override > 0:
                max_iterations = min(max_iterations, override)
        except Exception:
            pass

        iteration = 0

        _loop = asyncio.get_running_loop()

        def _token_cb(kind: str, text: str):
            topic = "agent_thinking" if kind == "reasoning" else "agent_token"
            try:
                asyncio.run_coroutine_threadsafe(
                    self.event_bus.publish(topic, {"delta": text}), _loop)
            except Exception:
                pass

        _stream_tokens = not (
            self.guardrail_engine and self.guardrail_engine.has_output_guardrail
        )
        _active_token_cb = _token_cb if _stream_tokens else None

        while iteration < max_iterations:
            iteration += 1

            self._drain_steering()
            if self._cancel_requested:
                await self._finish_cancelled(iteration, effort)
                return

            if iteration > 1 and self._conversation_budget_exceeded():
                self._log("SYSTEM", "BUDGET_EXCEEDED",
                          f"conversation token budget reached "
                          f"({self._conversation_tokens_used} tokens)")
                await self.event_bus.publish('agent_response', {
                    "message": ("⚠ This conversation reached its token budget. "
                                "Start a new conversation to continue."),
                    "budget_exceeded": True,
                })
                self._record_turn_metrics(iteration, effort)
                self.state.current_state = AgentState.IDLE
                return

            # Host-scope budget (per-user/app/tenant, injected by the API
            # layer). Soft like the conversation cap: in-flight work lands,
            # then the turn ends with the host's message (limit + reset time)
            # instead of dying mid-tool.
            if iteration > 1 and self._budget_check is not None:
                try:
                    # The probe hits the host's DB (spend windows) — run it
                    # off the event loop so a slow query can't stall every
                    # concurrent turn in the process.
                    _budget_msg = await asyncio.to_thread(self._budget_check)
                except Exception:
                    _budget_msg = None
                if _budget_msg:
                    self._log("SYSTEM", "USER_BUDGET_EXCEEDED", _budget_msg)
                    await self.event_bus.publish('agent_response', {
                        "message": f"⚠ {_budget_msg}",
                        "budget_exceeded": True,
                    })
                    self._record_turn_metrics(iteration, effort)
                    self.state.current_state = AgentState.IDLE
                    return

            if (effort == ReasoningEffort.COMPLEX and iteration > 1
                    and iteration % self.REFLECTION_INTERVAL == 0):
                from core.planner import build_reflection_nudge
                self._append_message({"role": "user", "content": build_reflection_nudge("interval")})
                self._log("SYSTEM", "REFLECTION", f"Injected interval reflection at iter {iteration}")
            
            self.state.transition_to(AgentState.REASONING)
            
            response = await self.llm_client.generate(
                system_prompt,
                self.conversation_history,
                tools=openai_tools,
                token_callback=_active_token_cb,
            )

            self._record_usage(response.get("usage"),
                               response.get("served_model"))

            if response.get("error"):
                # LLM failure — end the turn WITHOUT recording the error text
                # as an assistant answer: it would pollute history/compaction
                # and read like a real reply. The user's message stays, so a
                # simple retry works.
                err_text = response.get("content") or "LLM call failed"
                self._log("SYSTEM", "LLM_ERROR", err_text)
                await self.event_bus.publish('agent_response', {
                    "message": f"⚠ {err_text}",
                    "error": True,
                })
                self._record_turn_metrics(iteration, effort)
                self.state.current_state = AgentState.IDLE
                return

            self.state.transition_to(AgentState.VALIDATING)

            finish_reason = response.get("finish_reason", "stop")
            content = response.get("content")
            tool_calls = response.get("tool_calls", [])
            
            self._log("MODEL", "LLM_RESPONSE", content or "(tool calls only)",
                      tool_calls if tool_calls else None)
            
            
            if finish_reason == "stop" or (not tool_calls and content):
                message = content or ""
                
                _guardrail_active = (
                    self.guardrail_engine is not None
                    and self.guardrail_engine.has_input_guardrail
                )
                _nudges_on = self._nudges_enabled()
                if (_nudges_on
                        and self._passivity_corrections < _MAX_PASSIVITY_CORRECTIONS
                        and not self._prompt_overrides.get("disable_passivity", False)
                        and not _guardrail_active
                        and self._detect_passivity(message)):
                    self._passivity_corrections += 1
                    self._log("SYSTEM", "PASSIVITY_CORRECTION",
                              f"Correction #{self._passivity_corrections}: model refused without trying tools")
                    self._append_message({"role": "assistant", "content": message})
                    self._append_message({
                        "role": "user",
                        "content": (
                            "[SYSTEM] Do NOT say you cannot do something. "
                            f"{self._available_tools_hint()} "
                            "Use the appropriate tool to act on the request. "
                            "If unsure which tool, try the most relevant one."
                        )
                    })
                    continue

                if (_nudges_on
                        and self._continuation_nudges < _MAX_CONTINUATION_NUDGES
                        and not tool_calls
                        and not _guardrail_active
                        and self._looks_like_intent_to_continue(message)):
                    self._continuation_nudges += 1
                    self._log("SYSTEM", "CONTINUATION_NUDGE",
                              f"Nudge #{self._continuation_nudges}: model narrated a next step but called no tool")
                    self._append_message({"role": "assistant", "content": message})
                    self._append_message({
                        "role": "user",
                        "content": (
                            "[SYSTEM] You described the next step but did not call a "
                            "tool, so nothing happened. If more work remains, call the "
                            "appropriate tool NOW to perform that step. If the task is "
                            "actually complete, reply with your final summary and do "
                            "not describe any further actions."
                        )
                    })
                    continue

                self.state.transition_to(AgentState.RESPONDING)
                if self.guardrail_engine and self.guardrail_engine.has_output_guardrail:
                    message = await self.guardrail_engine.apply_output_filter_async(message)
                self._log("MODEL", "FINAL_RESPONSE", message)
                self._append_message({"role": "assistant", "content": message})
                await self.event_bus.publish('agent_response', {"message": message})
                self._record_turn_metrics(iteration, effort)
                self.state.transition_to(AgentState.IDLE)
                return
            
            if finish_reason == "length":
                self.state.transition_to(AgentState.RESPONDING)
                message = (content or "") + "\n[Response truncated due to length limit]"
                if self.guardrail_engine and self.guardrail_engine.has_output_guardrail:
                    message = await self.guardrail_engine.apply_output_filter_async(message)
                self._append_message({"role": "assistant", "content": message})
                await self.event_bus.publish('agent_response', {"message": message})
                self._record_turn_metrics(iteration, effort)
                self.state.transition_to(AgentState.IDLE)
                return
            
            if not tool_calls:
                self.state.transition_to(AgentState.RESPONDING)
                msg = content or "(No response)"
                if self.guardrail_engine and self.guardrail_engine.has_output_guardrail:
                    msg = await self.guardrail_engine.apply_output_filter_async(msg)
                self._append_message({"role": "assistant", "content": msg})
                await self.event_bus.publish('agent_response', {"message": msg})
                self._record_turn_metrics(iteration, effort)
                self.state.transition_to(AgentState.IDLE)
                return
            
            assistant_msg = {"role": "assistant", "content": content}
            if response.get("thinking_blocks"):
                # Native Anthropic path: raw thinking blocks are replayed
                # verbatim on later iterations so the model keeps its
                # reasoning chain across tool calls (signatures intact).
                assistant_msg["_thinking"] = response["thinking_blocks"]
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{iteration}_{i}"),
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]) if isinstance(tc["args"], dict) else tc["args"]
                    }
                }
                for i, tc in enumerate(tool_calls)
            ]
            self._append_message(assistant_msg)
            
            if content:
                await self.event_bus.publish('agent_activity', {"label": content})
            
            loop_stop = self._loop_guard_tripped(tool_calls)
            if loop_stop:
                self._log("SYSTEM", "LOOP_GUARD", loop_stop)
                self._append_message({
                    "role": "user",
                    "content": (f"[SYSTEM] {loop_stop} Do NOT call that tool again. "
                                "Summarize what you have so far and respond to the user.")
                })
                if assistant_msg.get("tool_calls"):
                    _old_tokens = assistant_msg.get("_tokens", 0)
                    del assistant_msg["tool_calls"]
                    assistant_msg["content"] = assistant_msg.get("content") or "(stopped: repeated tool call)"
                    assistant_msg["_tokens"] = self._compute_msg_tokens(assistant_msg)
                    # Keep the running total honest — the message shrank
                    # after it was appended.
                    self._total_history_tokens += assistant_msg["_tokens"] - _old_tokens
                continue

            if self._cancel_requested:
                await self._finish_cancelled(iteration, effort)
                return

            self.state.transition_to(AgentState.EXECUTING)
            self._log("MODEL", "TOOL_CALLS", json.dumps([tc["name"] for tc in tool_calls]))

            for tc in tool_calls:
                args, _err = self._parse_tool_args(tc)
                await self.event_bus.publish('tool_start', {
                    "tool_name": tc["name"],
                    "tool_args": args
                })
            
            results = await self._execute_tools_parallel(tool_calls)
            
            for i, (tc, result) in enumerate(zip(tool_calls, results)):
                tool_name = tc["name"]
                call_id = tc.get("id", f"call_{iteration}_{i}")
                
                self._log("SYSTEM", f"TOOL_{tool_name.upper()}", str(result)[:2000])
                self.checkpoint_mgr.record_tool_call(tool_name, str(result)[:100])
                
                await self.event_bus.publish('tool_end', {
                    "tool_name": tool_name
                })
                
                from core.injection_guard import guard_tool_result
                guarded = guard_tool_result(tool_name, str(result))
                if guarded is not result and len(guarded) > len(str(result)):
                    self._log("SYSTEM", "INJECTION_FLAGGED",
                              f"tool={tool_name} (security framing injected)")

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": self._cap_tool_result(tool_name, guarded)
                }
                if self._is_failure_result(tool_name, str(result)):
                    # Structured error flag — the native Anthropic adapter
                    # maps this to tool_result is_error (the trained signal)
                    # instead of relying on string sniffing.
                    tool_msg["_is_error"] = True
                self._append_message(tool_msg)

            await self._publish_citations(tool_calls)

            for tc in tool_calls:
                fail_count = self._consecutive_failures.get(tc["name"], 0)
                if fail_count < self.MAX_CONSECUTIVE_FAILURES:
                    continue
                self._consecutive_failures[tc["name"]] = 0

                if tc["name"] in self._PAGE_INTERACTION_TOOLS:
                    self._interaction_fail_cycles += 1
                    if self._interaction_fail_cycles >= 2:
                        self._log("SYSTEM", "INTERACTION_GIVEUP",
                                  f"{tc['name']} unresolvable after repeated tries — "
                                  f"terminating turn")
                        await self.event_bus.publish('agent_response', {
                            "message": (
                                "I wasn't able to interact with those fields on this "
                                "screen. Could you fill them in directly? Once that's "
                                "done, tell me and I'll continue from there."),
                            "interaction_failed": True,
                        })
                        self._record_turn_metrics(iteration, effort)
                        self.state.current_state = AgentState.IDLE
                        return

                from core.planner import build_reflection_nudge
                self._append_message({
                    "role": "user",
                    "content": (
                        f"[SYSTEM] Tool '{tc['name']}' has failed "
                        f"{fail_count} consecutive times. "
                        + build_reflection_nudge("failure")
                    )
                })
                self._log("SYSTEM", "REFLECTION", f"Failure reflection for {tc['name']}")
            
            if self.checkpoint_mgr.should_checkpoint():
                cp = self.checkpoint_mgr.create_checkpoint()
                self._log("SYSTEM", "CHECKPOINT",
                          f"Checkpoint at iteration {cp['iteration']}")

            self._maybe_todo_reminder()
            
            if self._effective_context_tokens() > self._context_trigger_tokens():
                self.state.transition_to(AgentState.PRUNING)
                await self._prune_history()

            tool_names = ', '.join(tc['name'] for tc in tool_calls)
            logger.debug("Executing tools: %s (iter %d, ~%d tok ctx)",
                         tool_names, iteration, self._effective_context_tokens())
        
        self.state.transition_to(AgentState.RESPONDING)
        safety_msg = f"[Agent reached {max_iterations} iteration limit for this task]"
        self._log("SYSTEM", "MAX_ITERATIONS", safety_msg)
        await self.event_bus.publish('agent_response', {"message": safety_msg})
        self.state.transition_to(AgentState.IDLE)


    def _drain_steering(self):
        """Append queued mid-turn user messages (suffix-only, cache-safe)."""
        while self._steering_queue:
            text = self._steering_queue.pop(0)
            self._log("USER", "STEERING", text)
            self._append_message({
                "role": "user",
                "content": ("[USER MESSAGE — arrived mid-task; adjust your "
                            f"work accordingly]\n{text}"),
            })

    async def _finish_cancelled(self, iteration: int, effort):
        """End the turn cleanly on user cancellation."""
        self._cancel_requested = False
        self._repair_tool_pairing()
        self._log("SYSTEM", "CANCELLED",
                  f"turn stopped by user at iteration {iteration}")
        await self.event_bus.publish('agent_response', {
            "message": "⏹ Stopped. Tell me how you'd like to proceed.",
            "cancelled": True,
        })
        self._record_turn_metrics(iteration, effort)
        self.state.current_state = AgentState.IDLE

    def _observe_review_signals(self, user_input: str):
        """Watch the turn for learning-review triggers (engine observes,
        host decides): user corrections, explicit remember-requests. Machine
        turns never signal — same synthetic-identity gates as auto-memory."""
        self._skills_in_play.clear()
        self._turn_review_signals = {}
        low = (user_input or "").lower().strip()
        _uid = str(getattr(self.user_context, "user_id", "") or "")
        if (low.startswith(("[bot context", "[agent:", "[system]", "[cron"))
                or _uid.startswith(("__", "agent_", "cron", "bot_"))):
            return
        self._turn_review_signals = {
            "correction": (
                any(low.startswith(p) for p in _CORRECTION_PREFIX_ONLY)
                or any(low.startswith(s) or f" {s}" in low
                       for s in _CORRECTION_SIGNALS)),
            "remember": ("remember this" in low or "remember that" in low
                         or "save this" in low),
        }

    def seed_skills_in_play(self, names) -> None:
        """Declare skills that are available to this agent at creation time.

        Called by the factory when skills are injected into the system
        prompt (passive usage). This is separate from skill_view() tracking
        (active usage) — both feed into collect_review_signals().

        The factory is the right caller: it knows which skills it injected.
        The engine doesn't parse its own prompt to discover them.
        """
        self._skills_in_play.update(names)

    def collect_review_signals(self) -> Dict[str, Any]:
        """Post-turn probe for the host's learning loop.

        The engine only OBSERVES; the host owns policy, model choice, budget
        and the reviewer spawn (api layer learning_service). should_review is
        a convenience aggregate — hosts may apply stricter policy on top.
        """
        try:
            from core.config import app_config
            min_iters = app_config.get_int("agent.review_min_iterations", 8)
        except Exception:
            min_iters = 8
        sig = dict(self._turn_review_signals or {})
        sig["iterations"] = self._last_turn_iterations
        sig["skills_in_play"] = sorted(self._skills_in_play)
        sig["should_review"] = bool(
            sig.get("correction") or sig.get("remember")
            or (min_iters > 0 and sig["iterations"] >= min_iters)
            or sig["skills_in_play"])
        return sig

    def _record_turn_metrics(self, iterations: int, effort):
        """Record per-turn observability (best-effort, never raises)."""
        self._last_turn_iterations = iterations
        try:
            from core.telemetry import telemetry
            telemetry.observe("agent_turn_iterations", iterations,
                              effort=getattr(effort, "value", str(effort)))
            telemetry.incr("agent_turns_total",
                           effort=getattr(effort, "value", str(effort)))
        except Exception:
            pass


    DEFAULT_TOOL_RESULT_MAX_CHARS = 30_000

    def _cap_tool_result(self, tool_name: str, content: str) -> str:
        """Cap a tool result at insertion time (the Claude Code model).

        Oversized results must never enter history in full: they are billed
        on EVERY subsequent iteration of the turn, and the recent-message
        tail is unprunable. Head+tail are kept with an explicit affordance —
        the model re-invokes the tool with narrower arguments (line ranges,
        filters, grep) when it needs the omitted middle.
        """
        try:
            from core.config import app_config
            cap = app_config.get_int("agent.tool_result_max_chars",
                                     self.DEFAULT_TOOL_RESULT_MAX_CHARS)
        except Exception:
            cap = self.DEFAULT_TOOL_RESULT_MAX_CHARS
        if cap <= 0 or len(content) <= cap:
            return content
        head = content[: int(cap * 0.75)]
        tail = content[-int(cap * 0.15):]
        omitted = len(content) - len(head) - len(tail)
        self._log("SYSTEM", "RESULT_CAPPED",
                  f"tool={tool_name} capped {len(content)}→{cap} chars")
        return (f"{head}\n…[{omitted} chars omitted — output capped at {cap} "
                f"chars; call {tool_name} again with narrower arguments "
                f"(line range / filter / more specific query) to see the "
                f"omitted part]…\n{tail}")

    def _select_turn_tools(self, user_input: str) -> list:
        """Pick the tools array for this turn.

        cache_first: the tools array is part of the provider's cached prefix
        (it precedes system + messages), so it must be BYTE-STABLE across
        calls. A large-but-stable array is cheaper than a small-but-changing
        one: cached input is ~10% of full price, while every subset change
        re-bills system + tools + the entire history.
          - catalog ≤ agent.stable_tools_max → send the FULL set every call.
          - larger (hundreds of dynamic API ops) → grow-only session subset:
            keyword/embedding selection may ADD tools (one cache miss when
            they first appear) but never removes them, so the prefix is
            stable between growth events. Ordering follows the registry, so
            an unchanged set serializes identically.

        legacy: fresh per-turn subset (original behavior — cheapest single
        request, but defeats prefix caching across turns).
        """
        all_tools = self._openai_tools_cache or []
        try:
            from core.config import app_config
            if not app_config.get_bool("agent.dynamic_tools", True):
                return all_tools
            from core.tool_selection import select_tool_names
            names = {t["function"]["name"] for t in all_tools}
            reg_tools = getattr(self.tool_registry, "_tools", {})
            dynamic_tools = {n for n, sch in reg_tools.items()
                             if isinstance(sch, dict) and sch.get("json_schema")}

            cache_first = self._context_strategy() != "legacy"
            if cache_first:
                stable_max = app_config.get_int("agent.stable_tools_max", 80) or 80
                if len(all_tools) <= stable_max:
                    return all_tools

            keep = select_tool_names(user_input, names,
                                     dynamic_tools=dynamic_tools,
                                     dynamic_picker=self._dynamic_picker)
            if cache_first:
                # Grow-only AND append-only: new names go to the END of the
                # session order (registry-ordered among themselves for
                # determinism), keeping the already-cached schema prefix
                # byte-identical — a growth event only re-bills the new tail.
                registry_order = [t["function"]["name"] for t in all_tools]
                for name in registry_order:
                    if name in keep and name not in self._session_tool_names:
                        self._session_tool_names.add(name)
                        self._session_tool_order.append(name)
                by_name = {t["function"]["name"]: t for t in all_tools}
                subset = [by_name[n] for n in self._session_tool_order
                          if n in by_name]
            else:
                subset = [t for t in all_tools if t["function"]["name"] in keep]
            chosen = subset if subset else all_tools
            logging.getLogger("agent.tools").info(
                "tool_subset: sent=%d / total=%d%s", len(chosen),
                len(all_tools),
                " (session-sticky)" if cache_first else "")
            return chosen
        except Exception:
            return all_tools


    # Loop protection — the Claude Code model. What stops a runaway turn:
    #   1. IDENTICAL-INPUT detection (below): the same tool called with the
    #      same arguments repeatedly is the actual signature of a loop.
    #   2. User interruption (request_cancel / mid-turn steering).
    #   3. Budgets: the iteration cap and agent.conversation_token_budget.
    #   4. Blast radius for outward actions is the approval/guardrail layer's
    #      job, not loop detection's.
    # There is deliberately NO per-tool-name count cap: 15 distinct edits,
    # 20 distinct shell runs, or one memory_store per processed email are
    # NORMAL work, and prod showed such caps killing real turns mid-task.
    MAX_SAME_SIGNATURE_PER_TURN = 3
    _LOOP_GUARD_EXEMPT = frozenset({"page_observe", "page_read", "page_wait"})
    _PAGE_ACTION_TOOLS = frozenset({"page_act", "page_navigate"})
    MAX_PAGE_ACTIONS_PER_TURN = 14

    def _loop_guard_tripped(self, tool_calls: list) -> Optional[str]:
        """Record this batch's calls and return a stop reason if a loop is
        detected (identical-input repeats, or the embed surface's
        page-action bound)."""
        for tc in tool_calls:
            name = tc.get("name", "")
            if name in self._LOOP_GUARD_EXEMPT:
                continue
            self._turn_tool_counts[name] = self._turn_tool_counts.get(name, 0) + 1
            if name in self._PAGE_ACTION_TOOLS:
                page_total = sum(self._turn_tool_counts.get(t, 0)
                                 for t in self._PAGE_ACTION_TOOLS)
                if page_total >= self.MAX_PAGE_ACTIONS_PER_TURN:
                    return (f"{page_total} page actions this turn without "
                            f"completing the task — stopping to avoid a runaway "
                            f"navigation loop.")
            args, _ = self._parse_tool_args(tc)
            try:
                sig = f"{name}::{json.dumps(args, sort_keys=True, default=str)}"
            except Exception:
                sig = f"{name}::{args}"
            self._turn_sig_counts[sig] = self._turn_sig_counts.get(sig, 0) + 1
            if self._turn_sig_counts[sig] >= self.MAX_SAME_SIGNATURE_PER_TURN:
                return (f"Tool '{name}' was called with identical arguments "
                        f"{self._turn_sig_counts[sig]} times — stopping to avoid a loop.")
        return None

    TODO_STALE_ITERATIONS = 10

    def _maybe_todo_reminder(self):
        """Suffix-only reminder when the todo list has gone stale.

        The Claude Code mechanism: the model owns the plan via todo_write;
        the harness only nudges when items are open but the list hasn't
        changed for TODO_STALE_ITERATIONS tool iterations. Appending a
        message is cache-safe; it costs its own tokens and no extra call.
        """
        snapshot = self.todos.render()
        if not self.todos.items or self.todos.open_count == 0:
            self._todo_snapshot = snapshot
            self._todo_stale_iters = 0
            return
        if snapshot == self._todo_snapshot:
            self._todo_stale_iters += 1
        else:
            self._todo_snapshot = snapshot
            self._todo_stale_iters = 0
        if self._todo_stale_iters >= self.TODO_STALE_ITERATIONS:
            self._todo_stale_iters = 0
            self._append_message({
                "role": "user",
                "content": (
                    "[SYSTEM REMINDER] Your todo list has not changed in a "
                    "while. If items are actually done, mark them completed "
                    "via todo_write now; if the plan changed, revise the "
                    "list. Current list:\n" + snapshot
                ),
            })
            self._log("SYSTEM", "TODO_REMINDER", "stale todo list reminder injected")

    async def _maybe_plan(self, effort, user_input: str):
        """Generate and inject an execution plan for COMPLEX tasks.

        Legacy mechanism, default OFF since todo_write landed: the model now
        maintains its own live plan (no extra LLM call, revisable), which is
        the Claude Code model. Enable `agent.planning_enabled` to restore
        the one-shot planner call for models that need it.
        """
        if effort != ReasoningEffort.COMPLEX:
            return
        from core.config import app_config
        if not app_config.get_bool("agent.planning_enabled", False):
            return
        if user_input.strip().lower().startswith("[bot context"):
            return
        try:
            from core.planner import Planner
            planner = Planner(self.llm_client)
            steps, usage = await planner.make_plan(
                user_input, tools_hint=self._available_tools_hint()
            )
            self._record_usage(usage)
            if len(steps) >= 2:
                block = Planner.format_plan(steps)
                self._append_message({"role": "user", "content": block})
                self._log("SYSTEM", "PLAN", block)
                try:
                    await self.event_bus.publish('plan', {"steps": steps})
                except Exception:
                    pass
        except Exception as e:
            self._log("SYSTEM", "PLAN_ERROR", str(e))

    async def _publish_citations(self, tool_calls: list):
        """Emit a `citations` event with structured brain sources (N).

        After a brain tool runs, the BrainMCPTools instance holds the
        structured sources (doc title/path/snippet). Pop and publish them so
        the UI can render citation chips. Degrades silently if the brain
        instance isn't attached (e.g. no brain tools in this persona).
        """
        if not any(str(tc.get("name", "")).startswith("brain") for tc in tool_calls):
            return
        inst = getattr(self.tool_registry, "_brain_mcp_instance", None)
        popper = getattr(inst, "pop_sources", None)
        if not callable(popper):
            return
        try:
            sources = popper()
            if sources:
                await self.event_bus.publish('citations', {"sources": sources})
        except Exception as e:
            self._log("SYSTEM", "CITATIONS_ERROR", str(e))


    def generate_session_summary(self) -> dict:
        """Generate a structured summary of the current conversation.
        
        Called at session end for cross-session continuity.
        """
        if not self.conversation_history:
            return {}
        
        objective = ""
        for msg in self.conversation_history:
            if msg.get("role") == "user" and not msg.get("content", "").startswith("["):
                objective = msg["content"][:200]
                break
        
        outcome = ""
        for msg in reversed(self.conversation_history):
            if msg.get("role") == "assistant" and msg.get("content"):
                outcome = msg["content"][:200]
                break
        
        files = []
        if self.checkpoint_mgr.checkpoints:
            for cp in self.checkpoint_mgr.checkpoints:
                files.extend(cp.get("files_modified", []))
            files = list(set(files))
        
        return {
            "objective": objective,
            "outcome": outcome,
            "files_modified": files,
            "tool_calls_count": self.checkpoint_mgr._iteration_count,
            "messages_count": len(self.conversation_history),
        }
