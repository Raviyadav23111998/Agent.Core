"""
Conversation compaction engine for Agent.

Identical to Agent's approach:
- When context hits ~80% of budget, compact old messages into a structured summary
- Summary preserves: objective, decisions, findings, unresolved issues, files touched
- Old messages are replaced with the compact summary (~200 tokens)
- This is fundamentally different from pruning: pruning deletes, compaction summarizes

The compaction summary is injected as a system message after the task anchor,
ensuring the agent never loses critical context even in 200-iteration conversations.
"""

import re
from typing import List, Dict, Any


_LLM_COMPACT_SYSTEM = (
    "You compress agent-conversation history into a continuation summary. "
    "The agent will resume work with ONLY your summary plus the last few "
    "messages — write what a competent colleague needs to CONTINUE, not a "
    "narrative of what happened. Structure (omit empty sections):\n"
    "1. Task: the user's goal and explicit requirements/constraints.\n"
    "2. Current state: what is DONE and verified vs in progress.\n"
    "3. Key decisions: choices made and why (max 6 bullets).\n"
    "4. Files & resources: each file/ticket/resource touched, with its "
    "relevant state (e.g. 'auth.py: added retry wrapper at login()').\n"
    "5. Learnings: discovered facts, gotchas, failed approaches NOT to retry.\n"
    "6. Errors & fixes: problems hit and how they were resolved.\n"
    "7. Pending: remaining todo items / unresolved issues.\n"
    "8. Next step: the single concrete action to take next.\n"
    "Be dense and specific (names, paths, ids, exact values). No preamble, "
    "no commentary. Stay under 400 words."
)


class CompactionEngine:
    """Generates structured compaction summaries from conversation history.

    Two modes:
    1. Heuristic extraction (default) — fast, zero LLM cost, works offline
    2. LLM-powered (compact_llm) — escalation tier for dense COMPLEX
       sessions; one cheap call produces a higher-fidelity summary, and any
       failure degrades back to the heuristic.

    The heuristic mode extracts structured data directly from message content
    using pattern matching. It's surprisingly effective because tool results
    and assistant messages follow predictable formats.
    """
    
    _DECISION_SIGNALS = [
        r"(?:i(?:'ll| will)|let(?:'s| us)|going to|decided to|choosing|using|switching to)\s+(.{10,80})",
        r"(?:changed|updated|modified|refactored|replaced|fixed)\s+(.{10,80})",
    ]
    
    _UNRESOLVED_SIGNALS = [
        r"(?:todo|fixme|hack|workaround|temporary|later|still need|not yet|pending)\s*:?\s*(.{10,80})",
        r"(?:failed|error|couldn't|can't|unable to)\s+(.{10,60})",
    ]
    
    _FILE_PATTERN = re.compile(
        r'(?:file:///|/[\w/.-]+\.\w{1,6}|[\w.-]+\.(?:py|js|ts|md|yaml|json|toml|sh|css|html))'
    )
    
    @classmethod
    def compact(
        cls,
        conversation_history: List[Dict[str, Any]],
        compact_up_to: int,
        keep_first: int = 1,
    ) -> str:
        """Generate a compaction summary from messages [keep_first..compact_up_to].
        
        Args:
            conversation_history: Full conversation history
            compact_up_to: Index up to which messages will be compacted
            keep_first: Number of messages to preserve at the start (task anchor)
        
        Returns:
            Structured compaction summary string (~200 tokens)
        """
        if compact_up_to <= keep_first:
            return ""
        
        messages = conversation_history[keep_first:compact_up_to]
        if not messages:
            return ""
        
        objective = cls._extract_objective(conversation_history[:keep_first + 2])
        decisions = cls._extract_decisions(messages)
        findings = cls._extract_findings(messages)
        unresolved = cls._extract_unresolved(messages)
        files = cls._extract_files(messages)
        tool_count = sum(1 for m in messages if m.get("role") == "tool")
        turn_count = sum(1 for m in messages if m.get("role") == "user"
                        and not m.get("content", "").startswith("["))
        
        parts = [f"[COMPACTED CONTEXT — {len(messages)} messages, {tool_count} tool calls, {turn_count} user turns]"]
        
        if objective:
            parts.append(f"Objective: {objective}")
        
        if decisions:
            parts.append("Decisions made:")
            for d in decisions[:5]:
                parts.append(f"  - {d}")
        
        if findings:
            parts.append("Key findings:")
            for f in findings[:5]:
                parts.append(f"  - {f}")
        
        if unresolved:
            parts.append("Unresolved:")
            for u in unresolved[:3]:
                parts.append(f"  - {u}")
        
        if files:
            parts.append(f"Files touched: {', '.join(sorted(files)[:10])}")
        
        return "\n".join(parts)
    
    @classmethod
    async def compact_llm(
        cls,
        conversation_history: List[Dict[str, Any]],
        compact_up_to: int,
        keep_first: int,
        llm_client: Any,
        on_usage: Any = None,
    ) -> str:
        """LLM compaction tier (roadmap E.1) — falls back to heuristic.

        Sends a truncated transcript digest to a (cheap, via model routing)
        model and returns its structured summary in the same [COMPACTED
        CONTEXT …] envelope the rest of the pipeline expects. Any failure or
        empty answer degrades to the heuristic `compact()`.

        on_usage(usage_dict, served_model) — the caller's accounting hook:
        compaction is real tenant spend and must land in the same ledger as
        every other call.
        """
        heuristic = lambda: cls.compact(conversation_history, compact_up_to, keep_first)
        if llm_client is None or compact_up_to <= keep_first:
            return heuristic()

        messages = conversation_history[keep_first:compact_up_to]
        if not messages:
            return ""

        parts = []
        total = 0
        for m in messages:
            content = (m.get("content") or "").strip()
            if not content and m.get("tool_calls"):
                content = "(called tools: " + ", ".join(
                    tc.get("function", {}).get("name", "?")
                    for tc in m["tool_calls"]) + ")"
            if not content:
                continue
            line = f"[{m.get('role', '?')}] {content[:400]}"
            parts.append(line)
            total += len(line)
            if total > 12_000:
                break
        if not parts:
            return heuristic()

        tool_count = sum(1 for m in messages if m.get("role") == "tool")
        try:
            from core.model_routing import resolve_model
            resp = await llm_client.generate(
                _LLM_COMPACT_SYSTEM,
                [{"role": "user", "content": "\n".join(parts)}],
                # Profile small_model wins over config routing — a session
                # pinned to a profile must not route aux calls elsewhere.
                model=(getattr(llm_client, "profile_small_model", None)
                       or resolve_model("compaction")),
                stream=False,
            )
            if on_usage is not None:
                try:
                    on_usage((resp or {}).get("usage"),
                             (resp or {}).get("served_model"))
                except Exception:
                    pass
            summary = ((resp or {}).get("content") or "").strip()
            if not summary:
                return heuristic()
            header = (f"[COMPACTED CONTEXT — {len(messages)} messages, "
                      f"{tool_count} tool calls (LLM summary)]")
            return f"{header}\n{summary}"
        except Exception:
            return heuristic()

    @classmethod
    def _extract_objective(cls, first_messages: List[Dict]) -> str:
        """Extract the user's objective from the first user message."""
        for msg in first_messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content and not content.startswith("["):
                    first_sentence = content.split(".")[0].split("\n")[0]
                    return first_sentence[:150].strip()
        return ""
    
    @classmethod
    def _extract_decisions(cls, messages: List[Dict]) -> List[str]:
        """Extract decisions from assistant messages."""
        decisions = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "") or ""
            for pattern in cls._DECISION_SIGNALS:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    decision = match.group(0).strip()
                    if len(decision) > 15 and decision not in decisions:
                        decisions.append(decision[:100])
        return decisions
    
    @classmethod
    def _extract_findings(cls, messages: List[Dict]) -> List[str]:
        """Extract key findings from assistant messages (non-trivial observations)."""
        findings = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "") or ""
            for line in content.split("\n"):
                line = line.strip()
                if (line.startswith(("- ", "* ", "• ")) and len(line) > 20
                        and not line.startswith(("- [", "- `["))):
                    finding = line.lstrip("-*• ").strip()[:120]
                    if finding not in findings:
                        findings.append(finding)
        return findings[:8]
    
    @classmethod
    def _extract_unresolved(cls, messages: List[Dict]) -> List[str]:
        """Extract unresolved issues from conversation."""
        unresolved = []
        for msg in messages:
            content = msg.get("content", "") or ""
            for pattern in cls._UNRESOLVED_SIGNALS:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    issue = match.group(0).strip()[:100]
                    if issue not in unresolved:
                        unresolved.append(issue)
        return unresolved
    
    @classmethod
    def _extract_files(cls, messages: List[Dict]) -> set:
        """Extract file paths mentioned in messages."""
        files = set()
        for msg in messages:
            content = msg.get("content", "") or ""
            for match in cls._FILE_PATTERN.finditer(content):
                path = match.group(0)
                if path.startswith("file:///"):
                    path = path[7:]
                basename = path.split("/")[-1]
                if "." in basename and len(basename) < 60:
                    files.add(basename)
        return files
