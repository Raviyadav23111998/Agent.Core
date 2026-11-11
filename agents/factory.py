"""
Agent — Agent Factory
==================================
Spawns headless AgentLoop instances from AgentDefinitions.

Each spawned agent gets:
- Filtered ToolRegistry (only allowed tools)
- Custom system prompt (agent-specific role)
- Shared: LLM, Memory, Brain, EventBus
- Isolated: conversation history, checkpoints

The factory also handles tool auto-inference from descriptions.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.registry import AgentDefinition, AgentRegistry
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ── RunResult: the factory's observable output ──────────────────────────

_DIGEST_TAIL = 30       # messages to keep in the compact replay
_DIGEST_MSG_CHARS = 400 # per-message truncation


def _build_run_digest(conversation_history: list,
                      skills_in_play: list = None) -> str:
    """Compact conversation replay for the background reviewer.

    Pure function (no DB, no policy) — lives in core because it only
    reads the agent's conversation_history attribute, which is engine
    state.  The host never needs to build this itself.
    """
    parts = []
    for m in (conversation_history or [])[-_DIGEST_TAIL:]:
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        if not content and m.get("tool_calls"):
            content = "(called: " + ", ".join(
                tc.get("function", {}).get("name", "?")
                for tc in m["tool_calls"]) + ")"
        if content:
            parts.append(f"[{role}] {content[:_DIGEST_MSG_CHARS]}")
    digest = "\n".join(parts)
    header = "Conversation digest (most recent last):\n"
    if skills_in_play:
        header += ("Skills that were IN PLAY this conversation "
                   f"(prefer patching these): {', '.join(skills_in_play)}\n")
    return (header + digest)[:24_000]


@dataclass
class RunResult:
    """Observable outcome of a factory-spawned agent run.

    The factory collects everything the host needs to make policy
    decisions (learning, analytics, debugging) and packages it here.
    The host never needs to reach into the AgentLoop — the engine
    observes, the host decides.
    """
    output: str                                            # final response text
    review_signals: Dict[str, Any] = field(default_factory=dict)
    conversation_digest: str = ""                          # compact replay
    agent_name: str = ""
    run_kind: str = "subagent"
    status: str = "ok"                                     # ok | error | timeout

_TOOL_HINTS = {
    "code": ["view_file", "write_to_file", "replace_file_content",
             "multi_replace_file_content", "grep_search", "run_command", "list_dir"],
    "file": ["view_file", "write_to_file", "replace_file_content",
             "multi_replace_file_content", "list_dir"],
    "read": ["view_file", "list_dir"],
    "write": ["write_to_file", "replace_file_content", "multi_replace_file_content"],
    "edit": ["view_file", "replace_file_content", "multi_replace_file_content"],
    "debug": ["view_file", "grep_search", "run_command", "brain_search", "list_dir"],
    "fix": ["view_file", "write_to_file", "replace_file_content",
            "multi_replace_file_content", "grep_search", "run_command",
            "git_branch", "git_commit", "list_dir"],
    "test": ["run_command", "view_file", "grep_search"],
    "build": ["run_command", "view_file"],
    "deploy": ["run_command"],
    "command": ["run_command"],
    "script": ["run_command", "view_file", "write_to_file"],
    "directory": ["list_dir"],

    "git": ["git_branch", "git_commit", "git_status", "git_diff"],
    "branch": ["git_branch", "git_status"],
    "commit": ["git_commit", "git_status", "git_diff"],
    "diff": ["git_diff", "view_file"],
    "merge": ["git_branch", "git_status", "run_command"],

    "email": ["fetch_recent_emails", "search_emails", "brain_sync", "brain_search"],
    "outlook": ["fetch_recent_emails", "search_emails", "brain_sync",
                "outlook_reply", "outlook_send"],
    "mail": ["fetch_recent_emails", "search_emails", "brain_sync",
             "outlook_reply", "brain_search"],
    "reply": ["outlook_reply", "fetch_recent_emails", "search_emails"],
    "send": ["outlook_send"],
    "forward": ["outlook_forward"],
    "inbox": ["fetch_recent_emails", "search_emails"],

    "jira": ["jira_create_ticket", "jira_update_ticket", "jira_transition_ticket",
             "jira_add_comment", "jira_add_worklog", "jira_list_tickets",
             "brain_sync", "brain_search"],
    "ticket": ["jira_create_ticket", "jira_update_ticket", "jira_transition_ticket",
               "jira_add_comment", "jira_add_worklog", "jira_list_tickets"],
    "create ticket": ["jira_create_ticket"],
    "worklog": ["jira_add_worklog", "jira_list_tickets"],
    "log time": ["jira_add_worklog", "jira_list_tickets"],
    "transition": ["jira_transition_ticket", "jira_list_tickets"],
    "bug": ["jira_create_ticket", "jira_update_ticket", "jira_list_tickets",
            "view_file", "grep_search"],
    "issue": ["jira_create_ticket", "jira_update_ticket", "jira_transition_ticket",
              "jira_list_tickets"],

    "brain": ["brain_search", "brain_ask", "brain_sync"],
    "search": ["brain_search", "grep_search"],
    "sync": ["brain_sync"],
    "knowledge": ["brain_search", "brain_ask", "brain_ingest_text"],
    "ingest": ["brain_ingest_file", "brain_ingest_directory", "brain_ingest_text"],
    "slack": ["brain_sync", "brain_search"],
    "web": ["search_web"],

    "memory": ["memory_recall", "memory_store", "memory_stats"],
    "remember": ["memory_store", "memory_recall"],
    "context": ["memory_recall", "memory_user_context"],

    "schedule": ["schedule", "cron_list", "cron_remove"],
    "cron": ["schedule", "cron_list", "cron_remove"],
    "monitor": ["manage_task", "cron_list"],
    "background": ["manage_task", "run_command"],

    "agent": ["agent_create", "agent_list", "agent_run", "agent_delete"],
}

_AGENT_MISSION_TEMPLATE = """
<agent_mission>
You are "{name}", an autonomous AI agent.

Your mission: {description}

Your available tools: {tool_list}

Directives:
1. Execute your mission autonomously — do not ask questions.
2. Use your tools to gather information and take action.
3. If something fails, try alternative approaches before giving up.
4. You run HEADLESS: nobody reads text between tool calls. Produce no
   progress narration — only tool calls, then ONE final report.
5. If the mission's trigger condition is not met (nothing new to process,
   nothing to do), stop immediately with a one-line report saying so —
   do not run the rest of the mission "just in case".
6. Report results clearly and concisely when done.
</agent_mission>
"""


BUILTIN_AGENTS = {
    # Read-only investigator — the Claude Code "Explore" agent. The parent
    # delegates broad searches here so the raw reads/greps burn THIS
    # context window, not the parent's; only the report returns.
    "explore": AgentDefinition(
        name="explore",
        description=(
            "Read-only exploration agent. Investigates files, code, and "
            "knowledge sources to answer a specific question, then reports "
            "findings with concrete references (paths, line numbers, ids). "
            "It cannot modify anything. Its report must directly answer the "
            "brief it was given — conclusions first, evidence after."
        ),
        tools=[
            "view_file", "list_dir", "grep_search", "glob_files",
            "brain_search", "brain_ask", "search_web",
            "query_neo4j", "search_elasticsearch",
            "memory_recall", "get_memories",
        ],
    ),
    # Background learning reviewer — the hermes-agent loop on our substrate
    # (docs/SKILLS_LEARNING_IMPLEMENTATION_PLAN_2026-07-04.md). Spawned by
    # the api layer's learning_service post-turn with run_kind="review";
    # tool whitelist = skills + memory ONLY. Its system prompt is
    # prompts/reviewer.md, loaded at spawn (see _builtin_system_prompt).
    "reviewer": AgentDefinition(
        name="reviewer",
        description=(
            "Background learning reviewer. Reads a conversation digest and "
            "persists at most two durable lessons: a skill update (how to "
            "do a class of task) or a user memory. Replies 'Nothing to "
            "save.' when the session produced no lesson."
        ),
        tools=[
            "skills_list", "skill_view", "skill_manage",
            "memory_store", "memory_recall",
        ],
    ),
    # Library maintenance — runs as a system cron ([AGENT:curator]). May
    # also archive; never touches pinned or bundled skills (store-enforced).
    "curator": AgentDefinition(
        name="curator",
        description=(
            "Skill-library curator. Reviews the skill index for overlap, "
            "staleness, and bad effectiveness (high correction_after_use), "
            "then merges/patches/archives. Skills flagged in the task "
            "briefing get attention first. Never archives pinned or "
            "bundled skills. Ends with a one-line change log."
        ),
        tools=["skills_list", "skill_view", "skill_manage"],
    ),
}

_BUILTIN_PROMPT_FILES = {"reviewer": "reviewer.md", "curator": "curator.md"}


def _builtin_system_prompt(name: str) -> str:
    """Load a builtin agent's dedicated prompt file (cached), if any."""
    fname = _BUILTIN_PROMPT_FILES.get(name)
    if not fname:
        return ""
    try:
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "prompts", fname)
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def resolve_agent_definition(registry: AgentRegistry, name: str):
    """DB-registered definitions first, then builtins."""
    definition = None
    try:
        definition = registry.load(name)
    except Exception:
        definition = None
    return definition or BUILTIN_AGENTS.get((name or "").strip().lower())


class AgentFactory:
    """Spawns isolated AgentLoop instances from AgentDefinitions.

    Usage:
        factory = AgentFactory(cli_instance, registry)

        # Run an agent — returns RunResult with output, signals, digest
        run_result = await factory.run_agent(definition)
        print(run_result.output)  # the agent's response text

        # Infer tools from description
        tools = factory.infer_tools("sync emails and reply to questions")
        # → ["brain_sync", "brain_search", "outlook_reply", ...]
    """

    def __init__(self, cli_instance, registry: AgentRegistry):
        self._cli = cli_instance
        self._registry = registry

    def create_filtered_registry(self, allowed_tools: list) -> ToolRegistry:
        """Create a ToolRegistry containing only the specified tools."""
        filtered = ToolRegistry()
        master = self._cli.tool_registry

        for tool_name in allowed_tools:
            if tool_name in master._tools:
                filtered._tools[tool_name] = master._tools[tool_name]
                filtered._callbacks[tool_name] = master._callbacks[tool_name]
                if tool_name in master._policies:
                    filtered._policies[tool_name] = master._policies[tool_name]
            else:
                logger.debug("agent_factory.tool_not_found: %s", tool_name)

        return filtered

    async def run_agent(
        self,
        definition: AgentDefinition,
        task_prompt: str = None,
        timeout: int = 600,
        app_id: str = None,
        run_kind: str = "subagent",
    ) -> "RunResult":
        """Spawn and run a headless agent.

        Args:
            definition: The agent's configuration
            task_prompt: Override prompt (default: agent's description)
            timeout: Max execution time in seconds (default 10 min)
            app_id: Owning application id, so the agent authenticates with
                    that app's connections (e.g. an app-scoped Jira credential
                    for a cron-run agent). Defaults to the definition's app_id,
                    then the parent context's.
            run_kind: Usage-ledger attribution for this run — "subagent" when
                    spawned from a live conversation, "cron" when the host is
                    a scheduler (cron work billed as subagent hides the real
                    cost center in per-kind reports).

        Returns:
            RunResult carrying the output text, review signals, and a
            compact conversation digest. Callers that only need the
            text use ``result.output``.
        """
        from core.agent_loop import AgentLoop

        run_id = uuid.uuid4().hex[:12]

        registry = self.create_filtered_registry(definition.tools)
        tool_count = len(registry._tools)

        if tool_count == 0:
            # Every caller consumes a RunResult (.output/.run_kind); a bare
            # string here AttributeErrors the host's learning/cron hooks.
            return RunResult(
                output=f"✗ Agent '{definition.name}' has no valid tools. "
                       f"Check tool names.",
                agent_name=definition.name, run_kind=run_kind, status="error")

        logger.info("agent_factory.spawning: name=%s tools=%d run_id=%s",
                    definition.name, tool_count, run_id)

        workspace = definition.workspace_path or self._cli.workspace_path
        system_prompt = definition.system_prompt
        if not system_prompt:
            # Builtins with a dedicated prompt file (reviewer) run LEAN —
            # they don't need the full identity/doctrine stack, and a small
            # stable prompt keeps every review cheap.
            system_prompt = _builtin_system_prompt(definition.name)
        if not system_prompt:
            system_prompt = self._build_agent_system_prompt(
                definition.name,
                definition.description,
                list(registry._tools.keys()),
                workspace,
                app_id=app_id or getattr(definition, "app_id", None),
            )

        # ── Model identity: ONE resolution, same chain as every other
        # surface (definition override > owning app's binding > tenant
        # default > legacy shared client). The resolved profile also feeds
        # the usage sink so ledger rows carry the profile that served them.
        _scope = getattr(self._cli, "llm_scope", None)
        _sub_app_id = app_id or getattr(definition, "app_id", None)
        _profile = None
        if _scope is not None:
            try:
                _profile = _scope.resolve_profile(
                    app_id=_sub_app_id,
                    requested_model=definition.model or None)
            except Exception as e:
                logger.debug("agent_factory: profile resolve failed: %s", e)

        if definition.model:
            # A model override needs the profile's FULL identity — a
            # cross-provider override (deepseek tenant → glm sub-agent)
            # requires the profile's own api_key + base_url; reusing the
            # parent's key against another vendor's endpoint just 401s.
            llm_client = None
            if _profile:
                # Profiles exist (post-V035): the definition's model must
                # name one. A miss resolves to the app/default profile — use
                # it rather than blindly overriding the model string on
                # the tenant endpoint, which sends e.g. a claude-* name
                # to a DeepSeek URL: every call 400s into the fallback
                # model forever (2026-07-04 prod incident).
                if definition.model not in (_profile.get("model"),
                                            _profile.get("name")):
                    logger.warning(
                        "agent_factory: definition model %r names no "
                        "active profile — using profile %r (%s) instead",
                        definition.model, _profile.get("name"),
                        _profile.get("model"))
                llm_client = _scope.client_for(_profile, None, stream=False)
            if llm_client is None:
                # Pre-V035 legacy: same-connection model swap. Only safe for
                # sibling models of the SAME provider (deepseek-v4-flash →
                # deepseek-v4-pro), never cross-vendor names.
                from client.cli import create_llm_client
                llm_client = create_llm_client(
                    api_key=self._cli.llm.api_key,
                    model=definition.model,
                    stream=False,
                )
            logger.info("agent_factory.model_override: name=%s model=%s",
                        definition.name, definition.model)
        elif _profile is not None:
            # No model override: honor the OWNING APP's profile binding, the
            # same resolution every other surface applies (chat/bot/cron).
            # Using the legacy shared client here silently exempted spawned
            # agents from per-app model separation.
            llm_client = _scope.client_for(_profile, self._cli.llm,
                                           stream=False)
        else:
            llm_client = self._cli.llm

        from core.user_context import create_agent_user
        agent_user_ctx = create_agent_user(
            agent_name=definition.name,
            parent_context=getattr(self._cli, 'agent', None)
                          and getattr(self._cli.agent, 'user_context', None),
            app_id=app_id or getattr(definition, "app_id", None) or None,
        )

        # Sub-agent spend is real tenant spend: meter it into the ledger and
        # keep it under the initiating app's budget (when the host attached
        # its scope service).
        _usage_sink = None
        _budget_check = None
        if _scope is not None:
            try:
                # Attribute sub-agent spend to the INITIATING user too —
                # otherwise a user's chat could route unlimited work around
                # their budget via sub-agents (rows landed user_id=NULL).
                # The scope service normalizes 'user_5'/'cron_user'/... to a
                # numeric id or None. The profile resolved above rides along
                # so ledger rows carry the identity that served them.
                _parent_ctx = (getattr(self._cli, 'agent', None)
                               and getattr(self._cli.agent, 'user_context', None))
                _parent_uid = getattr(_parent_ctx, "user_id", None)
                _usage_sink = _scope.make_usage_sink(
                    user_id=_parent_uid, app_id=_sub_app_id,
                    run_kind=run_kind,
                    profile_id=(_profile or {}).get("profile_id"))
                _budget_check = _scope.make_budget_check(
                    user_id=_parent_uid, app_id=_sub_app_id)
            except Exception as e:
                logger.debug("agent_factory: usage scope skipped: %s", e)

        from async_bus.event_bus import EventBus
        agent_bus = EventBus()
        agent = AgentLoop(
            workspace_path=workspace,
            llm_client=llm_client,
            tool_registry=registry,
            event_bus=agent_bus,
            transcript_logger=(getattr(self._cli, 'logger', None)
                               or getattr(self._cli, 'transcript_logger', None)),
            memory_manager=self._cli.memory,
            user_context=agent_user_ctx,
            usage_sink=_usage_sink,
            budget_check=_budget_check,
        )

        agent._cached_system_prompt = system_prompt

        # NOTE: sub-agents do NOT receive the skills index in their prompt
        # (only the chat surface calls set_skills_index), and the turn-start
        # reset in _observe_review_signals clears any pre-seeded set. use_count
        # and skills_in_play therefore reflect ACTIVE engagement only (a
        # skill_view during the run) — never blanket "every skill exists"
        # bookkeeping, which would make the effectiveness signal meaningless.

        prompt = task_prompt or definition.description

        response_text = None
        response_event = asyncio.Event()

        async def capture(payload):
            nonlocal response_text
            if response_text is not None:
                return
            response_text = payload.get("message", "")
            response_event.set()

        agent_bus.subscribe('agent_response', capture)

        try:
            # wait_for must wrap the WHOLE turn — the turn runs inline here
            # (LLM calls + tools), so waiting only on the response event after
            # it returns would make the timeout a no-op.
            await asyncio.wait_for(
                agent._process_input({"user_input": prompt}), timeout=timeout
            )
            if not response_event.is_set():
                response_event.set()
        except asyncio.TimeoutError:
            response_text = f"⏰ Agent '{definition.name}' timed out after {timeout}s."
        except Exception as e:
            response_text = f"✗ Agent '{definition.name}' error: {str(e)}"
            logger.error("agent_factory.run_failed: name=%s run_id=%s error=%s",
                        definition.name, run_id, str(e))
        finally:
            agent_bus.unsubscribe('agent_response', capture)

        output = response_text or "(No output)"

        status = "ok" if not output.startswith("✗") and not output.startswith("⏰") else "error"
        self._registry.update_run_state(
            definition.name, status,
            error=output[:200] if status == "error" else ""
        )
        self._registry.save_run_log(definition.name, output)

        # ── Collect observable outcomes for the host ──────────────────
        # The agent is still in scope here. Collect review signals and
        # build a compact digest BEFORE the agent goes out of scope.
        # These are pure observations — no DB calls, no policy.
        signals: Dict[str, Any] = {}
        digest = ""
        try:
            signals = agent.collect_review_signals()
            skills_in_play = signals.get("skills_in_play") or []
            digest = _build_run_digest(
                agent.conversation_history, skills_in_play)
        except Exception:
            logger.debug("agent_factory: signal collection failed",
                         exc_info=True)

        logger.info("agent_factory.completed: name=%s run_id=%s status=%s output=%d chars",
                    definition.name, run_id, status, len(output))

        return RunResult(
            output=output,
            review_signals=signals,
            conversation_digest=digest,
            agent_name=definition.name,
            run_kind=run_kind,
            status=status,
        )

    def infer_tools(self, description: str) -> list:
        """Auto-select tools based on agent description keywords.

        Scans the description for keywords and maps them to relevant tools.
        Only returns tools that actually exist in the master registry.
        """
        desc_lower = description.lower()
        selected = set()

        for keyword, tools in _TOOL_HINTS.items():
            if keyword in desc_lower:
                selected.update(tools)

        selected.add("brain_search")

        master_tools = set(self._cli.tool_registry._tools.keys())
        valid_tools = selected & master_tools
        missing = selected - master_tools

        if missing:
            logger.debug("agent_factory.tools_not_available: %s",
                        ", ".join(sorted(missing)))

        return sorted(valid_tools)

    def _build_agent_system_prompt(
        self, name: str, description: str, tools: list, workspace: str,
        app_id=None,
    ) -> str:
        """Build a full-quality system prompt for a spawned agent.
        
        GAP FIX: Spawned agents previously got a 404-char skeleton prompt.
        The main Agent agent gets 1140+ chars with environment context,
        constraints, and communication style — all loaded from prompt files.
        
        Now spawned agents get the SAME prompt sections PLUS an agent-specific
        mission overlay. This makes them as capable as the main agent.
        
        Assembly order:
        1. Identity (from prompts/identity.md)
        2. Environment (computed — workspace, OS, shell)
        3. Constraints (from prompts/constraints.md)
        4. Communication style (from prompts/communication.md)
        5. Agent mission overlay (name, description, tool list)
        """
        from core.prompt_builder import PromptBuilder, ReasoningEffort

        pb = PromptBuilder(workspace)
        pb.set_available_tools(tools)
        # Spawned agents see the same procedural memory as sessions do —
        # scoped to their owning app (no user scope for headless runs).
        store = getattr(self._cli, "skill_store", None)
        if store is not None:
            try:
                _aid = int(app_id) if app_id is not None else None
            except (TypeError, ValueError):
                _aid = None
            try:
                from core.prompt_sections import (
                    render_skills_index, SKILLS_KEY, SKILLS_ORDER)
                pb.set_section(SKILLS_KEY,
                               render_skills_index(store.index_for(_aid, None)),
                               SKILLS_ORDER)
            except Exception:
                pass

        base_prompt = pb.build_system_prompt(ReasoningEffort.STANDARD)
        
        tool_list = ", ".join(tools)
        mission = _AGENT_MISSION_TEMPLATE.format(
            name=name,
            description=description,
            tool_list=tool_list,
        )
        
        return base_prompt + "\n" + mission

    def generate_system_prompt(
        self, name: str, description: str, tools: list
    ) -> str:
        """Generate a system prompt from name, description, and tool list.
        
        Public API for external callers.
        """
        workspace = self._cli.workspace_path
        return self._build_agent_system_prompt(name, description, tools, workspace)
