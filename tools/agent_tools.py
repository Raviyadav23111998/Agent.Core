"""
Agent — Agent Management MCP Tools
===============================================
5 tools for creating and managing dynamic agents from chat:

- agent_create: Create a new autonomous agent from a description
- agent_list:   List all agents with status
- agent_run:    Trigger an agent manually
- agent_delete: Remove an agent
- agent_logs:   View recent run outputs
"""

import asyncio
import logging

from agents.registry import AgentDefinition
from core.tenant import TENANT_SCOPE

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run async code from sync context (tool callbacks are sync).
    
    FIX BUG 3: The previous implementation spawned a new thread with 
    asyncio.run() which created a SEPARATE event loop. This caused deadlocks
    because the factory's asyncio.Event and EventBus callbacks were bound to
    the MAIN event loop, not the thread's loop.
    
    Solution: Schedule the coroutine on the CURRENT running loop and block
    until it completes. This keeps all async primitives on the same loop.
    """
    try:
        loop = asyncio.get_running_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=660)
    except RuntimeError:
        return asyncio.run(coro)


_MODEL_ALIASES = {
    "flash": "deepseek/deepseek-v4-flash",
    "gemini-flash": "deepseek/deepseek-v4-flash",
    "pro": "google/gemini-2.5-pro",
    "gemini-pro": "google/gemini-2.5-pro",
    "sonnet": "anthropic/claude-sonnet-4",
    "claude-sonnet": "anthropic/claude-sonnet-4",
    "opus": "anthropic/claude-opus-4",
    "claude-opus": "anthropic/claude-opus-4",
    "haiku": "anthropic/claude-3.5-haiku",
    "claude-haiku": "anthropic/claude-3.5-haiku",
    "gpt4o": "openai/gpt-4o",
    "gpt-4o": "openai/gpt-4o",
    "gpt4o-mini": "openai/gpt-4o-mini",
    "o3": "openai/o3",
    "o3-mini": "openai/o3-mini",
    "deepseek": "deepseek/deepseek-r1",
    "deepseek-r1": "deepseek/deepseek-r1",
}


def _resolve_model_alias(model: str) -> str:
    """Resolve a model shorthand to its full OpenRouter path.
    
    Examples:
        "flash"    → "deepseek/deepseek-v4-flash"
        "sonnet"   → "anthropic/claude-sonnet-4"
        "gpt4o"    → "openai/gpt-4o"
        ""         → "" (use default)
        "anthropic/claude-sonnet-4" → "anthropic/claude-sonnet-4" (passthrough)
    """
    if not model:
        return ""
    model_lower = model.strip().lower()
    return _MODEL_ALIASES.get(model_lower, model.strip())

class AgentMCPTools:
    """MCP tool wrappers for the dynamic agent system."""

    def __init__(self, registry=None, factory=None, user_context_resolver=None):
        self._registry = registry
        self._factory = factory
        self._resolve_user = user_context_resolver

    def register_tools(self, mcp_server):
        registry = self._registry
        factory = self._factory
        resolve_user = self._resolve_user

        @mcp_server.tool()
        def agent_create(
            name: str,
            description: str,
            tools: str = "auto",
            schedule: str = "",
            workspace_path: str = "",
            model: str = "",
            visibility: str = "global",
        ) -> str:
            """Create a new autonomous agent from a natural language description.

            The agent will run headlessly with its own filtered tool set.
            Tools are auto-inferred from the description unless specified.
            Each agent can use a different LLM model for cost/quality tuning.

            Args:
                name: Agent name (e.g., "support-bot", "bug-fixer")
                description: What the agent should do (natural language)
                tools: Comma-separated tool names, or "auto" to infer from description
                schedule: Optional schedule ("every 30m", "every 1h", "0 9 * * *")
                workspace_path: Optional path to target codebase for code agents
                model: LLM model to use. Shortcuts: "flash", "pro", "sonnet", "opus", "gpt4o".
                       Or full model path like "anthropic/claude-sonnet-4". Empty = default.
                visibility: "global" (all users can see/run) or "private" (creator only)
            """
            if not registry or not factory:
                return "Agent system not initialized."

            if registry.exists(name):
                return (f"Agent '{name}' already exists. "
                        f"Delete it first with agent_delete, or use a different name.")

            resolved_model = _resolve_model_alias(model)

            if tools == "auto" or not tools:
                tool_list = factory.infer_tools(description)
                tool_source = "auto-inferred"
            else:
                tool_list = [t.strip() for t in tools.split(",") if t.strip()]
                tool_source = "manual"

            if not tool_list:
                return ("✗ Could not determine tools for this agent. "
                        "Try specifying tools manually: tools=\"brain_search,run_command\"")

            system_prompt = factory.generate_system_prompt(name, description, tool_list)

            owner_id = TENANT_SCOPE
            if resolve_user:
                ctx = resolve_user()
                if ctx:
                    owner_id = ctx.user_id

            agent_def = AgentDefinition(
                name=name,
                description=description,
                system_prompt=system_prompt,
                tools=tool_list,
                model=resolved_model,
                schedule=schedule,
                workspace_path=workspace_path,
                owner_id=owner_id,
                visibility=visibility,
            )

            registry.save(agent_def)

            cron_info = ""
            if schedule:
                try:
                    from cron.jobs import create_job, list_jobs
                    cron_name = f"agent_{name}"
                    existing = [j for j in list_jobs() if j.get("name") == cron_name]
                    if not existing:
                        create_job(
                            prompt=f"[AGENT:{name}] {description}",
                            schedule=schedule,
                            name=cron_name,
                        )
                        cron_info = f"\n  Schedule: {schedule}"
                except Exception as e:
                    cron_info = f"\n  ⚠ Schedule failed: {e}"

            model_info = f"\n  Model: {resolved_model}" if resolved_model else "\n  Model: default"
            lines = [
                f"✓ Created agent \"{name}\"",
                f"  Tools ({tool_source}): {', '.join(tool_list)}",
                model_info,
            ]
            if cron_info:
                lines.append(cron_info)
            lines.append(f"  Run: agent_run name=\"{name}\"")

            return "\n".join(lines)

        @mcp_server.tool()
        def agent_list() -> str:
            """List all created agents visible to the current user."""
            if not registry:
                return "Agent system not initialized."

            user_id = None
            if resolve_user:
                ctx = resolve_user()
                if ctx:
                    user_id = ctx.user_id
            
            if user_id:
                agents = registry.list_for_user(user_id)
            else:
                agents = registry.list_all()

            if not agents:
                return "No agents created yet. Use agent_create to create one."

            lines = [f"Agents ({len(agents)}):"]
            for a in agents:
                status_icon = "✓" if a.enabled else "⏸"
                last = a.last_run_at[:16] if a.last_run_at else "never"
                schedule = a.schedule or "manual"
                last_status = a.last_status or "-"

                lines.append(
                    f"\n  {status_icon} {a.name}"
                    f"\n    {a.description[:80]}"
                    f"\n    Tools: {len(a.tools)} | Model: {a.model or 'default'} | Schedule: {schedule}"
                    f"\n    Runs: {a.run_count} | Last: {last} ({last_status})"
                )

            return "\n".join(lines)

        @mcp_server.tool()
        def agent_run(name: str, task: str = "") -> str:
            """Manually trigger an agent to run.

            The agent executes headlessly with its configured tools and
            returns the result.

            Args:
                name: Agent name to run
                task: Optional task prompt override (default: agent's description)
            """
            if not registry or not factory:
                return "Agent system not initialized."

            agent_def = registry.load(name)
            if not agent_def:
                return f"Agent '{name}' not found. Use agent_list to see available agents."

            if not agent_def.enabled:
                return f"Agent '{name}' is disabled."

            try:
                run_result = _run_async(
                    factory.run_agent(agent_def, task_prompt=task or None)
                )
                return f"[Agent: {name}]\n{run_result.output}"
            except Exception as e:
                return f"✗ Agent '{name}' failed: {str(e)}"

        @mcp_server.tool()
        def agent_delete(name: str) -> str:
            """Delete an agent and its cron schedule.

            Args:
                name: Agent name to delete
            """
            if not registry:
                return "Agent system not initialized."

            if not registry.exists(name):
                return f"Agent '{name}' not found."

            try:
                from cron.jobs import list_jobs, remove_job
                cron_name = f"agent_{name}"
                for job in list_jobs():
                    if job.get("name") == cron_name:
                        remove_job(job["id"])
                        break
            except Exception:
                pass

            registry.delete(name)
            return f"✓ Deleted agent '{name}'"

        @mcp_server.tool()
        def agent_logs(name: str, count: int = 3) -> str:
            """View recent run logs for an agent.

            Args:
                name: Agent name
                count: Number of recent logs to show (default: 3)
            """
            if not registry:
                return "Agent system not initialized."

            if not registry.exists(name):
                return f"Agent '{name}' not found."

            logs = registry.get_logs(name, last_n=count)
            if not logs:
                return f"No run logs for agent '{name}' yet."

            return f"Recent logs for '{name}':\n\n" + "\n\n---\n\n".join(logs)
