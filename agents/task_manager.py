"""Sub-agent execution manager — the Claude Code Task-tool model.
═══════════════════════════════════════════════════════════════════

Sub-agents are the token-isolation mechanism: broad exploration runs in an
agent with its OWN context window, and only the final report enters the
parent's context. Two execution modes:

  await run(...)   — foreground: the caller awaits the report (used by the
                     `delegate_task` tool; several tool calls in one batch
                     run as genuinely parallel sub-agents because the tool
                     callback is async and executes on the loop).
  spawn(...)       — background: returns a task id immediately; completion
                     is published as a `subagent_completed` event which the
                     host re-injects into the conversation (same pattern as
                     background shell tasks).

Concurrency is capped per event loop; delegation depth is tracked with a
ContextVar so it propagates naturally into nested sub-agents (and never
leaks across concurrent conversations the way a class attribute did).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

logger = logging.getLogger("agent.subagents")

MAX_DELEGATION_DEPTH = 2

delegation_depth: ContextVar[int] = ContextVar("delegation_depth", default=0)

# The bus of the CONVERSATION whose turn is currently executing (bound by
# AgentLoop next to user/tenant context). Background sub-agent completions
# must be delivered here — API sessions run one bus per conversation, so the
# manager's constructor bus (the host/CLI bus) would deliver the report to
# the wrong loop.
current_conversation_bus: ContextVar[Optional[Any]] = ContextVar(
    "current_conversation_bus", default=None)

_SEMAPHORES: Dict[int, asyncio.Semaphore] = {}


def _concurrency_cap() -> int:
    try:
        from core.config import app_config
        return app_config.get_int("agent.subagent_concurrency", 4) or 4
    except Exception:
        return 4


def _semaphore() -> asyncio.Semaphore:
    key = id(asyncio.get_running_loop())
    sem = _SEMAPHORES.get(key)
    if sem is None:
        sem = asyncio.Semaphore(_concurrency_cap())
        _SEMAPHORES[key] = sem
    return sem


class SubAgentManager:
    """Runs sub-agents for one host (CLI/session), bounded and observable."""

    DEFAULT_TIMEOUT_S = 600

    def __init__(self, event_bus: Any):
        self._bus = event_bus
        self.background: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def check_depth() -> Optional[str]:
        """Return a refusal string when the delegation depth is exhausted."""
        if delegation_depth.get() >= MAX_DELEGATION_DEPTH:
            return ("✗ Delegation depth limit reached — a sub-agent this deep "
                    "must complete the work directly instead of delegating "
                    "again.")
        return None

    async def run(self, factory: Any, definition: Any, task: str,
                  timeout_s: float = None) -> str:
        """Foreground sub-agent: await and return its report."""
        timeout_s = timeout_s or self.DEFAULT_TIMEOUT_S
        token = delegation_depth.set(delegation_depth.get() + 1)
        try:
            async with _semaphore():
                run_result = await asyncio.wait_for(
                    factory.run_agent(definition, task_prompt=task,
                                      timeout=int(timeout_s)),
                    timeout=timeout_s + 30,
                )
                return run_result.output
        except asyncio.TimeoutError:
            return (f"⏰ Sub-agent '{definition.name}' timed out after "
                    f"{int(timeout_s)}s.")
        finally:
            delegation_depth.reset(token)

    def spawn(self, factory: Any, definition: Any, task: str,
              timeout_s: float = None) -> str:
        """Background sub-agent: returns a task id; result arrives as a
        `subagent_completed` event on the SPAWNING conversation's bus."""
        task_id = f"sub_{uuid.uuid4().hex[:8]}"
        timeout_s = timeout_s or self.DEFAULT_TIMEOUT_S
        deliver_bus = current_conversation_bus.get() or self._bus

        async def _run():
            result = await self.run(factory, definition, task, timeout_s)
            entry = self.background.get(task_id)
            if entry is not None:
                entry["status"] = "completed"
                entry["result"] = result
            try:
                await deliver_bus.publish("subagent_completed", {
                    "task_id": task_id,
                    "agent_name": definition.name,
                    "result": result,
                })
            except Exception as e:
                logger.error("subagent completion publish failed: %s", e)

        # ContextVars (tenant/user/depth) are captured at task creation, so
        # the background agent inherits the spawning conversation's context.
        token = delegation_depth.set(delegation_depth.get())
        try:
            aio_task = asyncio.get_running_loop().create_task(_run())
        finally:
            delegation_depth.reset(token)
        self.background[task_id] = {
            "status": "running", "agent": definition.name,
            "task": task[:200], "aio": aio_task,
        }
        return task_id
