"""
Planner & Reflection — explicit reasoning scaffolds for the agent loop.
═══════════════════════════════════════════════════════════════════════

Two lightweight, cost-aware aids that lift accuracy on hard tasks without
turning every turn into a multi-call chain:

1. Planner.make_plan() — ONE cheap LLM call (COMPLEX tasks only) that turns
   the objective into a short, ordered checklist. The plan is injected into
   context so the model tracks progress instead of wandering.

2. build_reflection_nudge() — a zero-extra-call self-critique message injected
   at intervals or when the agent is failing repeatedly. The model reflects on
   its NEXT turn, so the only cost is a few injected tokens.

Both are intentionally tolerant: a parsing failure degrades to "no plan",
never an exception that breaks the loop.
"""

import re
import logging
from typing import Any, List

logger = logging.getLogger("agent.planner")


_PLAN_SYSTEM_PROMPT = (
    "You are a planning module for an autonomous agent. Given a task and the "
    "tools available, produce a SHORT ordered checklist (3-7 steps) of the "
    "concrete actions needed to complete it.\n"
    "RULES:\n"
    "- Each step is one line, imperative, starting with a verb.\n"
    "- Reference specific tools where obvious.\n"
    "- No prose, no preamble, no numbering style other than '1.' '2.' …\n"
    "- If the task is trivial, return a single step.\n"
    "Return ONLY the numbered checklist."
)


class Planner:
    """Generates a concise execution plan via one cheap LLM call."""

    MAX_STEPS = 7

    def __init__(self, llm_client: Any):
        self._llm = llm_client

    async def make_plan(self, objective: str, tools_hint: str = ""):
        """Return (steps, usage). steps may be empty on failure/degradation."""
        if not self._llm or not objective.strip():
            return [], {}
        user = (
            f"TASK:\n{objective.strip()}\n\n"
            f"AVAILABLE TOOLS: {tools_hint or '(general toolset)'}\n\n"
            "Produce the checklist."
        )
        try:
            from core.model_routing import resolve_model
            resp = await self._llm.generate(
                _PLAN_SYSTEM_PROMPT,
                [{"role": "user", "content": user}],
                model=resolve_model("planning"),
                stream=False,
            )
            content = (resp or {}).get("content") or ""
            usage = (resp or {}).get("usage") or {}
            steps = self.parse_steps(content)
            logger.info("planner.plan: steps=%d", len(steps))
            return steps, usage
        except Exception as e:
            logger.warning("planner.failed: %s", e)
            return [], {}

    @staticmethod
    def parse_steps(text: str) -> List[str]:
        """Extract numbered/bulleted steps from the model's plan output."""
        if not text:
            return []
        steps: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(?:\d+[.)]\s+|[-*•]\s+)(.*)$", line)
            if m:
                step = m.group(1).strip()
                if step:
                    steps.append(step)
        if not steps:
            steps = [l.strip() for l in text.splitlines() if l.strip()]
        return steps[: Planner.MAX_STEPS]

    @staticmethod
    def format_plan(steps: List[str]) -> str:
        """Render a plan as an injectable context block."""
        if not steps:
            return ""
        lines = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        return (
            "[PLAN — follow these steps; mark progress as you go]\n"
            f"{lines}\n"
            "Work through the steps in order. If a step proves wrong, adapt and say so."
        )


def build_reflection_nudge(reason: str = "interval") -> str:
    """A self-critique message that costs only its own tokens (no extra call)."""
    if reason == "failure":
        return (
            "[SYSTEM — REFLECT] Recent tool calls have failed repeatedly. "
            "Stop and reassess: Is the current approach working? Is there a "
            "different tool or strategy? State your revised approach in one "
            "sentence, then act on it."
        )
    return (
        "[SYSTEM — REFLECT] Pause and check progress against the plan: "
        "What is done, what remains, and are you still on the most direct path? "
        "If off-track, correct course now. Then continue."
    )
