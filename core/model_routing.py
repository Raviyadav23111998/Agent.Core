"""
Model routing — pick the right (cheap) model per call purpose.
═══════════════════════════════════════════════════════════════

Roadmap F.2: internal classification-style calls (brain query routing,
graph extraction, planning, memory salience, compaction summaries) don't
need the flagship model. Routing them to a small model cuts cost ~10-20x
on those calls with no user-visible quality loss.

Resolution order for a given purpose:
  1. llm.model.<purpose>   — explicit per-purpose override
  2. llm.small_model       — shared cheap model, used for CHEAP_PURPOSES only
  3. the caller's default  — usually llm.model (the flagship)

Unset configs mean every call keeps using the flagship model — this module
changes nothing until llm.small_model (or a per-purpose key) is configured.
"""

from typing import Optional

CHEAP_PURPOSES = frozenset({
    "routing",
    "extraction",
    "planning",
    "salience",
    "compaction",
})


def resolve_model(purpose: str, default: Optional[str] = None) -> Optional[str]:
    """Return the model to use for `purpose`, or `default` when unconfigured."""
    try:
        from core.config import app_config
        override = app_config.get(f"llm.model.{purpose}")
        if override:
            return override
        if purpose in CHEAP_PURPOSES:
            small = app_config.get("llm.small_model")
            if small:
                return small
    except Exception:
        pass
    return default
