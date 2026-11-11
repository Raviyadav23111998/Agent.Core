"""
Grant expansion at the persona/embed whitelist filter sites.
═══════════════════════════════════════════════════════════
The persona `allowed_tools` whitelist is exact-name and won't scale to N dynamic
ops. `effective_allowed_tools` expands an integration GRANT (ag_api.integration_grants)
to the concrete op tool_names the principal may use, so the existing exact-name
filters (chat_service.py / embed_sandbox.py) keep working unchanged.

Flag-gated: with `agent.dynamic_integrations` off this returns the base set
untouched, so callers are inert until a tenant opts in.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Set

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    try:
        from core.config import app_config
        return app_config.get_bool("agent.dynamic_integrations", False)
    except Exception:
        return False


def effective_allowed_tools(db, base_allowed: Iterable[str], *,
                            app_id: Optional[str] = None,
                            persona_id: Optional[str] = None,
                            embed_safe_only: bool = False) -> Set[str]:
    """base_allowed ∪ granted dynamic op tool_names for this principal.

    embed_safe_only=True (chatbot/embed) restricts to ops an admin marked
    embed-safe — internal business APIs are NOT exposed to anonymous visitors
    unless explicitly opted in. Best-effort: any failure returns base_allowed
    unchanged (fail-closed: a grant lookup error never widens access).
    """
    base = set(base_allowed or [])
    if not _enabled() or db is None:
        return base
    try:
        from core.integrations.store import IntegrationStore
        rows = IntegrationStore(db).catalog_for_principal(
            app_id=str(app_id or ""), persona_id=str(persona_id or ""),
            embed_safe_only=embed_safe_only)
        return base | {r["tool_name"] for r in rows}
    except Exception as e:
        logger.debug("effective_allowed_tools failed (returning base): %s", e)
        return base
