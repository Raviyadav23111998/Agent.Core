"""
Per-tool role authorization (roadmap I.4).
═══════════════════════════════════════════

Gates individual tools behind a minimum auth role, on top of the persona's
`allowed_tools` (which controls VISIBILITY) and `execution_policies` (which
constrains ARGUMENTS). This adds the WHO dimension: a Viewer may chat and
read, but only a Member+ may send email or run shell, etc.

Model — a role hierarchy + per-tool minimum role, both config-driven:
  - `security.role_levels`  (JSON role_name → int level; higher = more power).
    Sensible defaults match the ag_auth.roles table.
  - `security.tool_min_role` (JSON tool_name → minimum role_name).
    EMPTY by default → no enforcement (fully backward compatible). A tool not
    in the map is unrestricted.

A user is allowed iff level(user_role) >= level(required_role). System /
automation contexts (cron, CLI, headless agents) bypass entirely — they have
no human role and are already trusted.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Tuple

logger = logging.getLogger("agent.tool_authz")

_DEFAULT_ROLE_LEVELS: Dict[str, int] = {
    "tenant owner": 100,
    "administrator": 80,
    "admin": 80,
    "member": 50,
    "service agent": 50,
    "viewer": 10,
    "guest": 1,
    "": 50,
}


def _cfg_json(key: str) -> dict:
    try:
        from core.config import app_config
        raw = app_config.get(key)
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def _role_levels() -> Dict[str, int]:
    levels = dict(_DEFAULT_ROLE_LEVELS)
    for k, v in _cfg_json("security.role_levels").items():
        try:
            levels[str(k).strip().lower()] = int(v)
        except (TypeError, ValueError):
            continue
    return levels


def role_level(role_name: str) -> int:
    levels = _role_levels()
    return levels.get((role_name or "").strip().lower(), levels.get("", 50))


def tool_min_roles() -> Dict[str, str]:
    """tool_name → required role_name. Empty dict = no enforcement."""
    return _cfg_json("security.tool_min_role")


def check_tool_role(tool_name: str, role_name: str) -> Tuple[bool, str]:
    """Return (allowed, reason). Default-open: unlisted tools are unrestricted."""
    reqs = tool_min_roles()
    if not reqs:
        return True, ""
    required = reqs.get(tool_name)
    if not required:
        return True, ""
    if role_level(role_name) >= role_level(required):
        return True, ""
    return False, (
        f"requires role '{required}' or higher; your role "
        f"'{role_name or 'unknown'}' is insufficient"
    )
