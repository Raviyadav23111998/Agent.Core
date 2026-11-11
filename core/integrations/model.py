"""
The normalized integration model — everything lowers to this.
═══════════════════════════════════════════════════════════════
OpenAPI, Swagger 2.0 (converted), and MCP tools all become an `Integration`
holding `IntegrationOperation`s. The runtime depends on NOTHING format-specific.

Also the single source of truth for the HTTP-method → side-effect / approval /
ToolPolicy mapping, so a dynamic DELETE is gated exactly like `outlook_send` by
the existing ToolExecutor (no name-pattern guessing — an explicit policy is
attached at registration). See plan §2.2 / §5.

Pure data + pure functions: no DB, no HTTP, no I/O. Safe to import anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tools.tool_policy import READ, WRITE, IRREVERSIBLE, ToolPolicy


@dataclass(frozen=True)
class IntegrationOperation:
    """One API endpoint → one agent tool.

    `parameters_schema` is a FULL JSON Schema (path+query+header+body merged),
    passed verbatim to the OpenAI function definition via ToolRegistry's
    json_schema pass-through (registry.py).
    """
    operation_id: str
    tool_name: str
    http_method: str
    path_template: str
    summary: str = ""
    description: str = ""
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    side_effect: str = WRITE
    requires_approval: bool = True
    enabled: bool = True
    embed_safe: bool = False
    response_transform: Optional[Dict[str, Any]] = None
    rate_limit: Optional[Dict[str, Any]] = None

    def policy(self) -> ToolPolicy:
        """The explicit ToolPolicy attached at registration (drives the gate)."""
        return _policy(self.http_method, self.side_effect, self.requires_approval)


@dataclass
class Integration:
    """One imported API (spec + base_url + auth reference) and its operations."""
    slug: str
    display_name: str
    base_url: str
    spec_kind: str
    auth_provider: Optional[str] = None
    security_scheme: Optional[str] = None
    auth_type: Optional[str] = None
    auth_config: Dict[str, Any] = field(default_factory=dict)
    default_headers: Dict[str, str] = field(default_factory=dict)
    egress_allow: List[str] = field(default_factory=list)
    allow_private: bool = False
    status: str = "active"
    operations: List[IntegrationOperation] = field(default_factory=list)


_READ_METHODS = frozenset({"GET", "HEAD"})
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "PUT"})


def side_effect_for_method(method: str) -> str:
    """GET/HEAD→read, DELETE→irreversible, POST/PUT/PATCH→write."""
    m = (method or "").upper()
    if m in _READ_METHODS:
        return READ
    if m == "DELETE":
        return IRREVERSIBLE
    return WRITE


def requires_approval_for_method(method: str) -> bool:
    """Safe default: anything that isn't a pure read needs approval until a
    tenant opts the operation out per-op (ag_api.operations.requires_approval)."""
    return side_effect_for_method(method) != READ


def policy_for_method(method: str, requires_approval: Optional[bool] = None) -> ToolPolicy:
    """Build the explicit ToolPolicy for a dynamic op from its HTTP verb.

    `requires_approval` (the per-op override) promotes a write to the
    irreversible/approval class when True, or relaxes it when explicitly False.
    """
    se = side_effect_for_method(method)
    ra = requires_approval_for_method(method) if requires_approval is None else requires_approval
    return _policy(method, se, ra)


def _policy(method: str, side_effect: str, requires_approval: bool) -> ToolPolicy:
    m = (method or "").upper()
    idempotent = m in _IDEMPOTENT_METHODS
    if side_effect == READ:
        return ToolPolicy(side_effect=READ, idempotent=True, cacheable=True,
                          timeout_s=30.0, retries=1)
    if requires_approval:
        effective = IRREVERSIBLE
    else:
        effective = WRITE if side_effect == IRREVERSIBLE else side_effect
    return ToolPolicy(side_effect=effective, idempotent=idempotent,
                      cacheable=False, timeout_s=30.0, retries=0)


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_TOOL_NAME = 160


def slugify(value: str) -> str:
    """Lowercase, non-alphanumerics → single underscore, trimmed."""
    return _SLUG_RE.sub("_", (value or "").lower()).strip("_")


def tool_name_for(slug: str, operation_id: str) -> str:
    """"{slug}__{operationId}" — double underscore namespaces two APIs that both
    expose e.g. create_ticket, and lets selection group by slug. Truncated to the
    column width with a stable tail so collisions after truncation are unlikely.
    """
    base = f"{slugify(slug)}__{slugify(operation_id)}"
    if len(base) <= _MAX_TOOL_NAME:
        return base
    import hashlib
    tail = hashlib.sha1(base.encode()).hexdigest()[:8]
    return base[:_MAX_TOOL_NAME - 9] + "_" + tail
