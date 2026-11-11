"""
Agent — Tenant Context (Multi-Tenancy Phase 2)
===================================================
Per-request/per-task tenant identity, mirroring the Storeglance platform's
`TenantContext` (Storeglance.Microservices.Core.Context.TenantContext) which
SMSystems registers AddScoped and populates from the request domain.

C# (per request, scoped DI):
    HeaderData : TenantContext { TenantId, TenantSlug, TenantDomain, UserId, ... }
    middleware: currentDomain = ?tenantDomain ?? X-Tenant-Domain ?? Host
                CurrentTenant = TenantService.GetTenantByDomain(domain)

Python (per request/task, contextvar):
    set_tenant_context(TenantContext(...))   # in the request middleware
    current_tenant()                          # anywhere downstream (tools, brain)

This holds IDENTITY only — no DB handles. The connection resolver (N3/N4)
turns a TenantContext into live per-tenant stores.

See MULTITENANT_MASTER_PLAN.md §3.1/§4 (N2).
"""

from __future__ import annotations

import os
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_TENANT_ID = "default"


def multi_tenant_enabled() -> bool:
    """Master switch for physical multi-tenancy (default OFF).

    When off, everything resolves to the default tenant → identical to the
    legacy single-tenant runtime. Read live so tests/ops can toggle.
    """
    return os.environ.get("AG_MULTI_TENANT", "0") not in ("0", "false", "False", "")


def tenant_suffix8(tenant_id: Optional[str]) -> str:
    """Last 8 hex of the tenant UUID — the platform DB/index naming suffix.

    Verified convention: tenant 5fe3034b…6008**a30d9b4a** →
    db prod_tenant_a30d9b4a / agent_tenant_a30d9b4a, index brain_a30d9b4a.
    For the default tenant (no UUID) returns 'default'.
    """
    if not tenant_id or tenant_id == DEFAULT_TENANT_ID:
        return DEFAULT_TENANT_ID
    hexchars = tenant_id.replace("-", "").strip().lower()
    return hexchars[-8:] if len(hexchars) >= 8 else hexchars


@dataclass
class TenantContext:
    """Per-request tenant identity (mirrors Storeglance TenantContext fields)."""
    tenant_id: str = DEFAULT_TENANT_ID
    tenant_key: str = DEFAULT_TENANT_ID
    display_name: str = "Default Organization"
    tenant_domain: Optional[str] = None
    is_default: bool = True
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    language: str = "en"
    timezone: Optional[str] = None
    agent_type: Optional[str] = None
    version: Optional[str] = None

    @property
    def suffix8(self) -> str:
        return tenant_suffix8(self.tenant_id)

    @property
    def agent_db_name(self) -> str:
        """Convention-derived agent DB name (used when MSystem has no explicit row)."""
        return "agent_uat" if self.is_default else f"agent_tenant_{self.suffix8}"

    def __repr__(self) -> str:
        return (f"TenantContext(tenant={self.tenant_key!r}, suffix8={self.suffix8!r}, "
                f"domain={self.tenant_domain!r}, default={self.is_default})")


DEFAULT_CONTEXT = TenantContext()


_current_tenant: ContextVar[TenantContext] = ContextVar(
    "current_tenant_context", default=DEFAULT_CONTEXT
)


def set_tenant_context(ctx: TenantContext):
    """Bind the active tenant for this request/task. Returns the reset token."""
    return _current_tenant.set(ctx)


def reset_tenant_context(token) -> None:
    """Restore the previous context (use the token from set_tenant_context)."""
    try:
        _current_tenant.reset(token)
    except Exception:
        pass


def current_tenant() -> TenantContext:
    """The active tenant context (DEFAULT_CONTEXT if none bound)."""
    return _current_tenant.get()


def current_tenant_id() -> str:
    return current_tenant().tenant_id
