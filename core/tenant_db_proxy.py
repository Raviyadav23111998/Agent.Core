"""
Agent — Tenant DB Proxy (Multi-Tenancy Phase 2b)
=================================================
A drop-in stand-in for TenantDB that transparently routes every call to the
ACTIVE tenant's pool, read from the per-request contextvar.

This is the Python equivalent of how SMSystems works: repositories don't hold a
fixed connection — they ask DatabaseContext for one per call, and DatabaseContext
reads the scoped HeaderData.CurrentTenant. Here, repos keep `self._db` (the proxy)
and every `self._db.fetch_one(...)` lands on the right tenant's TenantDB.

Because of this proxy, M14 (per-tenant repos) needs **zero** repo changes — only
what `app.state.db` IS changes (main.py).

Routing:
    multi-tenant OFF or default context → the control DB (agent_uat) — unchanged behaviour
    real tenant context                 → resolver.for_tenant(ctx).agent_db

See MULTITENANT_MASTER_PLAN.md §4 (M14) and SMSystems DatabaseContext.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TenantDBProxy:
    """Routes TenantDB calls to the active tenant's pool via the contextvar."""

    def __init__(self, resolver, control_db):
        object.__setattr__(self, "_resolver", resolver)
        object.__setattr__(self, "_control_db", control_db)

    def _active(self):
        """Resolve the TenantDB for the current request context."""
        from core.tenant_context import current_tenant, multi_tenant_enabled
        ctx = current_tenant()
        if not multi_tenant_enabled() or ctx.is_default:
            return self._control_db
        return self._resolver.for_tenant(ctx).agent_db

    def __getattr__(self, name):
        return getattr(self._active(), name)

    def close(self):
        try:
            self._control_db.close()
        finally:
            try:
                self._resolver.close_all()
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        try:
            return self._active().is_connected
        except Exception:
            return False

    def __repr__(self) -> str:
        return "TenantDBProxy(routes→active tenant; control=%r)" % (self._control_db,)
