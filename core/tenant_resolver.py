"""
Agent — Tenant Connection Resolver (Multi-Tenancy Phase 2)
===========================================================
Turns a TenantContext into a cached, lazily-built TenantConnections bundle.
The Python twin of SMSystems' DatabaseContext + TenantStoreCache:

    C#: _headerData.CurrentTenant → DbEndpoint/DbName/DbUser/Decrypt(DbPassword)
        → NpgsqlConnectionStringBuilder → TenantStoreCache.GetOrAdd(connString)

    Py: ctx → MSystemRouter.resolve_agent_db(tenant_id) → DbEndpoint → DSN
        → TenantConnections (cached, LRU + idle eviction)

Unlike SMSystems' unbounded ConcurrentDictionary, we add an LRU cap + idle TTL
reaper so connection counts stay bounded across many tenants.

Safety (mirrors SMSystems "database not configured"): an ENTITLED tenant with no
provisioned agent DB raises TenantNotProvisioned — we never silently fall back to
the default DB (that would leak data into the wrong database).

See MULTITENANT_MASTER_PLAN.md §4 (N3)/§5.
"""

from __future__ import annotations

import os
import time
import logging
import threading
from collections import OrderedDict

from core.tenant_context import TenantContext
from core.tenant_connections import TenantConnections

logger = logging.getLogger(__name__)

_MAX_TENANTS = int(os.environ.get("AG_TENANT_CACHE_MAX", "50"))
_IDLE_TTL = int(os.environ.get("AG_TENANT_IDLE_TTL", "900"))


class TenantNotProvisioned(Exception):
    """Entitled tenant has no agent database row in MSystem yet."""


_GLOBAL_RESOLVER = None


def set_global_resolver(resolver) -> None:
    global _GLOBAL_RESOLVER
    _GLOBAL_RESOLVER = resolver


def get_global_resolver():
    return _GLOBAL_RESOLVER


def _msystem_field_key():
    """The key MSystem used to FieldEncryptor-encrypt app_databases creds.

    This is the MSystem control-plane key (Security:FieldEncryptionKey), which is
    DISTINCT from the agent's own AG_FIELD_ENCRYPTION_KEY (used for the agent's
    own secrets). Set AG_MSYSTEM_FIELD_KEY == MSystem Security:FieldEncryptionKey
    per env. Falls back to the agent's own key only if unset (single-key deploys).
    """
    import base64
    kb = os.environ.get("AG_MSYSTEM_FIELD_KEY", "")
    if not kb:
        return None
    try:
        return base64.b64decode(kb)
    except Exception as e:
        logger.error("tenant_resolver.bad_msystem_field_key: %s", e)
        return None


def _decrypt_db_password(row: dict) -> str:
    """Resolve the agent DB password from an MSystem routing row.

    MSystem stores it via FieldEncryptor (AES-256-GCM, 'ENC:...') under its OWN
    control-plane key — so we decrypt with AG_MSYSTEM_FIELD_KEY, not the agent's
    own AG_FIELD_ENCRYPTION_KEY. Plaintext (dev/superuser) passes through.
    Mirror of SMSystems DatabaseContext → FieldEncryptor.Decrypt(DbPassword).
    """
    pwd = row.get("db_password") or ""
    if not pwd:
        return ""
    try:
        from core.security import decrypt_field, is_field_encrypted
        if is_field_encrypted(pwd):
            return decrypt_field(pwd, key_override=_msystem_field_key())
    except Exception as e:
        logger.error("tenant_resolver.db_password_decrypt_failed: %s", e)
        raise
    return pwd


class TenantConnectionResolver:
    """Resolves + caches per-tenant connection bundles."""

    def __init__(self, msystem_router=None, default_db_url: str = None,
                 max_tenants: int = _MAX_TENANTS, idle_ttl: int = _IDLE_TTL):
        self._router = msystem_router
        self._default_db_url = default_db_url or os.environ.get("AG_DATABASE_URL")
        self._max = max_tenants
        self._idle = idle_ttl
        self._cache: "OrderedDict[str, tuple[float, TenantConnections]]" = OrderedDict()
        self._lock = threading.RLock()


    def for_tenant(self, ctx: TenantContext) -> TenantConnections:
        """Return the (cached) TenantConnections for this context."""
        key = ctx.tenant_id
        with self._lock:
            self._reap_idle_locked()
            item = self._cache.get(key)
            if item:
                self._cache.move_to_end(key)
                self._cache[key] = (time.time(), item[1])
                return item[1]

            url = self._resolve_agent_db_url(ctx)
            conns = TenantConnections(ctx, url)
            self._cache[key] = (time.time(), conns)
            self._evict_over_capacity_locked()
            logger.info("tenant_resolver.bound: tenant=%s suffix=%s (cache=%d)",
                        ctx.tenant_key, ctx.suffix8, len(self._cache))
            return conns

    def invalidate(self, tenant_id: str):
        with self._lock:
            item = self._cache.pop(tenant_id, None)
        if item:
            item[1].close()

    def close_all(self):
        with self._lock:
            items = list(self._cache.values())
            self._cache.clear()
        for _, conns in items:
            conns.close()


    def _resolve_agent_db_url(self, ctx: TenantContext) -> str:
        """Map a tenant to its agent DB DSN.

        Default tenant → the configured default/control DB (single-tenant compat).
        Real tenant    → MSystem app_databases row → DSN. Missing → not provisioned.
        """
        if ctx.is_default:
            if not self._default_db_url:
                raise TenantNotProvisioned("default tenant has no AG_DATABASE_URL")
            return self._default_db_url

        if not self._router:
            raise TenantNotProvisioned(f"no MSystem router to resolve {ctx.tenant_key}")

        row = self._router.resolve_agent_db(ctx.tenant_id)
        if not row:
            raise TenantNotProvisioned(
                f"tenant {ctx.tenant_key} ({ctx.suffix8}) is entitled to the agent app "
                f"but has no agent database provisioned (tenant_mgmt.app_databases)"
            )

        from core.dsn import build_dsn, endpoint_from_routing_row
        row = dict(row)
        row["db_password"] = _decrypt_db_password(row)
        return build_dsn(endpoint_from_routing_row(row))

    def _reap_idle_locked(self):
        if self._idle <= 0:
            return
        now = time.time()
        stale = [k for k, (ts, _) in self._cache.items() if now - ts > self._idle]
        for k in stale:
            _, conns = self._cache.pop(k)
            conns.close()
            logger.info("tenant_resolver.reaped_idle: tenant_id=%s", k)

    def _evict_over_capacity_locked(self):
        while len(self._cache) > self._max:
            k, (_, conns) = self._cache.popitem(last=False)
            conns.close()
            logger.info("tenant_resolver.evicted_lru: tenant_id=%s", k)
