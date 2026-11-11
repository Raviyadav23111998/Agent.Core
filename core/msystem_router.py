"""
Agent — MSystem Router (Multi-Tenancy Phase 1)
===================================================
Read-only connection to the MSystem master/control plane (msystems_prod) that
resolves a request **domain → tenant → per-app (agent) database**, exactly like
SMSystems' DatabaseContext.

Resolution chain (verified against tenant_mgmt):
    domain  → tenant_mgmt.tenant_domains → tenant_id
            → tenant_mgmt.tenants            (identity + status)
            → tenant_mgmt.app_databases [app='agent'] (+ tenant_instances)  → agent DB creds
            → tenant_mgmt.tenant_profile     (branding for /TenantConfig)

This module ONLY resolves — it never opens tenant store connections itself.
That is the resolver/connections layer (Phase 2). Results are cached with a TTL.

Config:
    AG_MSYSTEM_URL  postgresql://… to msystems_prod (read-only role in prod).
                    Defaults to the local master for dev.

See MULTITENANT_MASTER_PLAN.md §4 (N1) and MULTITENANT_MSYSTEM_ROUTING_ARCHITECTURE.md §5.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# OPTIONAL. The agent runs standalone (its own database is the only control
# plane) unless AG_MSYSTEM_URL is set explicitly. There is deliberately no
# fallback URL: a missing setting must mean "no MSystem", never "try a local one".
DEFAULT_MSYSTEM_URL = os.environ.get("AG_MSYSTEM_URL", "").strip()


def msystem_configured() -> bool:
    """True only when an MSystem control plane has been configured (AG_MSYSTEM_URL).
    Read live so tests/ops can toggle."""
    return bool(os.environ.get("AG_MSYSTEM_URL", "").strip())

_CACHE_TTL = int(os.environ.get("AG_MSYSTEM_CACHE_TTL", "300"))

APP_KEY = os.environ.get("AG_APP_KEY", "agent")

_REQUIRE_SUBSCRIPTION = os.environ.get("AG_REQUIRE_SUBSCRIPTION", "1") not in ("0", "false", "False")

_ENTITLED_STATES = {"active", "grace_period", "grace", "trial", "trialing"}


@dataclass(frozen=True)
class TenantIdentity:
    """Who a domain belongs to (no DB creds)."""
    tenant_id: str
    tenant_key: str
    display_name: str
    primary_domain: Optional[str]
    is_active: bool
    matched_domain: Optional[str] = None


class _TTLCache:
    def __init__(self, ttl: int):
        self._ttl = ttl
        self._d: Dict[str, tuple[float, Any]] = {}

    def get(self, key: str):
        item = self._d.get(key)
        if not item:
            return None
        ts, val = item
        if time.time() - ts > self._ttl:
            self._d.pop(key, None)
            return None
        return val

    def set(self, key: str, val: Any):
        self._d[key] = (time.time(), val)

    def clear(self):
        self._d.clear()


class MSystemRouter:
    """Resolves domain → tenant → agent DB against msystems_prod (read-only)."""

    def __init__(self, msystem_url: str = None, app_key: str = None,
                 require_subscription: bool = None):
        self._url = msystem_url or os.environ.get("AG_MSYSTEM_URL", "").strip()
        if not self._url:
            raise RuntimeError("MSystem not configured (AG_MSYSTEM_URL unset) — standalone mode")
        self._app_key = app_key or APP_KEY
        self._require_sub = _REQUIRE_SUBSCRIPTION if require_subscription is None else require_subscription
        self._db = None
        self._cache = _TTLCache(_CACHE_TTL)

    def _conn(self):
        """Lazily open the (pooled) read-only connection to MSystem."""
        if self._db is None:
            from core.tenant_db import TenantDB
            self._db = TenantDB(self._url)
            self._db.connect()
            logger.info("msystem_router.connected: %s",
                        self._url.split("@")[-1] if "@" in self._url else "***")
        return self._db


    @staticmethod
    def normalize_host(raw: str) -> str:
        """Lowercase, strip port and a leading www."""
        host = (raw or "").strip().lower()
        if "://" in host:
            host = host.split("://", 1)[1]
        host = host.split("/")[0].split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        return host

    def resolve(self, identifier: str,
                require_subscription: Optional[bool] = None) -> Optional[TenantIdentity]:
        """Resolve a full host ('portal.gangacs.com') OR a subdomain label
        ('portal') OR a tenant_key to a TenantIdentity — SCOPED to this app.

        By default (require_subscription on) a tenant is returned ONLY if it
        holds an entitled subscription to this app (AG_APP_KEY, default 'agent').
        Tenants on other apps (smsystems, site_builder, …) resolve to None so the
        agent site never shows them. Cached (including the not-entitled negative).
        """
        require_sub = self._require_sub if require_subscription is None else require_subscription
        ident = self.normalize_host(identifier)
        if not ident:
            return None
        cache_key = f"id:{int(require_sub)}:{ident}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached or None

        db = self._conn()
        label = ident.split(".")[0]

        row = db.fetch_one(
            """
            SELECT t.tenant_id::text AS tenant_id, t.tenant_key, t.display_name,
                   t.primary_domain, t.is_active, d.domain AS matched_domain
            FROM tenant_mgmt.tenants t
            JOIN tenant_mgmt.tenant_domains d ON d.tenant_id = t.tenant_id
            WHERE lower(d.domain) = %s
               OR split_part(lower(d.domain), '.', 1) = %s
            ORDER BY (lower(d.domain) = %s) DESC, d.is_primary DESC, d.verified DESC
            LIMIT 1
            """,
            (ident, label, ident),
        )
        if not row:
            row = db.fetch_one(
                """
                SELECT tenant_id::text AS tenant_id, tenant_key, display_name,
                       primary_domain, is_active, NULL AS matched_domain
                FROM tenant_mgmt.tenants WHERE tenant_key = %s LIMIT 1
                """,
                (ident,),
            )

        if not row:
            self._cache.set(cache_key, False)
            logger.info("msystem_router.unresolved: %s", ident)
            return None

        identity = TenantIdentity(
            tenant_id=row["tenant_id"],
            tenant_key=row["tenant_key"],
            display_name=row["display_name"] or row["tenant_key"],
            primary_domain=row.get("primary_domain"),
            is_active=bool(row["is_active"]),
            matched_domain=row.get("matched_domain"),
        )

        if require_sub and not self.is_subscribed(identity.tenant_id):
            self._cache.set(cache_key, False)
            logger.info("msystem_router.not_entitled: host=%s tenant=%s app=%s "
                        "(not subscribed) — hidden from agent site",
                        ident, identity.tenant_id, self._app_key)
            return None

        self._cache.set(cache_key, identity)
        return identity


    def is_subscribed(self, tenant_id: str) -> bool:
        """True if the tenant holds an entitled subscription to THIS app
        (status active/grace/trial — not expired/cancelled). Cached.
        """
        ck = f"sub:{self._app_key}:{tenant_id}"
        cached = self._cache.get(ck)
        if cached is not None:
            return bool(cached)

        db = self._conn()
        row = db.fetch_one(
            """
            SELECT lower(s.status) AS status
            FROM tenant_mgmt.subscriptions s
            JOIN tenant_mgmt.apps a ON a.id = s.app_id
            WHERE s.tenant_id = %s::uuid AND a.app_key = %s
            ORDER BY s.current_period_end DESC NULLS LAST
            LIMIT 1
            """,
            (tenant_id, self._app_key),
        )
        entitled = bool(row and row["status"] in _ENTITLED_STATES)
        self._cache.set(ck, entitled)
        return entitled


    def resolve_agent_db(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Return the agent (app='agent') database row for a tenant, or None
        if not provisioned yet (Phase 5 fills tenant_mgmt.app_databases).
        """
        cached = self._cache.get(f"db:{tenant_id}")
        if cached is not None:
            return cached or None

        db = self._conn()
        row = db.fetch_one(
            """
            SELECT ad.db_name, ad.db_endpoint, ad.db_port, ad.db_user,
                   ad.db_password, ad.connection_is_secret,
                   i.server_endpoint, i.region, i.instance_key
            FROM tenant_mgmt.app_databases ad
            JOIN tenant_mgmt.apps a ON a.id = ad.app_id AND a.app_key = 'agent'
            JOIN tenant_mgmt.tenant_instances i ON i.id = ad.instance_id
            WHERE ad.tenant_id = %s::uuid
            LIMIT 1
            """,
            (tenant_id,),
        )
        self._cache.set(f"db:{tenant_id}", row or False)
        if not row:
            logger.info("msystem_router.agent_db_unprovisioned: tenant=%s", tenant_id)
        return row

    def list_agent_tenants(self) -> List[Dict[str, Any]]:
        """Every ACTIVE tenant that has a provisioned agent (app='agent') DB.

        The authoritative source for fan-out work that must run for ALL tenants
        in one process (e.g. the cron scheduler ticking each tenant's own
        ag_agents.cron_jobs). DB-per-tenant means the row's DB IS its scope, so
        we return identities only — the caller binds each TenantContext and lets
        the routing proxy resolve the tenant's pool. Cached (short TTL).
        """
        cached = self._cache.get("agent_tenants_list")
        if cached is not None:
            return cached

        db = self._conn()
        rows = db.fetch_all(
            """
            SELECT DISTINCT t.tenant_id::text AS tenant_id, t.tenant_key,
                   t.display_name, t.primary_domain
            FROM tenant_mgmt.app_databases ad
            JOIN tenant_mgmt.apps a ON a.id = ad.app_id AND a.app_key = 'agent'
            JOIN tenant_mgmt.tenants t ON t.tenant_id = ad.tenant_id
            WHERE t.is_active = TRUE
            ORDER BY t.tenant_key
            """
        ) or []
        result = [dict(r) for r in rows]
        self._cache.set("agent_tenants_list", result)
        logger.info("msystem_router.agent_tenants: %d provisioned", len(result))
        return result


    def get_branding(self, tenant_id: str) -> Dict[str, Any]:
        """Return tenant_profile branding fields, or {} if none."""
        db = self._conn()
        row = db.fetch_one(
            """
            SELECT org_name, logo_url, cover_image_url, theme_color, motto,
                   email, phone, website, address_line1, address_line2,
                   city, state, country, pincode, timezone
            FROM tenant_mgmt.tenant_profile WHERE tenant_id = %s::uuid LIMIT 1
            """,
            (tenant_id,),
        )
        return row or {}


    def agent_config(self) -> Dict[str, str]:
        """All rows of tenant_mgmt.agent_config as {key: value}. Cached (TTL).
        Best-effort: returns {} if MSystem is unreachable so callers fall back to
        code DEFAULTS rather than crash."""
        cached = self._cache.get("agent_config")
        if cached is not None:
            return cached
        try:
            rows = self._conn().fetch_all(
                "SELECT config_key, config_value FROM tenant_mgmt.agent_config", ()
            ) or []
            d = {r["config_key"]: (r["config_value"] or "") for r in rows}
        except Exception as e:
            # Do NOT cache a failed read. Caching {} here would poison every
            # get_global() lookup for the whole TTL window after a single
            # transient MSystem/PG blip — which is exactly how a StorageManager
            # ended up built with an empty neo4j.password and then spammed
            # "Unauthorized / missing key credentials". Return empty so callers
            # fall back to DEFAULTS for THIS call, and retry on the next one.
            logger.warning("msystem_router.agent_config_failed (not cached, will retry): %s", e)
            return {}
        self._cache.set("agent_config", d)
        return d

    def set_agent_config(self, key: str, value: str, is_secret: bool = False) -> None:
        """Upsert a global config row and invalidate the cache."""
        self._conn().execute(
            """
            INSERT INTO tenant_mgmt.agent_config (config_key, config_value, is_secret, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (config_key) DO UPDATE SET
                config_value = EXCLUDED.config_value,
                is_secret = EXCLUDED.is_secret,
                updated_at = now()
            """,
            (key, value, is_secret),
        )
        self._cache._d.pop("agent_config", None)

    def invalidate(self):
        self._cache.clear()

    def close(self):
        if self._db:
            try:
                self._db.close()
            finally:
                self._db = None
