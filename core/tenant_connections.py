"""
Agent — Tenant Connections (Multi-Tenancy Phase 2)
===================================================
Holds ONE tenant's live backing-store handles, built lazily. The connection
resolver (N3) caches one of these per active tenant.

Analogous to SMSystems' TenantStoreCache (a per-connection-string IDocumentStore
cache) — but per tenant and holding all of the tenant's stores, not just PG.

Phase 2 (now): agent_db (Postgres) + service_connections (read from that DB).
Phase 3 (next): lexical (ES brain_<id>), graph (Neo4j shared+tenant_id), redis.

See MULTITENANT_MASTER_PLAN.md §3/§4 (N4).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from core.tenant_context import TenantContext
from core.service_connections import ServiceConnection, read_service_connections

logger = logging.getLogger(__name__)


class TenantConnections:
    """Lazily-built, cached bundle of one tenant's store handles."""

    def __init__(self, ctx: TenantContext, agent_db_url: str):
        self.ctx = ctx
        self._agent_db_url = agent_db_url
        self._agent_db = None
        self._service_conns: Optional[Dict[str, ServiceConnection]] = None
        self._storage = None
        self._brain_bundle = None
        self._redis = None


    @property
    def agent_db(self):
        """The tenant's agent Postgres pool (TenantDB), connected on first use."""
        if self._agent_db is None:
            from core.tenant_db import TenantDB
            self._agent_db = TenantDB(self._agent_db_url)
            self._agent_db.connect()
            logger.info("tenant_connections.agent_db_open: tenant=%s suffix=%s",
                        self.ctx.tenant_key, self.ctx.suffix8)
        return self._agent_db


    @property
    def service_connections(self) -> Dict[str, ServiceConnection]:
        if self._service_conns is None:
            self._service_conns = read_service_connections(self.agent_db)
            logger.info("tenant_connections.services_loaded: tenant=%s keys=%s",
                        self.ctx.tenant_key, sorted(self._service_conns.keys()))
        return self._service_conns

    def service(self, key: str) -> Optional[ServiceConnection]:
        return self.service_connections.get(key)

    def _override(self, service_key: str, field: str):
        """A per-tenant override value from service_connections, or None.

        service_connections is OPTIONAL — coordinates are DERIVED from tenant_id and
        shared infra comes from agent config. A row only exists when a tenant needs a
        non-default cluster/endpoint. See PRODUCTION_PLAN §5.1 (A-1)."""
        sc = self.service_connections.get(service_key)
        if not sc or not sc.is_enabled:
            return None
        if field == "secret":
            return sc.secret
        return sc.get(field)

    @staticmethod
    def _infra(key: str, default=None):
        """Shared-infra value from agent config (same cluster for all tenants).

        MUST use get_global() — never get() — because infrastructure (Neo4j, ES,
        Redis) is process-global and must resolve identically for every tenant.
        get() is tenant-aware: under a tenant context it reads only that tenant's
        ag_system.configs (which typically has NO infra rows) then DEFAULTS,
        skipping the msystems agent_config table.  get_global() resolves via
        ENV → msystems agent_config → DEFAULTS — the correct chain for shared
        infra.  See infra_config.py and MULTITENANT_CONFIG_ARCHITECTURE §2."""
        try:
            from core.config import app_config
            return app_config.get_global(key) or default
        except Exception:
            return default


    @property
    def storage_manager(self):
        """The tenant's StorageManager. Coordinates are DERIVED from the tenant id
        (ES index brain_<sfx>, graph keyed by tenant_id) with SHARED infra from agent
        config (ES/Neo4j cluster). `service_connections` rows, if present, override.
        No seeding required — a tenant works the moment its Postgres DB exists."""
        if self._storage is None:
            from brain.storage.manager import StorageManager
            sfx = self.ctx.suffix8

            es_url = self._override("elastic", "url") or self._infra("elasticsearch.url")
            es_index = self._override("elastic", "index") or f"brain_{sfx}"
            neo_uri = self._override("neo4j", "uri") or self._infra("neo4j.uri")
            neo_user = self._override("neo4j", "user") or self._infra("neo4j.user") or "neo4j"
            neo_pw = (self._override("neo4j", "secret") or self._override("neo4j", "password")
                      or self._infra("neo4j.password"))

            sm = StorageManager(
                db=self.agent_db,
                elasticsearch_url=es_url,
                elasticsearch_index=es_index,
                neo4j_uri=neo_uri,
                neo4j_user=neo_user,
                neo4j_password=neo_pw,
                graph_tenant_id=self.ctx.tenant_id,
            )
            sm.connect()
            self._storage = sm
            logger.info("tenant_connections.storage_open: tenant=%s index=%s (derived)",
                        self.ctx.tenant_key, es_index)
        return self._storage


    def brain_bundle(self, llm_client=None, want_brain: bool = True,
                     want_pipeline: bool = True):
        """The tenant's (storage, Brain, IngestionPipeline) triple, built once
        and cached for the LIFETIME of this connection bundle.

        Crucially this shares ONE lifecycle with the underlying handles: when the
        resolver evicts/invalidates this tenant, close() drops both the storage
        AND this bundle together, so a caller can never end up holding a Brain
        that points at an already-closed Postgres pool. (brain_mcp previously kept
        its own independent bundle cache, which is exactly that use-after-close.)
        """
        if self._brain_bundle is None:
            storage = self.storage_manager
            brain = None
            if want_brain:
                from brain.retrieval.brain import Brain
                brain = Brain(storage=storage, llm_client=llm_client)
            pipeline = None
            if want_pipeline:
                from brain.ingestion.pipeline import IngestionPipeline
                pipeline = IngestionPipeline(
                    storage=storage, llm_client=llm_client,
                    enable_graph_extraction=True,
                )
            self._brain_bundle = (storage, brain, pipeline)
            logger.info("tenant_connections.brain_bundle_built: tenant=%s", self.ctx.tenant_key)
        return self._brain_bundle

    def close(self):
        """Close all open handles. Called on eviction/shutdown."""
        if self._storage is not None:
            try:
                self._storage.disconnect()
            except Exception:
                pass
        if self._agent_db is not None:
            try:
                self._agent_db.close()
            except Exception:
                pass
        self._agent_db = None
        self._service_conns = None
        self._storage = self._brain_bundle = self._redis = None
        logger.info("tenant_connections.closed: tenant=%s", self.ctx.tenant_key)

    def __repr__(self) -> str:
        return f"TenantConnections(tenant={self.ctx.tenant_key!r}, suffix8={self.ctx.suffix8!r})"
