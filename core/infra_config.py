"""
Infrastructure config — process-global settings resolved WITHOUT tenant context.
═══════════════════════════════════════════════════════════════════════════════
Stage 0 of the runtime redesign (see AGENT_RUNTIME_REDESIGN.md §C2).

Infrastructure — Redis, Postgres, Elasticsearch, Neo4j — is shared by the whole
process. It MUST resolve identically regardless of which tenant a request belongs
to, and from background threads that have no tenant context at all.

Reading these through the tenant-aware `app_config.get()` caused a class of bugs:
a value stored only in a tenant overlay was invisible to the pub/sub invalidation
subscriber (a daemon thread with no tenant), so the subscriber silently never
connected and cross-process session invalidation died — config "needed a restart".

This module is the single, tenant-agnostic source for infra settings. Resolution
is `env var → control-plane cache → hardcoded default` (env is authoritative for
infra); the tenant overlay is intentionally NEVER consulted.
"""

from __future__ import annotations

from typing import Optional


def _g(key: str, default: Optional[str] = None, env: str = None) -> Optional[str]:
    from core.config import app_config
    return app_config.get_global(key, default=default, env=env)


def redis_url() -> str:
    """Redis connection URL, or "" when Redis is not configured (degraded mode)."""
    return (_g("redis.url", "", env="REDIS_URL") or "").strip()


def db_url() -> Optional[str]:
    return _g("db.url", env="AG_DATABASE_URL")


def es_url() -> str:
    return _g("elasticsearch.url", "http://localhost:9200", env="ELASTICSEARCH_URL")


def es_index() -> str:
    return _g("elasticsearch.index", "brain", env="ELASTICSEARCH_INDEX")


def neo4j_uri() -> str:
    return _g("neo4j.uri", "bolt://localhost:7687", env="NEO4J_URI")


def neo4j_user() -> str:
    return _g("neo4j.user", "neo4j", env="NEO4J_USER")


def neo4j_password() -> str:
    return _g("neo4j.password", "", env="NEO4J_PASSWORD")
