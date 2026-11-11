"""
Agent — Service Connections Reader (Multi-Tenancy Phase 2)
===========================================================
Reads ag_system.service_connections from a TENANT's own Postgres DB and returns
the resolved (decrypted) coordinates for that tenant's separate backing stores:
elastic, neo4j, redis, embeddings.

This is the Antigravity analog of how SMSystems keeps per-tenant configuration
INSIDE the tenant database (system.codes / system.modules). The tenant DB is the
single source of truth for that tenant's services — see MULTITENANT_MASTER_PLAN.md
§4 (N5)/D8 and MULTITENANT_ELASTIC_NEO4J_DESIGN.md §1.

Secrets are Fernet-encrypted (core.security), same vault scheme as credentials.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceConnection:
    """One resolved store connection for a tenant."""
    service_key: str
    config: Dict[str, Any] = field(default_factory=dict)
    secret: Optional[str] = None
    is_enabled: bool = True

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


def _decrypt(secret_enc: Optional[str]) -> Optional[str]:
    if not secret_enc:
        return None
    try:
        from core.security import decrypt_secret
        return decrypt_secret(secret_enc)
    except Exception as e:
        logger.warning("service_connections.decrypt_failed: %s", e)
        return None


def read_service_connections(db) -> Dict[str, ServiceConnection]:
    """Return {service_key: ServiceConnection} for a tenant DB.

    `db` is a connected TenantDB for the tenant's agent database. Degrades to an
    empty dict if the table is missing (pre-V027) so callers fall back cleanly.
    """
    try:
        rows = db.fetch_all(
            """
            SELECT service_key, config_json, secret_enc, is_enabled
            FROM ag_system.service_connections
            """
        )
    except Exception as e:
        logger.info("service_connections.unavailable (%s) — using none", e)
        return {}

    out: Dict[str, ServiceConnection] = {}
    for r in rows:
        cfg = r.get("config_json") or {}
        if isinstance(cfg, str):
            import json
            try:
                cfg = json.loads(cfg)
            except Exception:
                cfg = {}
        out[r["service_key"]] = ServiceConnection(
            service_key=r["service_key"],
            config=cfg,
            secret=_decrypt(r.get("secret_enc")),
            is_enabled=bool(r.get("is_enabled", True)),
        )
    return out
