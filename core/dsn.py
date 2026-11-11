"""
Agent — DSN Builder (Multi-Tenancy)
===================================================
Builds a PostgreSQL connection string from a tenant routing row resolved by
MSystem (tenant_mgmt.app_databases / tenant_databases) or from a service
connection config.

The connection layer is transport-agnostic: it always dials a resolved
(host, port). Today that is a DIRECT connection. A bastion SSH local-forward
branch is reserved (inert) so prod-behind-jump-host works later with no
call-site change — only this module and an optional dependency.

See MULTITENANT_MASTER_PLAN.md §4 (N6) and §6 Phase 6.
"""

from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DbEndpoint:
    """Resolved Postgres connection coordinates for one tenant/app DB.

    Mirrors SMSystems DatabaseContext (DbEndpoint/DbName/DbUser/DbPassword).
    """
    host: str
    port: int
    dbname: str
    user: str
    password: str = ""
    sslmode: Optional[str] = None
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_port: int = 22


def build_dsn(ep: DbEndpoint) -> str:
    """Return a postgresql:// DSN for the given endpoint.

    Direct path (active): dial ep.host:ep.port.
    SSH path (reserved):  if ep.ssh_host is set, a local forward would be
                          established and we'd dial 127.0.0.1:<local_port>.
                          Not active — guarded so behaviour is unchanged today.
    """
    host, port = ep.host, ep.port

    if ep.ssh_host:
        raise NotImplementedError(
            "SSH-tunnel DSN path is reserved (Phase 6). "
            "ssh_host was set but tunneling is not enabled."
        )

    user = urllib.parse.quote(ep.user or "", safe="")
    pwd = urllib.parse.quote(ep.password or "", safe="")
    auth = f"{user}:{pwd}@" if user else ""
    dsn = f"postgresql://{auth}{host}:{port}/{ep.dbname}"
    if ep.sslmode:
        dsn += f"?sslmode={ep.sslmode}"

    logger.debug("dsn.built: host=%s port=%s db=%s user=%s", host, port, ep.dbname, ep.user)
    return dsn


def endpoint_from_routing_row(row: dict) -> DbEndpoint:
    """Adapt an MSystem app_databases/tenant_databases row to a DbEndpoint.

    Accepts the column names used by tenant_mgmt.* (db_endpoint/db_name/...).
    The secret is expected already-decrypted by the caller (MSystemRouter).
    """
    return DbEndpoint(
        host=row.get("db_endpoint") or row.get("host") or "127.0.0.1",
        port=int(row.get("db_port") or row.get("port") or 5432),
        dbname=row.get("db_name") or row.get("dbname") or "",
        user=row.get("db_user") or row.get("user") or "",
        password=row.get("db_password") or row.get("password") or "",
        sslmode=row.get("sslmode"),
    )
