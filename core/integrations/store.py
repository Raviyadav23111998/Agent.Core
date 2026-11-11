"""
ag_api persistence — the only place that reads/writes the integration catalog.
═══════════════════════════════════════════════════════════════════════════════
Tenant-routed exactly like Connections: the `db` handed in is a TenantDBProxy, so
every query lands in the active tenant's DB. Used by ingest (write side) and, from
Phase 2, the runtime (read side). No business logic — just typed CRUD + the
spec-hash diff that keeps re-imports idempotent.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from core.integrations.model import Integration, IntegrationOperation

logger = logging.getLogger(__name__)


def _vec_literal(vec: List[float]) -> str:
    """pgvector text input: '[v1,v2,...]'."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class IntegrationStore:
    def __init__(self, db):
        self._db = db

    def get_integration_by_slug(self, slug: str) -> Optional[dict]:
        return self._db.fetch_one(
            "SELECT * FROM ag_api.integrations WHERE slug = %s", (slug,))

    def upsert_integration(self, integ: Integration, *, spec_raw: dict,
                           spec_hash: str, created_by: Optional[int]) -> int:
        """Create or update the integration row; returns integration_id."""
        row = self._db.execute_returning(
            """INSERT INTO ag_api.integrations
                   (slug, display_name, base_url, spec_kind, spec_raw, spec_hash,
                    auth_provider, security_scheme, auth_type, auth_config, default_headers,
                    egress_allow, allow_private, status, created_by)
               VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s)
               ON CONFLICT (slug) DO UPDATE SET
                   display_name=EXCLUDED.display_name, base_url=EXCLUDED.base_url,
                   spec_kind=EXCLUDED.spec_kind, spec_raw=EXCLUDED.spec_raw,
                   spec_hash=EXCLUDED.spec_hash, auth_provider=EXCLUDED.auth_provider,
                   security_scheme=EXCLUDED.security_scheme, auth_type=EXCLUDED.auth_type,
                   auth_config=EXCLUDED.auth_config,
                   -- ingest already merged admin values into default_headers.
                   default_headers=EXCLUDED.default_headers,
                   egress_allow=EXCLUDED.egress_allow, allow_private=EXCLUDED.allow_private,
                   status=EXCLUDED.status, updated_at=NOW()
               RETURNING integration_id""",
            (integ.slug, integ.display_name, integ.base_url, integ.spec_kind,
             json.dumps(spec_raw), spec_hash, integ.auth_provider, integ.security_scheme,
             integ.auth_type, json.dumps(integ.auth_config), json.dumps(integ.default_headers),
             json.dumps(integ.egress_allow), integ.allow_private, integ.status, created_by),
        )
        return row["integration_id"]

    def replace_operations(self, integration_id: int,
                           ops: List[IntegrationOperation],
                           req_maps: Dict[str, dict]) -> Dict[str, int]:
        """Upsert incoming ops; soft-disable ops no longer present.

        Idempotent: a re-import with the same spec touches rows but changes nothing
        observable. Returns {added, updated, deprecated}. Preserves admin per-op
        overrides (side_effect / requires_approval / enabled / embed_safe) on
        UPDATE — re-import must not silently re-open a disabled op.
        """
        existing = {r["operation_id"]: r for r in self._db.fetch_all(
            "SELECT operation_id, enabled FROM ag_api.operations WHERE integration_id=%s",
            (integration_id,))}
        incoming = {op.operation_id for op in ops}
        added = updated = 0

        for op in ops:
            row = self._db.execute_returning(
                """INSERT INTO ag_api.operations
                       (integration_id, operation_id, tool_name, http_method,
                        path_template, summary, description, parameters_schema,
                        request_map, side_effect, requires_approval, enabled)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s)
                   ON CONFLICT (integration_id, operation_id) DO UPDATE SET
                       tool_name=EXCLUDED.tool_name, http_method=EXCLUDED.http_method,
                       path_template=EXCLUDED.path_template, summary=EXCLUDED.summary,
                       description=EXCLUDED.description,
                       parameters_schema=EXCLUDED.parameters_schema,
                       request_map=EXCLUDED.request_map
                       -- NB: side_effect/requires_approval/enabled/embed_safe are
                       -- admin-owned overrides; NOT overwritten on re-import.
                   RETURNING (xmax = 0) AS inserted""",
                (integration_id, op.operation_id, op.tool_name, op.http_method,
                 op.path_template, op.summary, op.description,
                 json.dumps(op.parameters_schema), json.dumps(req_maps.get(op.operation_id, {})),
                 op.side_effect, op.requires_approval, op.enabled),
            )
            if row and row.get("inserted"):
                added += 1
            else:
                updated += 1

        stale = [oid for oid in existing if oid not in incoming]
        for oid in stale:
            self._db.execute(
                "UPDATE ag_api.operations SET enabled=FALSE "
                "WHERE integration_id=%s AND operation_id=%s", (integration_id, oid))
        return {"added": added, "updated": updated, "deprecated": len(stale)}

    def set_embeddings(self, integration_id: int,
                       vectors: Dict[str, List[float]]) -> int:
        """Store per-operation embedding vectors ({operation_id: vec})."""
        n = 0
        for op_id, vec in vectors.items():
            if not vec:
                continue
            self._db.execute(
                "UPDATE ag_api.operations SET embedding=%s::vector "
                "WHERE integration_id=%s AND operation_id=%s",
                (_vec_literal(vec), integration_id, op_id))
            n += 1
        return n

    def list_operations(self, integration_id: int) -> List[dict]:
        return self._db.fetch_all(
            "SELECT * FROM ag_api.operations WHERE integration_id=%s ORDER BY operation_pk",
            (integration_id,))

    def list_operations_display(self, integration_id: int) -> List[dict]:
        """Operations for the admin DETAIL view — only the columns the UI renders.

        `list_operations` does SELECT * which drags the 768-dim `embedding` vector
        (≈ 2 MB for a 200-op API) and the full `parameters_schema` over the wire,
        even though the admin table shows none of it. That made the detail drawer
        (which hosts the Grants panel) take ~12 s to open and look broken. Select
        the lean set instead.
        """
        return self._db.fetch_all(
            """SELECT operation_id, tool_name, http_method, path_template,
                      summary, description, side_effect, requires_approval,
                      enabled, embed_safe,
                      (embedding IS NOT NULL) AS embedded
               FROM ag_api.operations
               WHERE integration_id=%s ORDER BY operation_pk""",
            (integration_id,))

    def get_integration_by_provider(self, provider: str) -> Optional[dict]:
        """The integration that auto-created this ag_cred provider (`api_{slug}`) —
        used by the connect-time login exchange to read the login recipe."""
        return self._db.fetch_one(
            "SELECT integration_id, slug, base_url, auth_type, auth_config, "
            "default_headers, allow_private "
            "FROM ag_api.integrations WHERE auth_provider = %s", (provider,))

    def operations_missing_embeddings(self, integration_id: int) -> List[dict]:
        """Enabled ops that still have no vector — the backfill work-list."""
        return self._db.fetch_all(
            "SELECT operation_id, http_method, path_template, summary, description "
            "FROM ag_api.operations "
            "WHERE integration_id=%s AND enabled=TRUE AND embedding IS NULL "
            "ORDER BY operation_pk",
            (integration_id,))

    def get_op_for_execution(self, tool_name: str) -> Optional[dict]:
        """Op row + its integration's request fields, for a live call. Only
        enabled ops of active integrations — a disabled op is unresolvable."""
        return self._db.fetch_one(
            """SELECT o.operation_id, o.tool_name, o.http_method, o.path_template,
                      o.parameters_schema, o.request_map, o.side_effect,
                      o.requires_approval, o.response_transform, o.rate_limit,
                      i.integration_id, i.slug, i.base_url, i.auth_provider,
                      i.security_scheme, i.auth_type, i.auth_config, i.default_headers,
                      i.egress_allow, i.allow_private, i.spec_kind
               FROM ag_api.operations o
               JOIN ag_api.integrations i ON i.integration_id = o.integration_id
               WHERE o.tool_name = %s AND o.enabled = TRUE AND i.status = 'active'""",
            (tool_name,))

    _GRANT_WHERE = (
        "o.enabled = TRUE AND i.status = 'active' AND ("
        "  g.scope = 'org'"
        "  OR (g.scope = 'app'     AND g.ref = %(app)s)"
        "  OR (g.scope = 'persona' AND g.ref = %(persona)s))"
        "  AND (g.op_filter IS NULL OR g.op_filter ? o.operation_id)"
    )

    def catalog_for_principal(self, *, app_id: str = "", persona_id: str = "",
                              embed_safe_only: bool = False) -> List[dict]:
        """Ops of integrations GRANTED to this app/persona (or org-wide), for
        per-session registration. embed_safe_only restricts to ops an admin
        marked reachable by anonymous chatbot visitors."""
        safe = " AND o.embed_safe = TRUE" if embed_safe_only else ""
        return self._db.fetch_all(
            f"""SELECT DISTINCT o.tool_name, o.operation_id, o.http_method,
                       o.description, o.parameters_schema, o.side_effect,
                       o.requires_approval
                FROM ag_api.operations o
                JOIN ag_api.integrations i ON i.integration_id = o.integration_id
                JOIN ag_api.integration_grants g ON g.integration_id = o.integration_id
                WHERE {self._GRANT_WHERE}{safe}
                ORDER BY o.tool_name""",
            {"app": app_id or "", "persona": persona_id or ""})

    def search_catalog(self, query_vec: List[float], *, app_id: str = "",
                       persona_id: str = "", k: int = 8,
                       embed_safe_only: bool = False) -> List[str]:
        """Top-k granted op tool_names by cosine distance (HNSW). Embedding-less
        ops sort last (NULL distance) — callers union with keyword fallback."""
        safe = " AND o.embed_safe = TRUE" if embed_safe_only else ""
        rows = self._db.fetch_all(
            f"""SELECT DISTINCT o.tool_name, (o.embedding <=> %(vec)s::vector) AS dist
                FROM ag_api.operations o
                JOIN ag_api.integrations i ON i.integration_id = o.integration_id
                JOIN ag_api.integration_grants g ON g.integration_id = o.integration_id
                WHERE {self._GRANT_WHERE}{safe} AND o.embedding IS NOT NULL
                ORDER BY dist
                LIMIT %(k)s""",
            {"app": app_id or "", "persona": persona_id or "",
             "vec": _vec_literal(query_vec), "k": int(k)})
        return [r["tool_name"] for r in rows]

    def list_integrations(self) -> List[dict]:
        return self._db.fetch_all(
            """SELECT i.integration_id, i.slug, i.display_name, i.base_url,
                      i.spec_kind, i.auth_provider, i.security_scheme, i.auth_type,
                      i.status, i.allow_private, i.created_at,
                      (SELECT COUNT(*) FROM ag_api.operations o
                       WHERE o.integration_id = i.integration_id) AS op_count
               FROM ag_api.integrations i ORDER BY i.slug""")

    def get_integration(self, integration_id: int) -> Optional[dict]:
        return self._db.fetch_one(
            "SELECT * FROM ag_api.integrations WHERE integration_id = %s",
            (integration_id,))

    def update_integration(self, integration_id: int, *, status: Optional[str] = None,
                           egress_allow: Optional[list] = None,
                           allow_private: Optional[bool] = None,
                           auth_type: Optional[str] = None,
                           auth_config: Optional[dict] = None,
                           default_headers: Optional[dict] = None) -> None:
        sets, params = [], []
        if default_headers is not None:
            sets.append("default_headers = %s::jsonb"); params.append(json.dumps(default_headers))
        if status is not None:
            sets.append("status = %s"); params.append(status)
        if egress_allow is not None:
            sets.append("egress_allow = %s::jsonb"); params.append(json.dumps(egress_allow))
        if allow_private is not None:
            sets.append("allow_private = %s"); params.append(allow_private)
        if auth_type is not None:
            sets.append("auth_type = %s"); params.append(auth_type)
        if auth_config is not None:
            sets.append("auth_config = %s::jsonb"); params.append(json.dumps(auth_config))
        if not sets:
            return
        sets.append("updated_at = NOW()")
        params.append(integration_id)
        self._db.execute(
            f"UPDATE ag_api.integrations SET {', '.join(sets)} WHERE integration_id = %s",
            tuple(params))

    def update_operation(self, integration_id: int, operation_id: str, *,
                         enabled: Optional[bool] = None,
                         side_effect: Optional[str] = None,
                         requires_approval: Optional[bool] = None,
                         embed_safe: Optional[bool] = None) -> None:
        sets, params = [], []
        for col, val in (("enabled", enabled), ("side_effect", side_effect),
                         ("requires_approval", requires_approval),
                         ("embed_safe", embed_safe)):
            if val is not None:
                sets.append(f"{col} = %s"); params.append(val)
        if not sets:
            return
        params += [integration_id, operation_id]
        self._db.execute(
            f"UPDATE ag_api.operations SET {', '.join(sets)} "
            f"WHERE integration_id = %s AND operation_id = %s", tuple(params))

    def delete_integration(self, integration_id: int) -> None:
        self._db.execute("DELETE FROM ag_api.integrations WHERE integration_id = %s",
                         (integration_id,))

    def add_grant(self, integration_id: int, scope: str, ref: str,
                  op_filter: Optional[list], created_by: Optional[int]) -> int:
        row = self._db.execute_returning(
            """INSERT INTO ag_api.integration_grants
                   (integration_id, scope, ref, op_filter, created_by)
               VALUES (%s,%s,%s,%s::jsonb,%s)
               ON CONFLICT (integration_id, scope, ref) DO UPDATE SET
                   op_filter = EXCLUDED.op_filter
               RETURNING grant_id""",
            (integration_id, scope, ref or "",
             json.dumps(op_filter) if op_filter is not None else None, created_by))
        return row["grant_id"]

    def list_grants(self, integration_id: int) -> List[dict]:
        return self._db.fetch_all(
            "SELECT grant_id, scope, ref, op_filter, created_at "
            "FROM ag_api.integration_grants WHERE integration_id = %s ORDER BY grant_id",
            (integration_id,))

    def delete_grant(self, grant_id: int) -> None:
        self._db.execute("DELETE FROM ag_api.integration_grants WHERE grant_id = %s",
                         (grant_id,))

    def first_probe_op(self, auth_provider: str) -> Optional[dict]:
        """A cheap GET op for the integration that uses this auth provider —
        used by the generic 'Test connection' verifier."""
        return self._db.fetch_one(
            """SELECT o.tool_name, o.http_method, o.path_template, o.request_map,
                      o.parameters_schema,
                      i.base_url, i.auth_config, i.security_scheme, i.auth_type,
                      i.default_headers, i.egress_allow, i.allow_private
               FROM ag_api.operations o
               JOIN ag_api.integrations i ON i.integration_id = o.integration_id
               WHERE i.auth_provider = %s AND o.enabled = TRUE
               ORDER BY (o.http_method = 'GET') DESC, o.operation_pk
               LIMIT 1""",
            (auth_provider,))
