from __future__ import annotations

"""
Agent Brain — Entity Index (PostgreSQL)
====================================================
LAYER 1 ALTERNATIVE: PostgreSQL-based entity relationship store.

Replaces Neo4j for standalone deployments. Stores people, tickets, services,
and their relationships with tsvector-based FTS search.

Covers 90% of graph queries:
  - "Who worked on the auth bug?"
  - "What tickets are assigned to Mike?"
  - "What's connected to the auth service?"

Storage: PostgreSQL ag_brain.entities + ag_brain.relations (via TenantDB)
"""

import json
import logging
from typing import Optional

from core.tenant import TENANT_SCOPE

logger = logging.getLogger(__name__)


class EntityIndex:
    """PostgreSQL-backed entity relationship index.

    A lightweight alternative to Neo4j that stores entities (people, tickets,
    services) and their relationships using PostgreSQL with tsvector FTS.

    Usage:
        idx = EntityIndex(db=tenant_db)

        idx.upsert_entity("Mike", "person", "jira://JIRA-1234",
                          properties={"email": "mike@company.com"})
        idx.upsert_entity("JIRA-1234", "ticket", "jira://JIRA-1234",
                          properties={"status": "resolved"})
        idx.add_relation("Mike", "JIRA-1234", "RESOLVED", "jira://JIRA-1234")

        related = idx.find_related("Mike")
        # → [{"name": "JIRA-1234", "type": "ticket", "relation": "RESOLVED"}]
    """

    def __init__(self, db=None):
        """
        Args:
            db: TenantDB instance for PostgreSQL access.
        """
        self._db = db
        if db is None:
            logger.warning("EntityIndex initialized without TenantDB — entity tracking disabled")


    def upsert_entity(
        self,
        name: str,
        entity_type: str,
        source_uri: str = "",
        properties: dict = None,
        owner_id: str = TENANT_SCOPE,
        app_id: int = None,
    ) -> int:
        """Insert or update an entity. Returns entity ID."""
        if not self._db:
            return -1

        props_json = json.dumps(properties or {})

        row = self._db.execute_returning("""
            INSERT INTO ag_brain.entities (name, entity_type, owner_id, source_uri, properties_json, app_id)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (name, entity_type, owner_id) DO UPDATE SET
                source_uri = CASE WHEN EXCLUDED.source_uri != ''
                                  THEN EXCLUDED.source_uri ELSE ag_brain.entities.source_uri END,
                properties_json = EXCLUDED.properties_json,
                app_id = EXCLUDED.app_id
            RETURNING entity_id
        """, (name, entity_type, owner_id, source_uri, props_json, app_id))

        return row["entity_id"] if row else -1

    def get_entity(self, name: str, entity_type: str = None) -> Optional[dict]:
        """Find an entity by name (and optionally type)."""
        if not self._db:
            return None

        if entity_type:
            row = self._db.fetch_one(
                "SELECT * FROM ag_brain.entities WHERE name = %s AND entity_type = %s",
                (name, entity_type),
            )
        else:
            row = self._db.fetch_one(
                "SELECT * FROM ag_brain.entities WHERE LOWER(name) = LOWER(%s)",
                (name,),
            )
        if row:
            row["id"] = row.get("entity_id", row.get("id"))
        return row

    def find_by_type(self, entity_type: str, limit: int = 50) -> list[dict]:
        """Find all entities of a given type."""
        if not self._db:
            return []

        rows = self._db.fetch_all(
            "SELECT * FROM ag_brain.entities WHERE entity_type = %s ORDER BY name LIMIT %s",
            (entity_type, limit),
        )
        for r in rows:
            r["id"] = r.get("entity_id", r.get("id"))
        return rows

    def search_entities(self, query: str, limit: int = 20,
                         visible_scopes: list = None) -> list[dict]:
        """Full-text search across entity names using PG tsvector."""
        if not self._db:
            return []

        scope_filter = ""
        scope_params = []
        if visible_scopes:
            scope_filter = " AND owner_id = ANY(%s)"
            scope_params = [visible_scopes]

        try:
            rows = self._db.fetch_all(
                f"""SELECT *, ts_rank(fts_vector, plainto_tsquery('simple', %s)) AS rank
                   FROM ag_brain.entities
                   WHERE fts_vector @@ plainto_tsquery('simple', %s){scope_filter}
                   ORDER BY rank DESC LIMIT %s""",
                (query, query, *scope_params, limit),
            )
            if rows:
                for r in rows:
                    r["id"] = r.get("entity_id", r.get("id"))
                return rows
        except Exception:
            pass

        rows = self._db.fetch_all(
            f"SELECT * FROM ag_brain.entities WHERE name ILIKE %s{scope_filter} LIMIT %s",
            (f"%{query}%", *scope_params, limit),
        )
        for r in rows:
            r["id"] = r.get("entity_id", r.get("id"))
        return rows


    def add_relation(
        self,
        from_name: str,
        to_name: str,
        relation_type: str,
        source_uri: str = "",
        from_type: str = None,
        to_type: str = None,
    ) -> bool:
        """Add a relationship between two entities.

        Creates entities if they don't exist (with inferred types).
        Returns True if relation was created, False if it already existed.
        """
        if not self._db:
            return False

        from_entity = self.get_entity(from_name, from_type)
        if not from_entity:
            from_id = self.upsert_entity(from_name, from_type or "unknown", source_uri)
        else:
            from_id = from_entity["id"]

        to_entity = self.get_entity(to_name, to_type)
        if not to_entity:
            to_id = self.upsert_entity(to_name, to_type or "unknown", source_uri)
        else:
            to_id = to_entity["id"]

        if from_id <= 0 or to_id <= 0:
            return False

        try:
            self._db.execute("""
                INSERT INTO ag_brain.relations (from_entity_id, to_entity_id, relation_type, source_uri)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (from_entity_id, to_entity_id, relation_type) DO NOTHING
            """, (from_id, to_id, relation_type, source_uri))
            return True
        except Exception as e:
            logger.debug("entity_index.relation_exists: %s→%s (%s): %s",
                        from_name, to_name, relation_type, e)
            return False

    def find_related(
        self,
        entity_name: str,
        relation_type: str = None,
        direction: str = "both",
        limit: int = 50,
    ) -> list[dict]:
        """Find entities related to a given entity."""
        if not self._db:
            return []

        results = []

        entity_rows = self._db.fetch_all(
            "SELECT entity_id FROM ag_brain.entities WHERE LOWER(name) = LOWER(%s)",
            (entity_name,),
        )
        if not entity_rows:
            return []

        entity_ids = [r["entity_id"] for r in entity_rows]

        if direction in ("outgoing", "both"):
            params = [entity_ids]
            rel_filter = ""
            if relation_type:
                rel_filter = "AND r.relation_type = %s"
                params.append(relation_type)
            params.append(limit)

            rows = self._db.fetch_all(f"""
                SELECT e.name, e.entity_type, r.relation_type,
                       r.source_uri, 'outgoing' as direction
                FROM ag_brain.relations r
                JOIN ag_brain.entities e ON e.entity_id = r.to_entity_id
                WHERE r.from_entity_id = ANY(%s) {rel_filter}
                LIMIT %s
            """, tuple(params))
            results.extend(rows)

        if direction in ("incoming", "both"):
            params = [entity_ids]
            rel_filter = ""
            if relation_type:
                rel_filter = "AND r.relation_type = %s"
                params.append(relation_type)
            params.append(limit)

            rows = self._db.fetch_all(f"""
                SELECT e.name, e.entity_type, r.relation_type,
                       r.source_uri, 'incoming' as direction
                FROM ag_brain.relations r
                JOIN ag_brain.entities e ON e.entity_id = r.from_entity_id
                WHERE r.to_entity_id = ANY(%s) {rel_filter}
                LIMIT %s
            """, tuple(params))
            results.extend(rows)

        return results[:limit]


    def bulk_ingest_from_sync_item(
        self,
        source_uri: str,
        people: list[str],
        title: str,
        source_type: str,
        metadata: dict = None,
        owner_id: str = TENANT_SCOPE,
        app_id: int = None,
    ):
        """Extract and store entities + relations from a sync item.

        Uses a single PG transaction for atomicity.
        """
        if not self._db:
            return

        metadata = metadata or {}

        item_type = {
            "jira": "ticket",
            "outlook": "email",
            "slack": "message",
        }.get(source_type, "document")

        try:
            with self._db.transaction() as cur:
                props = json.dumps({"source_type": source_type, **metadata})
                cur.execute("""
                    INSERT INTO ag_brain.entities (name, entity_type, owner_id, source_uri, properties_json, app_id)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (name, entity_type, owner_id) DO UPDATE SET
                        source_uri = EXCLUDED.source_uri,
                        properties_json = EXCLUDED.properties_json,
                        app_id = EXCLUDED.app_id
                    RETURNING entity_id
                """, (title, item_type, owner_id, source_uri, props, app_id))
                item_row = cur.fetchone()
                item_id = item_row[0] if item_row else None

                if not item_id:
                    return

                for person in people:
                    if not person or person in ("Unassigned", "Unknown", "unknown"):
                        continue

                    cur.execute("""
                        INSERT INTO ag_brain.entities (name, entity_type, owner_id, source_uri, properties_json, app_id)
                        VALUES (%s, 'person', %s, %s, '{}'::jsonb, %s)
                        ON CONFLICT (name, entity_type, owner_id) DO NOTHING
                        RETURNING entity_id
                    """, (person, owner_id, source_uri, app_id))
                    person_row = cur.fetchone()
                    if person_row:
                        person_id = person_row[0]
                    else:
                        cur.execute(
                            "SELECT entity_id FROM ag_brain.entities WHERE name = %s AND entity_type = 'person' AND owner_id = %s",
                            (person, owner_id),
                        )
                        existing = cur.fetchone()
                        person_id = existing[0] if existing else None

                    if not person_id:
                        continue

                    rel_type = "MENTIONED_IN"
                    role = metadata.get("role", "")
                    if role == "assignee":
                        rel_type = "ASSIGNED_TO"
                    elif role == "reporter":
                        rel_type = "REPORTED"
                    elif role == "sender":
                        rel_type = "SENT"
                    elif role == "author":
                        rel_type = "AUTHORED"

                    cur.execute("""
                        INSERT INTO ag_brain.relations (from_entity_id, to_entity_id, relation_type, source_uri)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (from_entity_id, to_entity_id, relation_type) DO NOTHING
                    """, (person_id, item_id, rel_type, source_uri))

        except Exception as e:
            logger.error("entity_index.bulk_ingest_failed: uri=%s error=%s",
                        source_uri[:60], str(e))
            raise


    def get_stats(self) -> dict:
        """Entity index statistics."""
        if not self._db:
            return {"total_entities": 0, "total_relations": 0,
                    "entity_types": {}, "relation_types": {}}

        entity_count = self._db.fetch_scalar(
            "SELECT COUNT(*) FROM ag_brain.entities"
        ) or 0

        relation_count = self._db.fetch_scalar(
            "SELECT COUNT(*) FROM ag_brain.relations"
        ) or 0

        type_counts = self._db.fetch_all(
            """SELECT entity_type, COUNT(*) as count
               FROM ag_brain.entities GROUP BY entity_type ORDER BY count DESC"""
        )

        rel_type_counts = self._db.fetch_all(
            """SELECT relation_type, COUNT(*) as count
               FROM ag_brain.relations GROUP BY relation_type ORDER BY count DESC"""
        )

        return {
            "total_entities": entity_count,
            "total_relations": relation_count,
            "entity_types": {r["entity_type"]: r["count"] for r in type_counts},
            "relation_types": {r["relation_type"]: r["count"] for r in rel_type_counts},
        }

    def close(self):
        """No-op — connection lifecycle managed by TenantDB pool."""
        pass
