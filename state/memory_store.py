from __future__ import annotations

"""
Agent — Persistent Memory Store (PostgreSQL)
=========================================================
PostgreSQL-backed memory engine with:
  - Connection pooling via TenantDB
  - FTS via PG tsvector + GIN index (auto-generated column)
  - 4-layer memory: session → episodic → semantic → company
  - User isolation via user_id column
  - Entity-relationship graph for structured knowledge
  - Auto-categorization from content signals

Storage: PostgreSQL ag_memory.memories + ag_memory.memory_tags +
         ag_memory.entities + ag_memory.relations (via TenantDB)
"""

import hashlib
import logging
from typing import List, Dict, Any

from core.tenant import TENANT_SCOPE

logger = logging.getLogger(__name__)

CATEGORIES = ("preference", "fact", "pattern", "correction", "qa_fact", "entity", "procedural")

LAYERS = ("session", "episodic", "semantic", "company")

_CATEGORY_SIGNALS = {
    "preference": [
        "prefer", "always use", "never use", "i like", "i want", "i don't like",
        "please use", "don't use", "use this", "switch to", "dark mode", "light mode",
    ],
    "correction": [
        "actually", "no,", "wrong", "not that", "instead", "fix this",
        "that's incorrect", "should be", "correct this",
    ],
    "pattern": [
        "pattern", "convention", "standard", "architecture", "design",
        "always do", "rule", "naming", "structure",
    ],
    "procedural": [
        "how to", "steps to", "to do this", "procedure", "workflow",
        "first you", "the way to", "recipe",
    ],
}

LAYER_LIMITS = {
    "session": 50,
    "episodic": 200,
    "semantic": 500,
    "company": 1000,
}

GLOBAL_USER = TENANT_SCOPE
CLI_USER = "__cli__"


def auto_categorize(content: str) -> str:
    """Auto-categorize memory content based on signal words."""
    lower = content.lower()
    for category, signals in _CATEGORY_SIGNALS.items():
        for signal in signals:
            if signal in lower:
                return category
    return "fact"


class MemoryStore:
    """PostgreSQL-backed persistent memory with tsvector FTS.

    Design principles:
      - Zero local state (all data in PostgreSQL via TenantDB)
      - Connection-pooled via TenantDB (no per-thread connections)
      - tsvector + GIN index for O(1) ranked search
      - Graceful degradation if TenantDB not provided
    """

    MAX_MEMORIES = 500

    SEMANTIC_POOL_FACTOR = 3

    def __init__(self, db=None):
        """
        Args:
            db: TenantDB instance for PostgreSQL access.
        """
        self._db = db
        if db is None:
            logger.warning("MemoryStore initialized without TenantDB — memory disabled")
        self._embedder = None
        try:
            from core.embeddings import EmbeddingClient
            self._embedder = EmbeddingClient()
        except Exception:
            pass

    def _semantic_rerank(self, query: str, rows: List[Dict[str, Any]],
                         limit: int) -> List[Dict[str, Any]]:
        """Reorder candidate memories by semantic similarity to the query.

        Keyword FTS retrieves the pool; embeddings pick the most RELEVANT —
        catching paraphrases ("dark theme" → "prefers dark mode") that
        tsvector ranking misses. Preserves FTS order on any failure.
        """
        if (not rows or len(rows) <= 1 or not self._embedder
                or not getattr(self._embedder, "available", False)):
            return rows[:limit]
        try:
            from core.embeddings import cosine_similarity
            texts = [f"{r.get('key','')} {r.get('value','')}" for r in rows]
            vectors = self._embedder.embed([query] + texts)
            if not vectors or len(vectors) != len(rows) + 1:
                return rows[:limit]
            q_vec = vectors[0]
            scored = sorted(
                zip((cosine_similarity(q_vec, v) for v in vectors[1:]), rows),
                key=lambda x: x[0], reverse=True,
            )
            return [r for _, r in scored[:limit]]
        except Exception as e:
            logger.debug("memory.semantic_rerank_failed: %s", e)
            return rows[:limit]


    def remember(self, key: str, value: str, user_id: str = CLI_USER,
                 category: str = None, layer: str = "semantic",
                 source: str = "", confidence: float = 1.0,
                 tags: List[str] = None, expires_at: str = None,
                 app_id=None) -> int:
        """Store or update a memory.

        Uses UPSERT on the unique (app_id, user_id, key) constraint.

        <b>app_id is written, not left to the column default.</b> The column
        defaults to 1, so before this every memory from every application was
        stamped as application 1 regardless of where it came from — which is
        why the column looked populated and consistent while carrying no
        information at all. A read filter over that data would have been
        worse than none: confidently wrong.

        None keeps the default, for the CLI and for callers that genuinely
        have no application.
        """
        if not self._db:
            return 0

        if category is None:
            category = auto_categorize(value)
        if layer not in LAYERS:
            layer = "semantic"

        # COALESCE, not a Python default: passing NULL must fall back to the
        # column's own default rather than writing NULL into a NOT NULL column.
        row = self._db.execute_returning("""
            INSERT INTO ag_memory.memories
                (user_id, app_id, layer, category, key, value, confidence, source, expires_at)
            VALUES (%s, COALESCE(%s, 1), %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (app_id, user_id, key) DO UPDATE SET
                value = EXCLUDED.value,
                category = EXCLUDED.category,
                layer = EXCLUDED.layer,
                confidence = EXCLUDED.confidence,
                source = EXCLUDED.source,
                expires_at = EXCLUDED.expires_at
            RETURNING memory_id
        """, (user_id, int(app_id) if app_id is not None else None,
              layer, category, key, value, confidence, source, expires_at))

        row_id = row["memory_id"] if row else 0

        if tags and row_id:
            for tag in tags:
                try:
                    self._db.execute(
                        "INSERT INTO ag_memory.memory_tags (memory_id, tag) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (row_id, tag),
                    )
                except Exception:
                    pass

        logger.debug("memory.stored key=%s category=%s layer=%s", key, category, layer)
        return row_id

    def _scope_filter(self, user_id: str, include_global: bool,
                      visible_scopes: List[str], app_id) -> tuple:
        """The WHERE fragment every read shares: WHO, and then WHICH APP.

        <b>The app half was missing and that was the leak.</b> ``app_id`` has
        been a column on ag_memory.memories since it was created, and no query
        had ever referenced it — so a turn belonging to one application could
        read another's memories as long as the two resolved the same user. That
        is not hypothetical: a service key spends as the human who minted it, so
        an app calling the platform on behalf of its own end users was reading
        that person's private bucket, and their notes reached a completely
        unrelated product.

        ``UserContext`` has carried ``app_id`` since cron needed it for
        credential resolution. This makes memory honour the identity the
        platform already establishes, rather than inventing a second one.

        app_id=None means "unscoped" and is NOT the same as "no results": the
        CLI and legacy callers have no application, and failing them closed
        would break local use to fix a server-side leak. Every caller that HAS
        an application passes it, which is the case that leaked.
        """
        if visible_scopes:
            where, params = "user_id = ANY(%s)", [visible_scopes]
        elif include_global and user_id != GLOBAL_USER:
            where, params = "(user_id = %s OR user_id = %s)", [user_id, GLOBAL_USER]
        else:
            where, params = "user_id = %s", [user_id]

        if app_id is not None:
            where += " AND app_id = %s"
            params.append(int(app_id))

        return where, params

    def recall(self, query: str, user_id: str = CLI_USER,
               category: str = None, layer: str = None,
               limit: int = 10, include_global: bool = True,
               visible_scopes: List[str] = None,
               app_id=None) -> List[Dict[str, Any]]:
        """Search memories using tsvector FTS or ILIKE fallback.

        Args:
            visible_scopes: List of user_ids to search across (Teams model).
                            If provided, overrides user_id + include_global.
            app_id:         Owning application. Reads are confined to it —
                            see _scope_filter for why this is not optional
                            wherever a caller has one.
        """
        if not self._db:
            return []

        user_filter, user_params = self._scope_filter(
            user_id, include_global, visible_scopes, app_id)

        exact = self._db.fetch_one(
            f"SELECT * FROM ag_memory.memories WHERE {user_filter} AND key = %s LIMIT 1",
            (*user_params, query),
        )
        if exact:
            self._record_access(exact.get("memory_id") or exact.get("id"))
            return [exact]

        try:
            sanitized = self._sanitize_fts_query(query)
            if sanitized:
                where_parts = [user_filter]
                where_params = list(user_params)

                if category:
                    where_parts.append("category = %s")
                    where_params.append(category)
                if layer:
                    where_parts.append("layer = %s")
                    where_params.append(layer)

                where_sql = " AND ".join(where_parts)
                fetch_limit = limit * self.SEMANTIC_POOL_FACTOR
                params = [sanitized] + where_params + [sanitized, fetch_limit]

                rows = self._db.fetch_all(f"""
                    SELECT *,
                           ts_rank(fts_vector, plainto_tsquery('english', %s))
                             + 0.05 * LN(1 + COALESCE(access_count, 0))
                             + 0.10 * EXP(
                                 -GREATEST(EXTRACT(EPOCH FROM (NOW() - COALESCE(last_accessed, created_at))), 0)
                                 / 2592000.0)
                           AS rank
                    FROM ag_memory.memories
                    WHERE {where_sql}
                      AND fts_vector @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                """, tuple(params))

                if rows:
                    rows = self._semantic_rerank(query, rows, limit)
                    for row in rows:
                        self._record_access(row.get("memory_id") or row.get("id"))
                    return rows
        except Exception:
            pass

        where_parts = [user_filter]
        params = list(user_params)

        if category:
            where_parts.append("category = %s")
            params.append(category)
        if layer:
            where_parts.append("layer = %s")
            params.append(layer)

        tokens = query.lower().split()[:5]
        if tokens:
            like_clauses = []
            for token in tokens:
                if len(token) > 2:
                    like_clauses.append("(LOWER(key) LIKE %s OR LOWER(value) LIKE %s)")
                    params.extend([f"%{token}%", f"%{token}%"])
            if like_clauses:
                where_parts.append(f"({' OR '.join(like_clauses)})")

        params.append(limit)
        where_sql = " AND ".join(where_parts)

        rows = self._db.fetch_all(f"""
            SELECT * FROM ag_memory.memories
            WHERE {where_sql}
            ORDER BY last_accessed DESC
            LIMIT %s
        """, tuple(params))

        for row in rows:
            self._record_access(row.get("memory_id") or row.get("id"))
        return rows

    def recall_by_category(self, user_id: str = CLI_USER, category: str = None,
                           layer: str = None, limit: int = 20,
                           visible_scopes: List[str] = None,
                           app_id=None) -> List[Dict[str, Any]]:
        """Retrieve memories by category/layer without search query.

        Shares _scope_filter with recall(): a second hand-written copy of the
        WHERE clause is exactly how one read path ends up app-scoped and its
        neighbour does not.
        """
        if not self._db:
            return []

        scope_where, params = self._scope_filter(
            user_id, include_global=True,
            visible_scopes=visible_scopes, app_id=app_id)
        where_parts = [scope_where]

        if category:
            where_parts.append("category = %s")
            params.append(category)
        if layer:
            where_parts.append("layer = %s")
            params.append(layer)

        params.append(limit)
        where_sql = " AND ".join(where_parts)

        return self._db.fetch_all(f"""
            SELECT * FROM ag_memory.memories
            WHERE {where_sql}
            ORDER BY last_accessed DESC
            LIMIT %s
        """, tuple(params))

    def forget(self, key: str, user_id: str = CLI_USER) -> bool:
        """Delete a specific memory by key."""
        if not self._db:
            return False
        count = self._db.execute(
            "DELETE FROM ag_memory.memories WHERE user_id = %s AND key = %s",
            (user_id, key),
        )
        if count > 0:
            logger.debug("memory.forgot key=%s", key)
        return count > 0

    def clear(self, user_id: str = None, layer: str = None):
        """Clear memories. Scoped by user and/or layer."""
        if not self._db:
            return
        if user_id and layer:
            self._db.execute(
                "DELETE FROM ag_memory.memories WHERE user_id = %s AND layer = %s",
                (user_id, layer),
            )
        elif user_id:
            self._db.execute(
                "DELETE FROM ag_memory.memories WHERE user_id = %s",
                (user_id,),
            )
        elif layer:
            self._db.execute(
                "DELETE FROM ag_memory.memories WHERE layer = %s",
                (layer,),
            )
        else:
            self._db.execute("DELETE FROM ag_memory.memories")
        logger.info("memory.cleared user=%s layer=%s", user_id, layer)


    def add_entity(self, name: str, entity_type: str,
                   user_id: str = GLOBAL_USER, properties: str = "{}") -> int:
        """Create or get an entity node."""
        if not self._db:
            return 0

        self._db.execute("""
            INSERT INTO ag_memory.entities (name, entity_type, user_id, properties)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name, entity_type, user_id) DO NOTHING
        """, (name, entity_type, user_id, properties))

        row = self._db.fetch_one(
            "SELECT entity_id FROM ag_memory.entities WHERE name = %s AND entity_type = %s AND user_id = %s",
            (name, entity_type, user_id),
        )
        return row["entity_id"] if row else 0

    def add_relation(self, source_name: str, source_type: str,
                     target_name: str, target_type: str,
                     relation_type: str, user_id: str = GLOBAL_USER,
                     properties: str = "{}") -> bool:
        """Create a relationship between two entities (creates entities if needed)."""
        if not self._db:
            return False

        source_id = self.add_entity(source_name, source_type, user_id)
        target_id = self.add_entity(target_name, target_type, user_id)

        if not source_id or not target_id:
            return False

        try:
            self._db.execute("""
                INSERT INTO ag_memory.relations (source_id, target_id, relation_type, properties)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_id, target_id, relation_type) DO NOTHING
            """, (source_id, target_id, relation_type, properties))
            return True
        except Exception:
            return False

    def get_entity_relations(self, entity_name: str,
                             user_id: str = GLOBAL_USER,
                             visible_scopes: List[str] = None) -> List[Dict[str, Any]]:
        """Get all relationships for an entity."""
        if not self._db:
            return []

        if visible_scopes:
            rows = self._db.fetch_all("""
                SELECT e1.name as source, e1.entity_type as source_type,
                       r.relation_type,
                       e2.name as target, e2.entity_type as target_type
                FROM ag_memory.relations r
                JOIN ag_memory.entities e1 ON r.source_id = e1.entity_id
                JOIN ag_memory.entities e2 ON r.target_id = e2.entity_id
                WHERE (LOWER(e1.name) = LOWER(%s) OR LOWER(e2.name) = LOWER(%s))
                  AND e1.user_id = ANY(%s)
                LIMIT 50
            """, (entity_name, entity_name, visible_scopes))
        else:
            rows = self._db.fetch_all("""
                SELECT e1.name as source, e1.entity_type as source_type,
                       r.relation_type,
                       e2.name as target, e2.entity_type as target_type
                FROM ag_memory.relations r
                JOIN ag_memory.entities e1 ON r.source_id = e1.entity_id
                JOIN ag_memory.entities e2 ON r.target_id = e2.entity_id
                WHERE (LOWER(e1.name) = LOWER(%s) OR LOWER(e2.name) = LOWER(%s))
                  AND (e1.user_id = %s OR e1.user_id = %s)
                LIMIT 50
            """, (entity_name, entity_name, user_id, GLOBAL_USER))
        return rows


    def get_context_memories(self, query: str, user_id: str = CLI_USER,
                             top_k: int = 5, visible_scopes: List[str] = None,
                             app_id=None) -> str:
        """Get top-K relevant memories as a prompt-ready string.

        This is the INJECTION path — what comes back is pasted into a prompt
        without anybody choosing it, so an unscoped read here is worse than an
        unscoped tool call: no agent decided to look, and nothing in the reply
        says where the context came from.
        """
        if not query or not self._db:
            return ""

        results = self.recall(query, user_id=user_id, limit=top_k,
                              visible_scopes=visible_scopes, app_id=app_id)
        if not results:
            return ""

        lines = []
        for mem in results:
            cat = mem.get("category", "?")
            val = mem.get("value", "")[:200]
            lines.append(f"- [{cat}] {val}")

        return "\n".join(lines)

    def get_user_context(self, user_id: str = CLI_USER,
                         visible_scopes: List[str] = None,
                         app_id=None) -> str:
        """Get ALL memories for a user as a prompt-ready string.

        The widest read in the class — everything, unfiltered by query — which
        makes it the one where a missing app scope leaks the most in a single
        call. Goes through _scope_filter for that reason rather than carrying
        its own third copy of the WHERE clause.
        """
        if not self._db:
            return ""

        scope_where, params = self._scope_filter(
            user_id, include_global=True,
            visible_scopes=visible_scopes, app_id=app_id)

        rows = self._db.fetch_all(f"""
            SELECT category, key, value FROM ag_memory.memories
            WHERE {scope_where}
            ORDER BY layer DESC, last_accessed DESC
            LIMIT 50
        """, tuple(params))

        if not rows:
            return ""

        lines = []
        for row in rows:
            lines.append(f"- [{row['category']}] {row['value'][:200]}")
        return "\n".join(lines)


    def add_memory(self, memory_item: str, category: str = None) -> str:
        """Legacy interface — compatible with old MemoryManager.add_memory()."""
        content_hash = hashlib.md5(memory_item.encode()).hexdigest()[:8]
        cat = category or auto_categorize(memory_item)
        key = f"{cat}.{content_hash}"

        self.remember(
            key=key,
            value=memory_item.strip(),
            category=cat,
            source="add_memory",
        )

        stats = self.get_stats()
        return f"Memory stored ({cat}). Total: {stats['total_memories']}"

    def get_memories(self, query: str = None, category: str = None) -> str:
        """Legacy interface — compatible with old MemoryManager.get_memories()."""
        if not self._db:
            return "No memories stored."

        if not query and not category:
            rows = self._db.fetch_all("""
                SELECT category, value FROM ag_memory.memories
                WHERE user_id = %s OR user_id = %s
                ORDER BY last_accessed DESC LIMIT 10
            """, (CLI_USER, GLOBAL_USER))

            if not rows:
                return "No memories stored."

            lines = [f"- [{r['category']}] {r['value'][:200]}" for r in rows]
            return "\n".join(lines)

        if query:
            results = self.recall(query, category=category)
        else:
            results = self.recall_by_category(category=category)

        if not results:
            return f"No memories matching '{query or category}'."

        lines = [f"- [{r.get('category', '?')}] {r.get('value', '')[:200]}" for r in results]
        return "\n".join(lines)

    def clear_memories(self) -> str:
        """Legacy interface — compatible with old MemoryManager.clear_memories()."""
        self.clear()
        return "All memories cleared."


    def get_stats(self) -> Dict[str, Any]:
        """Return memory system statistics."""
        if not self._db:
            return {"total_memories": 0, "by_layer": {}, "by_category": {},
                    "entities": 0, "relations": 0, "db_size_bytes": 0,
                    "db_size_human": "0 KB"}

        total = self._db.fetch_scalar(
            "SELECT COUNT(*) FROM ag_memory.memories"
        ) or 0

        by_layer = {}
        for row in self._db.fetch_all(
            "SELECT layer, COUNT(*) as cnt FROM ag_memory.memories GROUP BY layer"
        ):
            by_layer[row["layer"]] = row["cnt"]

        by_category = {}
        for row in self._db.fetch_all(
            "SELECT category, COUNT(*) as cnt FROM ag_memory.memories GROUP BY category"
        ):
            by_category[row["category"]] = row["cnt"]

        entity_count = self._db.fetch_scalar(
            "SELECT COUNT(*) FROM ag_memory.entities"
        ) or 0
        relation_count = self._db.fetch_scalar(
            "SELECT COUNT(*) FROM ag_memory.relations"
        ) or 0

        return {
            "total_memories": total,
            "by_layer": by_layer,
            "by_category": by_category,
            "entities": entity_count,
            "relations": relation_count,
            "db_size_bytes": 0,
            "db_size_human": "PostgreSQL",
        }


    def _record_access(self, memory_id: int):
        """Update access tracking for a memory."""
        if not memory_id or not self._db:
            return
        try:
            self._db.execute("""
                UPDATE ag_memory.memories SET
                    access_count = access_count + 1,
                    last_accessed = NOW()
                WHERE memory_id = %s
            """, (memory_id,))
        except Exception:
            pass

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize user input for PG plainto_tsquery."""
        cleaned = query.replace("'", "").replace('"', '').replace('*', '')
        cleaned = cleaned.replace('(', '').replace(')', '')
        cleaned = cleaned.replace(':', '').replace('^', '')
        tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 1]
        if not tokens:
            return ""
        return " ".join(tokens)

    def close(self):
        """No-op — connection lifecycle managed by TenantDB pool."""
        pass
