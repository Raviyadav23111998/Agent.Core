from __future__ import annotations

"""
Agent Brain — Document Store (PostgreSQL)
=====================================================
The foundational storage layer. Every document ingested into the brain
lands here FIRST as raw text. This is the source-of-truth.

Design:
  - PostgreSQL via TenantDB (connection-pooled, production-grade)
  - Full-text search via PG tsvector + GIN index (auto-generated column)
  - SHA-256 content-hash dedup prevents re-ingestion
  - Shared between Agent.Core (chat agent) and Agent.API (web UI)

Multi-tenant access control:
  - Each document has an owner_id scoping it within the tenant's DB:
      '__tenant__'     → org-wide (Jira, Confluence, Slack)
      'user_{id}'      → user's private data (Outlook, uploads)
      'group_{id}'     → group-specific data
  - Read queries accept visible_scopes: list of owner_ids the user can see
  - Write operations require explicit owner_id

NOTE: This replaces the previous SQLite-based implementation. PostgreSQL is
the single source of truth, ensuring both the Chat Agent and Web Dashboard
always see the same documents.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List

from brain.models import Document, DocumentType, IngestionStatus
from brain.exceptions import DocumentNotFoundError, DuplicateDocumentError
from core.tenant import TENANT_SCOPE

logger = logging.getLogger(__name__)

_STATUS_TO_PG = {
    IngestionStatus.PENDING: "pending",
    IngestionStatus.PARSING: "pending",
    IngestionStatus.EXTRACTING: "pending",
    IngestionStatus.INDEXING: "pending",
    IngestionStatus.COMPLETED: "indexed",
    IngestionStatus.FAILED: "error",
}

_PG_TO_STATUS = {
    "pending": IngestionStatus.PENDING,
    "indexed": IngestionStatus.COMPLETED,
    "error": IngestionStatus.FAILED,
    "archived": IngestionStatus.COMPLETED,
}


class DocumentStore:
    """PostgreSQL-backed document storage via TenantDB.

    All operations go through the shared ag_brain.documents table,
    ensuring the Chat Agent and Web UI always see the same data.
    """

    def __init__(self, db):
        """
        Args:
            db: A TenantDB instance (already connected to PostgreSQL).
        """
        self._db = db
        logger.info("document_store.initialized: backend=postgresql (ag_brain.documents)")

    COMMON_OWNER = TENANT_SCOPE


    def get_watermark(self, source_key: str) -> Optional[str]:
        """Get the last sync timestamp for a data source."""
        row = self._db.fetch_one(
            "SELECT last_sync_at FROM ag_brain.sync_state WHERE source_key = %s",
            (source_key,),
        )
        return row["last_sync_at"].isoformat() if row and row.get("last_sync_at") else None

    def set_watermark(self, source_key: str, timestamp: str, items_synced: int = 0):
        """Update the sync watermark for a data source."""
        self._db.execute("""
            INSERT INTO ag_brain.sync_state (source_key, last_sync_at, docs_synced, status)
            VALUES (%s, %s, %s, 'idle')
            ON CONFLICT (source_key) DO UPDATE SET
                last_sync_at = EXCLUDED.last_sync_at,
                docs_synced = ag_brain.sync_state.docs_synced + EXCLUDED.docs_synced,
                status = 'idle'
        """, (source_key, timestamp, items_synced))


    def add_document(
        self,
        content: str,
        source_path: str,
        document_type: DocumentType,
        title: str = "",
        metadata: dict | None = None,
        deduplicate: bool = True,
        owner_id: str = TENANT_SCOPE,
        app_id: int = None,
    ) -> Document:
        """Add a new document to the store.

        Args:
            owner_id: Data owner within this tenant's DB.
                      '__tenant__' for org-wide data (Jira, Confluence).
                      'user_{id}' for private data (Outlook, uploads).
                      'group_{id}' for group-specific data.
            app_id: Application this document belongs to.

        Raises:
            DuplicateDocumentError: If content already exists and deduplicate=True
        """
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        metadata = metadata or {}

        if deduplicate:
            existing = self._find_by_hash(content_hash, owner_id=owner_id)
            if existing:
                raise DuplicateDocumentError(content_hash, str(existing["doc_id"]))

        token_count = len(content.split())
        pg_status = _STATUS_TO_PG.get(IngestionStatus.PENDING, "pending")
        doc_type_val = document_type.value if isinstance(document_type, DocumentType) else document_type

        row = self._db.execute_returning("""
            INSERT INTO ag_brain.documents
                (source_path, doc_type, title, content, metadata_json,
                 content_hash, token_count, status, owner_id, app_id)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)
            ON CONFLICT (source_path) DO UPDATE SET
                content = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                token_count = EXCLUDED.token_count,
                title = EXCLUDED.title,
                metadata_json = EXCLUDED.metadata_json,
                status = EXCLUDED.status,
                app_id = EXCLUDED.app_id
            RETURNING doc_id, source_path, doc_type, title, content,
                      metadata_json, content_hash, token_count, status,
                      owner_id, created_at, updated_at
        """, (
            source_path, doc_type_val, title, content,
            json.dumps(metadata), content_hash, token_count,
            pg_status, owner_id, app_id,
        ))

        doc = self._row_to_document(row)

        try:
            self._log_action(doc.id, "CREATED",
                             f"type={doc_type_val}, tokens={token_count}, owner={owner_id}, app={app_id}")
        except Exception:
            pass

        return doc

    def update_status(self, document_id: str, status: IngestionStatus):
        """Update the ingestion status of a document."""
        pg_status = _STATUS_TO_PG.get(status, "pending")
        doc_id = self._resolve_doc_id(document_id)
        result = self._db.execute(
            "UPDATE ag_brain.documents SET status = %s WHERE doc_id = %s",
            (pg_status, doc_id),
        )
        if result == 0:
            raise DocumentNotFoundError(document_id)
        self._log_action(document_id, "STATUS_CHANGE", f"new_status={pg_status}")

    def delete_document(self, document_id: str) -> bool:
        """Hard-delete a document and its ingestion log."""
        doc_id = self._resolve_doc_id(document_id)
        self._db.execute(
            "DELETE FROM ag_brain.ingestion_log WHERE doc_id = %s", (doc_id,)
        )
        result = self._db.execute(
            "DELETE FROM ag_brain.documents WHERE doc_id = %s", (doc_id,)
        )
        return result > 0


    def get_document(self, document_id: str) -> Document:
        """Fetch a single document by ID."""
        doc_id = self._resolve_doc_id(document_id)
        row = self._db.fetch_one(
            "SELECT * FROM ag_brain.documents WHERE doc_id = %s", (doc_id,)
        )
        if not row:
            raise DocumentNotFoundError(document_id)
        return self._row_to_document(row)

    def get_documents_by_type(self, document_type: DocumentType, limit: int = 100,
                              visible_scopes: List[str] = None,
                              app_id: int = None) -> List[Document]:
        """Fetch documents filtered by type, scoped to visible owner_ids and app."""
        doc_type_val = document_type.value if isinstance(document_type, DocumentType) else document_type
        params = [doc_type_val]
        where = "WHERE doc_type = %s"
        if visible_scopes:
            where += " AND owner_id = ANY(%s)"
            params.append(visible_scopes)
        if app_id:
            where += " AND app_id = %s"
            params.append(app_id)
        params.append(limit)
        rows = self._db.fetch_all(
            f"""SELECT * FROM ag_brain.documents
               {where}
               ORDER BY created_at DESC LIMIT %s""",
            tuple(params),
        )
        return [self._row_to_document(r) for r in rows]

    def get_documents_by_status(self, status: IngestionStatus, limit: int = 100) -> List[Document]:
        """Fetch documents filtered by ingestion status."""
        pg_status = _STATUS_TO_PG.get(status, "pending")
        rows = self._db.fetch_all(
            """SELECT * FROM ag_brain.documents
               WHERE status = %s
               ORDER BY created_at ASC LIMIT %s""",
            (pg_status, limit),
        )
        return [self._row_to_document(r) for r in rows]

    def get_documents_by_ids(self, document_ids: list[str]) -> List[Document]:
        """Fetch multiple documents by their IDs (for context bundle loading)."""
        if not document_ids:
            return []
        numeric_ids = [self._resolve_doc_id(did) for did in document_ids]
        rows = self._db.fetch_all(
            "SELECT * FROM ag_brain.documents WHERE doc_id = ANY(%s)",
            (numeric_ids,),
        )
        return [self._row_to_document(r) for r in rows]

    def count_documents(self, visible_scopes: List[str] = None, app_id: int = None) -> dict:
        """Get document count statistics, optionally scoped to visible owner_ids and app."""
        conditions = []
        params_list = []
        if visible_scopes:
            conditions.append("owner_id = ANY(%s)")
            params_list.append(visible_scopes)
        if app_id is not None:
            conditions.append("app_id = %s")
            params_list.append(app_id)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params = tuple(params_list) if params_list else None

        total = self._db.fetch_scalar(
            f"SELECT COUNT(*) FROM ag_brain.documents {where}", params
        ) or 0

        by_type = {}
        rows = self._db.fetch_all(
            f"SELECT doc_type, COUNT(*) as cnt FROM ag_brain.documents {where} GROUP BY doc_type",
            params,
        )
        for row in rows:
            by_type[row["doc_type"]] = row["cnt"]

        by_status = {}
        rows = self._db.fetch_all(
            f"SELECT status, COUNT(*) as cnt FROM ag_brain.documents {where} GROUP BY status",
            params,
        )
        for row in rows:
            by_status[row["status"]] = row["cnt"]

        total_tokens = self._db.fetch_scalar(
            f"SELECT COALESCE(SUM(token_count), 0) FROM ag_brain.documents {where}", params
        ) or 0

        by_owner = {}
        rows = self._db.fetch_all(
            f"SELECT owner_id, COUNT(*) as cnt FROM ag_brain.documents {where} GROUP BY owner_id",
            params,
        )
        for row in rows:
            by_owner[row["owner_id"]] = row["cnt"]

        return {
            "total_documents": total,
            "total_tokens": total_tokens,
            "by_type": by_type,
            "by_status": by_status,
            "by_owner": by_owner,
        }

    def search_by_title(self, query: str, limit: int = 20,
                        visible_scopes: List[str] = None) -> List[Document]:
        """Simple LIKE search on document titles, scoped to visible owner_ids."""
        if visible_scopes:
            rows = self._db.fetch_all(
                """SELECT * FROM ag_brain.documents
                   WHERE title ILIKE %s AND owner_id = ANY(%s)
                   ORDER BY created_at DESC LIMIT %s""",
                (f"%{query}%", visible_scopes, limit),
            )
        else:
            rows = self._db.fetch_all(
                """SELECT * FROM ag_brain.documents
                   WHERE title ILIKE %s
                   ORDER BY created_at DESC LIMIT %s""",
                (f"%{query}%", limit),
            )
        return [self._row_to_document(r) for r in rows]

    def search_fts(self, query: str, limit: int = 10,
                   visible_scopes: List[str] = None,
                   app_id: int = None) -> List[Document]:
        """Full-text search across title and content using PostgreSQL tsvector.

        Uses the auto-generated fts_vector column with GIN index for fast search.

        Args:
            visible_scopes: List of owner_ids the user can see.
                            If None, searches all (admin/unscoped).
            app_id: Application ID to scope search to.
                    If None, searches across all apps.
        """
        try:
            sanitized = query.replace("'", "").replace('"', '').replace('*', '')
            sanitized = sanitized.replace('(', '').replace(')', '')
            tokens = [t for t in sanitized.split() if len(t) > 1]
            if not tokens:
                return []
            fts_query = " ".join(tokens)

            conditions = ["fts_vector @@ plainto_tsquery('english', %s)"]
            params = [fts_query, fts_query]
            if visible_scopes:
                conditions.append("owner_id = ANY(%s)")
                params.append(visible_scopes)
            if app_id is not None:
                conditions.append("app_id = %s")
                params.append(app_id)
            params.append(limit)

            where = " AND ".join(conditions)
            rows = self._db.fetch_all(
                f"""SELECT *, ts_rank(fts_vector, plainto_tsquery('english', %s)) AS rank
                   FROM ag_brain.documents
                   WHERE {where}
                   ORDER BY rank DESC
                   LIMIT %s""",
                tuple(params),
            )
            return [self._row_to_document(r) for r in rows]
        except Exception:
            return self.search_by_title(query, limit, visible_scopes=visible_scopes)


    _DEFAULT_EMBED_DIM = 1536

    @property
    def EMBED_DIM(self) -> int:
        """Embedding dimension — MUST match the ag_brain.documents.embedding
        column. Config `llm.embedding_dimensions` (e.g. 768 for
        nomic-embed-text after migration V018); defaults to the original 1536.
        """
        try:
            from core.config import app_config
            d = app_config.get_int("llm.embedding_dimensions", 0)
            return d if d > 0 else self._DEFAULT_EMBED_DIM
        except Exception:
            return self._DEFAULT_EMBED_DIM

    @staticmethod
    def _to_pgvector(vector: List[float]) -> str:
        """Render a float list as a pgvector literal: '[0.1,0.2,…]'."""
        return "[" + ",".join(repr(float(x)) for x in vector) + "]"

    def update_embedding(self, document_id: str, vector: List[float],
                         model: str = "") -> bool:
        """Store a document's embedding. No-op (logs) on dimension mismatch.

        Best-effort: a misconfigured embedding model must degrade (skip)
        rather than break ingestion.
        """
        if not vector:
            return False
        if len(vector) != self.EMBED_DIM:
            logger.warning(
                "document_store.embedding_dim_mismatch: doc=%s got=%d expected=%d "
                "(skipping — a migration is needed to change the column dimension)",
                document_id, len(vector), self.EMBED_DIM)
            return False
        try:
            doc_id = self._resolve_doc_id(document_id)
            self._db.execute(
                "UPDATE ag_brain.documents "
                "SET embedding = %s::vector, embedding_model = %s WHERE doc_id = %s",
                (self._to_pgvector(vector), model or None, doc_id),
            )
            return True
        except Exception as e:
            logger.warning("document_store.update_embedding_failed: doc=%s err=%s",
                           document_id, e)
            return False

    def search_vector(self, query_vector: List[float], limit: int = 10,
                      visible_scopes: List[str] = None,
                      app_id: int = None) -> List[Document]:
        """Semantic KNN search via cosine distance (<=>) over embeddings.

        Only considers rows that have been embedded. Returns [] when no
        embeddings exist yet (degraded → caller falls back to FTS).
        """
        if not query_vector or len(query_vector) != self.EMBED_DIM:
            return []
        try:
            conditions = ["embedding IS NOT NULL"]
            params: list = [self._to_pgvector(query_vector)]
            if visible_scopes:
                conditions.append("owner_id = ANY(%s)")
                params.append(visible_scopes)
            if app_id is not None:
                conditions.append("app_id = %s")
                params.append(app_id)
            params.append(limit)
            where = " AND ".join(conditions)
            rows = self._db.fetch_all(
                f"""SELECT *, (embedding <=> %s::vector) AS distance
                    FROM ag_brain.documents
                    WHERE {where}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s""",
                tuple([params[0]] + params[1:-1] + [params[0], params[-1]]),
            )
            return [self._row_to_document(r) for r in rows]
        except Exception as e:
            logger.warning("document_store.search_vector_failed: %s", e)
            return []


    def replace_chunks(self, document_id: str, chunks: List[dict],
                       model: str = "") -> int:
        """Replace all chunks for a document. chunks = [{content, embedding}].

        Atomic per-doc: deletes the doc's old chunks then inserts the new set,
        so re-ingesting a document doesn't leave stale chunks. embedding may be
        None (chunk stored but not vector-searchable until backfilled).
        """
        doc_id = self._resolve_doc_id(str(document_id))
        if not doc_id:
            return 0
        try:
            self._db.execute(
                "DELETE FROM ag_brain.document_chunks WHERE doc_id = %s", (doc_id,))
            n = 0
            for i, ch in enumerate(chunks):
                content = (ch.get("content") or "").strip()
                if not content:
                    continue
                vec = ch.get("embedding")
                emb = self._to_pgvector(vec) if (vec and len(vec) == self.EMBED_DIM) else None
                self._db.execute(
                    """INSERT INTO ag_brain.document_chunks
                           (doc_id, chunk_index, content, token_count,
                            embedding, embedding_model)
                       VALUES (%s, %s, %s, %s, %s::vector, %s)
                       ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
                           content = EXCLUDED.content,
                           token_count = EXCLUDED.token_count,
                           embedding = EXCLUDED.embedding,
                           embedding_model = EXCLUDED.embedding_model""",
                    (doc_id, i, content, max(len(content) // 4, 1), emb,
                     model or None),
                )
                n += 1
            return n
        except Exception as e:
            logger.warning("document_store.replace_chunks_failed: doc=%s err=%s",
                           document_id, e)
            return 0

    def search_chunks_vector(self, query_vector: List[float], limit: int = 10,
                             visible_scopes: List[str] = None,
                             app_id: int = None, pool: int = None) -> List[Document]:
        """KNN over CHUNK embeddings → return PARENT documents (parent-child).

        Matches the precise passage, then collapses to the best chunk per
        parent document. The matched chunk text is attached to the returned
        Document as `.content_snippet` so the synthesizer/citations show the
        exact supporting passage. Returns [] (degraded) on any error or when
        no chunks are embedded.
        """
        if not query_vector or len(query_vector) != self.EMBED_DIM:
            return []
        pool = pool or max(limit * 4, 20)
        try:
            conditions = ["c.embedding IS NOT NULL"]
            qv = self._to_pgvector(query_vector)
            params: list = [qv]
            if visible_scopes:
                conditions.append("d.owner_id = ANY(%s)")
                params.append(visible_scopes)
            if app_id is not None:
                conditions.append("d.app_id = %s")
                params.append(app_id)
            where = " AND ".join(conditions)
            params += [qv, pool]
            rows = self._db.fetch_all(
                f"""
                SELECT * FROM (
                    SELECT DISTINCT ON (d.doc_id) d.*,
                           c.content AS _chunk_content,
                           (c.embedding <=> %s::vector) AS distance
                    FROM ag_brain.document_chunks c
                    JOIN ag_brain.documents d ON d.doc_id = c.doc_id
                    WHERE {where}
                    ORDER BY d.doc_id, c.embedding <=> %s::vector
                ) best
                ORDER BY best.distance
                LIMIT %s
                """,
                tuple(params),
            )
            docs = []
            for r in rows:
                snippet = r.pop("_chunk_content", None)
                doc = self._row_to_document(r)
                if snippet:
                    setattr(doc, "content_snippet", snippet[:500])
                docs.append(doc)
            return docs[:limit]
        except Exception as e:
            logger.warning("document_store.search_chunks_vector_failed: %s", e)
            return []

    def count_chunks(self, app_id: int = None) -> dict:
        """Return {'docs_with_chunks': N, 'embedded_chunks': M}."""
        try:
            scope = "WHERE d.app_id = %s" if app_id is not None else ""
            args = (app_id,) if app_id is not None else ()
            row = self._db.fetch_one(
                f"""SELECT COUNT(DISTINCT c.doc_id) AS docs_with_chunks,
                           COUNT(c.embedding) AS embedded_chunks
                    FROM ag_brain.document_chunks c
                    JOIN ag_brain.documents d ON d.doc_id = c.doc_id {scope}""",
                args)
            return {"docs_with_chunks": row.get("docs_with_chunks", 0),
                    "embedded_chunks": row.get("embedded_chunks", 0)}
        except Exception as e:
            logger.warning("document_store.count_chunks_failed: %s", e)
            return {"docs_with_chunks": 0, "embedded_chunks": 0}

    def get_documents_without_chunks(self, limit: int = 100,
                                     app_id: int = None) -> List[Document]:
        """Docs that have no chunks yet (for the chunk backfill)."""
        try:
            scope = "AND d.app_id = %s" if app_id is not None else ""
            args = [app_id, limit] if app_id is not None else [limit]
            rows = self._db.fetch_all(
                f"""SELECT d.* FROM ag_brain.documents d
                    WHERE NOT EXISTS (
                        SELECT 1 FROM ag_brain.document_chunks c WHERE c.doc_id = d.doc_id)
                      {scope}
                    ORDER BY d.doc_id LIMIT %s""",
                tuple(args))
            return [self._row_to_document(r) for r in rows]
        except Exception as e:
            logger.warning("document_store.get_documents_without_chunks_failed: %s", e)
            return []

    def count_embedded(self, app_id: int = None) -> dict:
        """Return {'total': N, 'embedded': M} for backfill/status reporting."""
        try:
            scope = "WHERE app_id = %s" if app_id is not None else ""
            args = (app_id,) if app_id is not None else ()
            row = self._db.fetch_one(
                f"""SELECT COUNT(*) AS total,
                           COUNT(embedding) AS embedded
                    FROM ag_brain.documents {scope}""", args)
            return {"total": row.get("total", 0), "embedded": row.get("embedded", 0)}
        except Exception as e:
            logger.warning("document_store.count_embedded_failed: %s", e)
            return {"total": 0, "embedded": 0}

    def get_documents_without_embedding(self, limit: int = 100,
                                        app_id: int = None) -> List[Document]:
        """Fetch indexed docs that have no embedding yet (for backfill)."""
        try:
            scope = "AND app_id = %s" if app_id is not None else ""
            args = [app_id, limit] if app_id is not None else [limit]
            rows = self._db.fetch_all(
                f"""SELECT * FROM ag_brain.documents
                    WHERE embedding IS NULL {scope}
                    ORDER BY doc_id ASC LIMIT %s""",
                tuple(args),
            )
            return [self._row_to_document(r) for r in rows]
        except Exception as e:
            logger.warning("document_store.get_unembedded_failed: %s", e)
            return []

    def get_recent_documents(self, limit: int = 20,
                             visible_scopes: List[str] = None) -> List[Document]:
        """Fetch the most recently ingested documents, scoped to visible owner_ids."""
        if visible_scopes:
            rows = self._db.fetch_all(
                """SELECT * FROM ag_brain.documents
                   WHERE owner_id = ANY(%s)
                   ORDER BY created_at DESC LIMIT %s""",
                (visible_scopes, limit),
            )
        else:
            rows = self._db.fetch_all(
                "SELECT * FROM ag_brain.documents ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
        return [self._row_to_document(r) for r in rows]

    def find_by_source_path(self, source_path: str) -> Optional[Document]:
        """Find a document by its source_path (unique source URI).

        Used by SyncWorker to check if an item already exists before upserting.
        """
        row = self._db.fetch_one(
            "SELECT * FROM ag_brain.documents WHERE source_path = %s LIMIT 1",
            (source_path,),
        )
        return self._row_to_document(row) if row else None

    def update_document(
        self,
        doc_id: str,
        content: str = None,
        title: str = None,
        metadata: dict = None,
    ):
        """Update an existing document's content, title, or metadata.

        Used by SyncWorker to update summaries on re-sync.
        Only updates fields that are provided (non-None).
        """
        updates = []
        params = []

        if content is not None:
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            token_count = len(content.split())
            updates.extend(["content = %s", "content_hash = %s", "token_count = %s"])
            params.extend([content, content_hash, token_count])

        if title is not None:
            updates.append("title = %s")
            params.append(title)

        if metadata is not None:
            updates.append("metadata_json = %s::jsonb")
            params.append(json.dumps(metadata))

        if not updates:
            return

        resolved_id = self._resolve_doc_id(doc_id)
        params.append(resolved_id)

        self._db.execute(
            f"UPDATE ag_brain.documents SET {', '.join(updates)} WHERE doc_id = %s",
            tuple(params),
        )
        self._log_action(doc_id, "UPDATED", f"fields={len(updates)}")


    def _resolve_doc_id(self, document_id: str) -> int:
        """Resolve a document_id (which may be a UUID or a numeric PG ID) to a PG doc_id.

        The old SQLite store used UUIDs. The new PG store uses BIGINT serial.
        For backward compatibility, if a UUID is passed, we look it up by source_path
        or just try to cast it.
        """
        try:
            return int(document_id)
        except (ValueError, TypeError):
            row = self._db.fetch_one(
                "SELECT doc_id FROM ag_brain.documents WHERE source_path = %s LIMIT 1",
                (document_id,),
            )
            if row:
                return row["doc_id"]
            return 0

    def _find_by_hash(self, content_hash: str, owner_id: str = None) -> Optional[dict]:
        """Find a document by content hash, optionally scoped to owner."""
        if owner_id:
            row = self._db.fetch_one(
                "SELECT doc_id, source_path FROM ag_brain.documents WHERE content_hash = %s AND owner_id = %s",
                (content_hash, owner_id),
            )
        else:
            row = self._db.fetch_one(
                "SELECT doc_id, source_path FROM ag_brain.documents WHERE content_hash = %s",
                (content_hash,),
            )
        return dict(row) if row else None

    def _log_action(self, document_id: str, action: str, details: str = ""):
        """Log an ingestion action. Non-critical — failures are silently ignored."""
        try:
            doc_id = self._resolve_doc_id(str(document_id))
            if doc_id == 0:
                return
            self._db.execute(
                """INSERT INTO ag_brain.ingestion_log (doc_id, action, details)
                   VALUES (%s, %s, %s)""",
                (doc_id, action, details),
            )
        except Exception:
            pass

    @staticmethod
    def _row_to_document(row: dict) -> Document:
        """Convert a PostgreSQL row dict to a brain Document model."""
        if row is None:
            raise DocumentNotFoundError("None")

        metadata = row.get("metadata_json", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}

        pg_status = row.get("status", "pending")
        status = _PG_TO_STATUS.get(pg_status, IngestionStatus.PENDING)

        doc_type_str = row.get("doc_type", "document")
        try:
            doc_type = DocumentType(doc_type_str)
        except ValueError:
            doc_type = DocumentType.DOCUMENT

        created_at = row.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        updated_at = row.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now(timezone.utc)

        return Document(
            id=str(row.get("doc_id", "")),
            source_path=row.get("source_path", ""),
            document_type=doc_type,
            title=row.get("title", ""),
            content=row.get("content", ""),
            metadata=metadata,
            content_hash=row.get("content_hash", ""),
            token_count=row.get("token_count", 0),
            status=status,
            created_at=created_at,
            updated_at=updated_at,
        )

    def close(self):
        """No-op — connection lifecycle managed by TenantDB pool."""
        pass
