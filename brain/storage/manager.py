from __future__ import annotations

"""
Agent Brain — Storage Manager (PostgreSQL)
======================================================
Unified facade that coordinates all storage pillars:
  Pillar 0: Document Store (PostgreSQL) — Raw content source-of-truth
  Pillar 1: Graph Store (Neo4j)         — Entities & relationships
  Pillar 2: Lexical Store (ES)          — BM25 keyword search
  Pillar 3: Context Cache (FS)          — LLM context window bundles

All SQLite-based stores have been migrated to PostgreSQL via TenantDB.
The ingestion pipeline calls StorageManager to write across all stores.
The retrieval agents call StorageManager to read from specific stores.
"""

from typing import Optional
import logging

from brain.models import (
    Document, DocumentType, IngestionStatus,
    GraphExtractionResult, SearchDocument,
)
from brain.storage.document_store import DocumentStore
from brain.storage.context_cache import ContextCache
from brain.storage.content_store import ContentStore
from brain.storage.entity_index import EntityIndex
from core.tenant import TENANT_SCOPE

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Unified storage coordinator.
    Single entry point for all read/write operations across pillars.

    All primary stores use PostgreSQL via TenantDB.
    External services (Neo4j, Elasticsearch) are optional — the brain
    degrades gracefully if they're not configured.
    """

    def __init__(
        self,
        db=None,
        context_cache_dir: str = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        elasticsearch_url: Optional[str] = None,
        elasticsearch_index: str = "brain",
        graph_tenant_id: Optional[str] = None,
    ):
        """
        Args:
            db: TenantDB instance — shared PostgreSQL connection pool.
                All stores (documents, cache, entities, etc.) use this.
            graph_tenant_id: when set, the (shared) Neo4j graph is keyed by this
                tenant id (Mode C) — every node/edge carries it and reads filter it.
                None = single-tenant legacy graph. See ELASTIC_NEO4J_DESIGN §3.7.
        """
        self.documents = DocumentStore(db=db)

        self.context = ContextCache(db=db)

        self.content_cache = ContentStore(db=db)

        self.entity_index = EntityIndex(db=db)

        self.graph = None
        if neo4j_uri and not (neo4j_password or "").strip():
            # A URI with no credential is a misconfiguration, not a working graph:
            # Neo4j rejects an empty password with "Unauthorized / missing key
            # `credentials`" on every call. Treat the graph pillar as disabled and
            # say so once, instead of constructing it and spamming auth failures.
            logger.warning(
                "storage_manager.graph_disabled_no_credential: neo4j_uri=%s set but no "
                "password configured (NEO4J_PASSWORD / neo4j.password) — graph pillar disabled",
                neo4j_uri,
            )
        elif neo4j_uri:
            try:
                from brain.storage.graph_store import GraphStore
                self.graph = GraphStore(
                    uri=neo4j_uri,
                    user=neo4j_user or "neo4j",
                    password=neo4j_password,
                    tenant_id=graph_tenant_id,
                )
            except ImportError:
                logger.info("Neo4j driver not installed — graph store disabled")

        self.lexical = None
        if elasticsearch_url:
            try:
                from brain.storage.lexical_store import LexicalStore
                self.lexical = LexicalStore(
                    url=elasticsearch_url,
                    index_name=elasticsearch_index,
                )
            except ImportError:
                logger.info("Elasticsearch client not installed — lexical store disabled")

        try:
            from core.embeddings import EmbeddingClient
            self.embeddings = EmbeddingClient()
        except Exception:
            self.embeddings = None

        from brain.storage.index_state import IndexStateStore
        self._db = db
        self.index_state = IndexStateStore(db)
        self.extractor = None

        self._connected = False

    def _configured_pillars(self) -> dict:
        """Which optional pillars are wired in this manager. 'configured' means
        the dependency is present — NOT that it's reachable (the reconciler's
        preflight surfaces configured-but-unreachable as a failure, fail-loud)."""
        return {
            "lexical": self.lexical is not None,
            "graph": self.graph is not None,
            "vector": bool(self.embeddings and getattr(self.embeddings, "available", False)),
        }

    def connect(self):
        """Connect to all external stores (Neo4j, Elasticsearch).

        PostgreSQL stores are already connected via TenantDB pool.
        This method only initializes optional external services.
        """
        errors = []

        if self.graph:
            try:
                self.graph.connect()
            except Exception as e:
                errors.append(f"Neo4j: {e}")
                logger.warning("storage_manager.neo4j_unavailable: %s", e)

        if self.lexical:
            try:
                self.lexical.connect()
            except Exception as e:
                errors.append(f"Elasticsearch: {e}")
                logger.warning("storage_manager.elasticsearch_unavailable: %s", e)

        self._connected = True

        if errors:
            logger.warning("storage_manager.partial_connection: %s", errors)
        else:
            logger.info("storage_manager.fully_connected")

        return {"connected": True, "errors": errors}

    def disconnect(self):
        """Gracefully close external connections."""
        if self.graph:
            self.graph.close()
        if self.lexical:
            self.lexical.close()
        self._connected = False
        logger.info("storage_manager.disconnected")

    def evict_expired_cache(self) -> int:
        """Evict expired items from content cache. Called periodically."""
        return self.content_cache.evict_expired()


    def ingest_document(
        self,
        content: str,
        source_path: str,
        document_type: DocumentType,
        title: str = "",
        metadata: dict | None = None,
        graph_extraction: Optional[GraphExtractionResult] = None,
        owner_id: str = TENANT_SCOPE,
        app_id: int = 1,
    ) -> Document:
        """
        Full ingestion: store document → index for search → store graph data.

        This is the primary entry point for the ingestion pipeline.
        It writes to ALL available pillars in a single coordinated operation.

        Args:
            app_id: Application this document belongs to (default: 1 = Internal AI).
        """
        doc = self.documents.add_document(
            content=content,
            source_path=source_path,
            document_type=document_type,
            title=title,
            metadata=metadata,
            owner_id=owner_id,
            app_id=app_id,
        )
        self.documents.update_status(doc.id, IngestionStatus.INDEXING)

        configured = self._configured_pillars()
        self.index_state.seed(doc.id, configured)

        if configured["lexical"]:
            self._attempt_pillar(doc.id, "lexical",
                                 lambda: self._write_lexical(doc, owner_id, app_id))
        if configured["graph"]:
            self._attempt_pillar(doc.id, "graph",
                                 lambda: self._write_graph(doc, graph_extraction))
        if configured["vector"]:
            self._attempt_pillar(doc.id, "vector",
                                 lambda: self._write_vector(doc, title, content))

        self.documents.update_status(doc.id, IngestionStatus.COMPLETED)
        logger.info("storage_manager.ingestion_complete: doc=%s title=%s owner=%s app=%s",
                     doc.id, title[:50], owner_id, app_id)
        return doc


    def _attempt_pillar(self, doc_id, pillar: str, write) -> bool:
        """Run a pillar writer and record ok/failed. Never raises (ingestion of a
        doc must not fail because one optional pillar is down)."""
        try:
            write()
            self.index_state.mark(doc_id, pillar, "ok")
            return True
        except Exception as e:
            logger.warning("storage_manager.%s_failed: doc=%s err=%s", pillar, doc_id, e)
            self.index_state.mark(doc_id, pillar, "failed", error=str(e))
            return False

    def _write_lexical(self, doc, owner_id, app_id) -> None:
        """Index the doc into Elasticsearch (idempotent: _id = doc_id upsert)."""
        search_doc = SearchDocument(
            document_id=doc.id,
            title=doc.title,
            content=doc.content,
            document_type=getattr(doc.document_type, "value", str(doc.document_type)),
            source_path=doc.source_path,
            author=doc.metadata.get("author"),
            tags=doc.metadata.get("tags", []),
            created_at=doc.created_at.isoformat(),
            owner_id=owner_id,
            app_id=app_id,
        )
        self.lexical.index_document(search_doc)

    def _write_graph(self, doc, graph_extraction=None) -> None:
        """Store graph entities/relations for the doc (idempotent: MERGE by id).

        Uses a pre-computed extraction when supplied (inline ingest); otherwise
        re-extracts via self.extractor (reconciler path). A doc that yields zero
        entities is still a SUCCESS — there is simply nothing to store."""
        if graph_extraction is None and self.extractor is not None:
            from brain.models import DocumentType
            dt = doc.document_type
            if not isinstance(dt, DocumentType):
                try:
                    dt = DocumentType(dt)
                except Exception:
                    dt = DocumentType.DOCUMENT
            extracted = self.extractor.extract(
                document_id=str(doc.id),
                content=doc.content,
                document_type=dt,
                language=doc.metadata.get("language", ""),
            )
            import inspect, asyncio as _asyncio
            graph_extraction = _asyncio.run(extracted) if inspect.iscoroutine(extracted) else extracted
        if graph_extraction is None:
            raise RuntimeError("graph reindex needs a precomputed extraction or an extractor")
        real_doc_id = str(doc.id)
        graph_extraction.document_id = real_doc_id
        for _ent in getattr(graph_extraction, "entities", []) or []:
            _ent.source_document_id = real_doc_id
        for _rel in getattr(graph_extraction, "relations", []) or []:
            _rel.source_document_id = real_doc_id
        self.graph.store_extraction(graph_extraction)

    def _write_vector(self, doc, title: str = None, content: str = None) -> None:
        """Embed the doc (doc-level + parent-child chunks). Idempotent: overwrites
        the stored vector. Raises if the embedding endpoint errors (→ retried)."""
        title = doc.title if title is None else title
        content = doc.content if content is None else content
        embed_text = f"{title}\n\n{content}" if title else content
        vector = self.embeddings.embed_one(embed_text[:8000])
        if not vector:
            raise RuntimeError("embedding endpoint returned no vector")
        self.documents.update_embedding(doc.id, vector, model=self.embeddings.model or "")
        if self._chunking_enabled():
            self._embed_chunks(doc.id, title, content)

    def reindex(self, doc_id: str, pillar: str) -> tuple:
        """Re-run ONE pillar for a document (the reconciler's unit of work).
        Returns (ok, error). Idempotent — safe to call repeatedly."""
        configured = self._configured_pillars()
        if not configured.get(pillar):
            self.index_state.mark(doc_id, pillar, "skipped")
            return True, ""
        try:
            doc = self.documents.get_document(doc_id)
        except Exception as e:
            self.index_state.mark(doc_id, pillar, "failed", error=f"load: {e}")
            return False, str(e)
        if doc is None:
            self.index_state.mark(doc_id, pillar, "skipped", error="document missing")
            return True, ""
        try:
            if pillar == "lexical":
                meta = self._db.fetch_one(
                    "SELECT owner_id, app_id FROM ag_brain.documents WHERE doc_id=%s",
                    (int(doc_id),)) or {}
                self._write_lexical(doc, meta.get("owner_id") or "__tenant__",
                                    meta.get("app_id") or 1)
            elif pillar == "vector":
                self._write_vector(doc)
            elif pillar == "graph":
                self._write_graph(doc)
            else:
                self.index_state.mark(doc_id, pillar, "skipped")
                return True, ""
            self.index_state.mark(doc_id, pillar, "ok")
            return True, ""
        except Exception as e:
            logger.warning("storage_manager.reindex_failed: doc=%s pillar=%s err=%s", doc_id, pillar, e)
            self.index_state.mark(doc_id, pillar, "failed", error=str(e))
            return False, str(e)

    def index_health(self) -> dict:
        """One-call answer to 'is ingestion working': configured pillars + the
        per-pillar/per-state document counts."""
        return {
            "configured": self._configured_pillars(),
            "pillars": self.index_state.health(),
        }


    @staticmethod
    def _pct(n: int, total: int) -> float:
        return round(100.0 * n / total, 1) if total else 100.0

    def health_probe(self) -> dict:
        """A REAL health check: actually hit every store and report honest status.

        Not "is it configured" — it pings Elasticsearch, runs a Cypher against
        Neo4j (so a bad/empty credential shows as unhealthy, not silently ok),
        makes a live embedding call, and compares each pillar's indexed-doc count
        against the document total. Returns a structured report whose top-level
        `status` is healthy | degraded | unhealthy.
        """
        cfg = self._configured_pillars()
        state = self.index_state.health()
        try:
            total = int(self.documents.count_documents().get("total_documents", 0))
        except Exception:
            total = 0

        comps = {
            "document_store": self._probe_documents(total),
            "lexical": self._probe_lexical(cfg["lexical"], state.get("lexical", {}), total),
            "graph": self._probe_graph(cfg["graph"], state.get("graph", {})),
            "vector": self._probe_vector(cfg["vector"], state.get("vector", {}), total),
        }

        overall = "healthy"
        for c in comps.values():
            if c["status"] == "unhealthy":
                overall = "unhealthy"
                break
            if c["status"] == "degraded":
                overall = "degraded"
        return {"status": overall, "documents": total, "components": comps}

    def _probe_documents(self, total: int) -> dict:
        try:
            self._db.fetch_one("SELECT 1", ())
            return {"status": "healthy", "reachable": True, "documents": total}
        except Exception as e:
            return {"status": "unhealthy", "reachable": False, "error": str(e)[:200]}

    def _probe_lexical(self, configured: bool, st: dict, total: int) -> dict:
        if not configured:
            return {"status": "not_configured", "reachable": False}
        try:
            reachable = bool(self.lexical._client.ping())
            es_docs = int(self.lexical._client.count(index=self.lexical.index_name).get("count", 0)) if reachable else 0
        except Exception as e:
            return {"status": "unhealthy", "reachable": False, "error": str(e)[:200]}
        if not reachable:
            return {"status": "unhealthy", "reachable": False, "error": "ping failed"}
        pending = st.get("pending", 0) + st.get("failed", 0)
        return {
            "status": "healthy" if pending == 0 else "degraded",
            "reachable": True, "index": self.lexical.index_name, "es_docs": es_docs,
            "indexed_ok": st.get("ok", 0), "pending": pending,
            "coverage_pct": self._pct(st.get("ok", 0), total),
        }

    def _probe_graph(self, configured: bool, st: dict) -> dict:
        if not configured:
            return {"status": "not_configured", "reachable": False}
        try:
            if getattr(self.graph, "_driver", None) is None:
                self.graph.connect()
            with self.graph._session() as s:
                s.run("RETURN 1").consume()
        except Exception as e:
            return {"status": "unhealthy", "reachable": False, "error": str(e)[:200]}
        pending = st.get("pending", 0) + st.get("failed", 0)
        return {
            "status": "healthy" if pending == 0 else "degraded",
            "reachable": True, "indexed_ok": st.get("ok", 0), "pending": pending,
        }

    def _probe_vector(self, configured: bool, st: dict, total: int) -> dict:
        if not configured:
            return {"status": "not_configured", "reachable": False}
        try:
            vec = self.embeddings.embed_one("healthcheck")
        except Exception as e:
            return {"status": "unhealthy", "reachable": False, "error": str(e)[:200]}
        if not vec:
            return {"status": "unhealthy", "reachable": False, "error": "no vector returned"}
        pending = st.get("pending", 0) + st.get("failed", 0)
        return {
            "status": "healthy" if pending == 0 else "degraded",
            "reachable": True, "model": self.embeddings.model, "dim": len(vec),
            "embedded_ok": st.get("ok", 0), "pending": pending,
            "coverage_pct": self._pct(st.get("ok", 0), total),
        }


    @staticmethod
    def _chunking_enabled() -> bool:
        try:
            from core.config import app_config
            return app_config.get_bool("brain.chunking_enabled", True)
        except Exception:
            return True

    def _embed_chunks(self, doc_id, title: str, content: str) -> int:
        """Chunk a document, embed each chunk, and store them (parent-child).

        Batches the embedding calls. Returns the number of chunks stored.
        RAISES if any chunk could not be embedded, so the caller marks the vector
        pillar not-'ok' and the reconciler re-attempts (idempotent) until every
        chunk is embedded. Partially-good vectors are still persisted (kept, not
        discarded) so the retry only has to fill the gaps; the doc stays
        searchable via FTS + the doc-level vector meanwhile.
        """
        from brain.ingestion.chunker import chunk_text
        from core.config import app_config
        try:
            size = app_config.get_int("brain.chunk_size", 2000)
            overlap = app_config.get_int("brain.chunk_overlap", 320)
        except Exception:
            size, overlap = 2000, 320

        body = f"{title}\n\n{content}" if title else content
        chunks = chunk_text(body, chunk_size=size, overlap=overlap)
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        vectors: list = []
        batch = 32
        for i in range(0, len(texts), batch):
            want = len(texts[i:i + batch])
            got = self.embeddings.embed(texts[i:i + batch]) or []
            # Keep whatever came back; pad missing with None instead of throwing
            # away the good vectors when a batch partially (or fully) fails.
            if len(got) != want:
                got = (list(got) + [None] * want)[:want]
            vectors.extend(got)
        if len(vectors) != len(texts):  # safety net
            vectors = (list(vectors) + [None] * len(texts))[:len(texts)]

        payload = [{"content": c.content, "embedding": v}
                   for c, v in zip(chunks, vectors)]
        n = self.documents.replace_chunks(
            doc_id, payload, model=self.embeddings.model or "")
        missing = sum(1 for v in vectors if not v)
        logger.info("storage_manager.chunked: doc=%s chunks=%d unembedded=%d",
                    doc_id, n, missing)
        if missing:
            raise RuntimeError(
                f"{missing}/{len(vectors)} chunks have no embedding "
                f"(embedding endpoint returned fewer vectors than requested)")
        return n


    def get_system_stats(self) -> dict:
        """Get comprehensive statistics from all storage pillars."""
        stats = {
            "document_store": self.documents.count_documents(),
            "context_cache": self.context.get_stats(),
        }

        if self.graph:
            try:
                stats["graph_store"] = self.graph.get_stats()
            except Exception:
                stats["graph_store"] = {"status": "error"}
        else:
            stats["graph_store"] = {"status": "not_configured"}

        if self.lexical:
            try:
                stats["lexical_store"] = self.lexical.get_stats()
            except Exception:
                stats["lexical_store"] = {"status": "error"}
        else:
            stats["lexical_store"] = {"status": "not_configured"}

        stats["content_cache"] = self.content_cache.get_stats()
        stats["entity_index"] = self.entity_index.get_stats()

        return stats
