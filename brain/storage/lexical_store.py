"""
Agent Brain — Lexical Search Store (Elasticsearch)
====================================================
PILLAR 2: The Keyword Engine

Elasticsearch provides BM25-based full-text search — the same algorithm
that powers Google's early search. No vectors, no embeddings, just
classic inverted-index keyword matching.

Why Lexical Search alongside a Knowledge Graph?
- Graphs are great for relational queries ("who emailed whom")
- But terrible at: "find all documents containing error code ERR-4092"
- BM25 excels at exact keyword matching, error codes, IDs, specific names
- It's instant (milliseconds), while LLM context reads take seconds

The Parser writes TO Elasticsearch during ingestion.
The Search Agent reads FROM Elasticsearch during retrieval.
"""

from typing import Optional
import logging
import threading
try:
    from elasticsearch import Elasticsearch, NotFoundError as ESNotFoundError
except ImportError:
    Elasticsearch = None
    ESNotFoundError = Exception

from brain.models import SearchDocument, SearchResult
from brain.exceptions import SearchConnectionError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Process-wide shared Elasticsearch client(s)
# ─────────────────────────────────────────────────────────────────────────────
# The elasticsearch-py client is thread-safe and is *designed* to be created
# ONCE and shared for the lifetime of the process (it owns an internal urllib3
# connection pool). In this system tenants are isolated by INDEX (brain_<sfx>),
# NOT by client — so every tenant talking to the same ES cluster must share one
# client. Building one client per tenant and closing it on cache eviction was
# the root cause of `ClosedPoolError: Pool is closed`: an evicted tenant would
# close the pool out from under another request still using it.
#
# We therefore keep a small registry of shared clients keyed by cluster URL.
# Per-tenant LexicalStore instances borrow from this registry and NEVER close
# the shared transport on disconnect(); the client lives until process exit.
_SHARED_CLIENTS: "dict[str, Elasticsearch]" = {}
_SHARED_LOCK = threading.Lock()


def _shared_client(url: str) -> "Elasticsearch":
    """Return the process-wide ES client for `url`, creating it on first use.

    Configured with bounded retries/timeouts so a transient blip self-heals
    instead of surfacing as a hard error."""
    if Elasticsearch is None:
        raise SearchConnectionError("elasticsearch client library not installed")
    cli = _SHARED_CLIENTS.get(url)
    if cli is not None:
        return cli
    with _SHARED_LOCK:
        cli = _SHARED_CLIENTS.get(url)
        if cli is None:
            cli = Elasticsearch(
                url,
                max_retries=3,
                retry_on_timeout=True,
                request_timeout=30,
            )
            _SHARED_CLIENTS[url] = cli
            logger.info("lexical_store.shared_client_created: url=%s", url)
        return cli


def close_shared_clients() -> None:
    """Close every shared ES client. Call ONLY at process shutdown — never on
    per-tenant eviction (that is what broke the pool before)."""
    with _SHARED_LOCK:
        for url, cli in list(_SHARED_CLIENTS.items()):
            try:
                cli.close()
            except Exception:
                pass
        _SHARED_CLIENTS.clear()


INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "brain_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "stop",
                        "snowball",
                    ],
                },
                "code_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase"],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "document_id": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "brain_analyzer",
                "fields": {"raw": {"type": "keyword"}},
            },
            "content": {
                "type": "text",
                "analyzer": "brain_analyzer",
                "fields": {
                    "code": {
                        "type": "text",
                        "analyzer": "code_analyzer",
                    },
                },
            },
            "document_type": {"type": "keyword"},
            "source_path": {"type": "keyword"},
            "author": {"type": "keyword"},
            "tags": {"type": "keyword"},
            "created_at": {"type": "date"},
            "owner_id": {"type": "keyword"},
            "app_id": {"type": "integer"},
        },
    },
}


class LexicalStore:
    """
    Elasticsearch-backed full-text search.
    Provides BM25 keyword search across all ingested documents.
    """

    def __init__(self, url: str = "http://localhost:9200", index_name: str = "company_brain"):
        self.url = url
        self.index_name = index_name
        self._client: Optional[Elasticsearch] = None


    def connect(self):
        """Borrow the process-wide ES client and ensure this tenant's index.

        Idempotent and self-healing: it (re)acquires the shared client every
        call, so a store whose local reference was dropped transparently
        reconnects on next use. Raising is reserved for a genuinely unreachable
        cluster."""
        try:
            self._client = _shared_client(self.url)
            info = self._client.info()
            logger.info("lexical_store.connected: url=%s cluster=%s version=%s", self.url, info.get("cluster_name", "unknown"), info.get("version", {}).get("number", "unknown"))
            self._ensure_index()
        except Exception as e:
            self._client = None
            raise SearchConnectionError(f"Failed to connect to Elasticsearch at {self.url}: {e}")

    def close(self):
        """Detach this per-tenant store from the shared client.

        Deliberately does NOT close the transport — the client is shared across
        every tenant on this cluster and lives for the process lifetime. Dropping
        the local reference is enough; the `client` property re-borrows on next
        use. (Closing the shared pool here was the original `Pool is closed` bug.)
        """
        self._client = None
        logger.info("lexical_store.detached")

    @property
    def client(self) -> Elasticsearch:
        """Always return a live shared client, re-borrowing if we were detached."""
        if self._client is None:
            self._client = _shared_client(self.url)
        return self._client

    def _ensure_index(self):
        """Create the index with our custom mapping if it doesn't exist."""
        if not self.client.indices.exists(index=self.index_name):
            try:
                self.client.indices.create(index=self.index_name, body=INDEX_MAPPING)
                logger.info("lexical_store.index_created: index=%s", self.index_name)
            except Exception as e:
                if "resource_already_exists_exception" in str(e):
                    logger.info("lexical_store.index_already_created_by_another_process: index=%s", self.index_name)
                else:
                    raise
        else:
            logger.info("lexical_store.index_exists: index=%s", self.index_name)


    def index_document(self, doc: SearchDocument):
        """
        Index a document for full-text search.
        Uses the document_id as the Elasticsearch _id for idempotent upserts.
        """
        body = {
            "document_id": doc.document_id,
            "title": doc.title,
            "content": doc.content,
            "document_type": doc.document_type,
            "source_path": doc.source_path,
            "author": doc.author,
            "tags": doc.tags,
            "created_at": doc.created_at or None,
            "owner_id": getattr(doc, 'owner_id', '__tenant__'),
            "app_id": getattr(doc, 'app_id', None),
        }

        self.client.index(
            index=self.index_name,
            id=doc.document_id,
            document=body,
        )
        logger.info("lexical_store.indexed: document_id=%s title=%s", doc.document_id, doc.title[:50])

    def bulk_index(self, documents: list[SearchDocument]):
        """Bulk-index multiple documents efficiently."""
        if not documents:
            return

        operations = []
        for doc in documents:
            operations.append({"index": {"_index": self.index_name, "_id": doc.document_id}})
            operations.append({
                "document_id": doc.document_id,
                "title": doc.title,
                "content": doc.content,
                "document_type": doc.document_type,
                "source_path": doc.source_path,
                "author": doc.author,
                "tags": doc.tags,
                "created_at": doc.created_at or None,
                "owner_id": getattr(doc, 'owner_id', '__tenant__'),
                "app_id": getattr(doc, 'app_id', None),
            })

        self.client.bulk(operations=operations, refresh=True)
        logger.info("lexical_store.bulk_indexed: count=%s", len(documents))


    def search(
        self,
        query: str,
        document_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 10,
        visible_scopes: Optional[list[str]] = None,
        app_id: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Full-text BM25 search across all indexed documents.

        Args:
            query: The search terms (natural language or keywords)
            document_type: Optional filter by type (email, code, policy, etc.)
            tags: Optional filter by tags
            limit: Maximum results to return

        Returns:
            List of SearchResult with highlighted content snippets
        """
        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "title^3",
                        "content",
                        "content.code",
                        "tags^2",
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                },
            },
        ]

        filter_clauses = []
        if visible_scopes:
            filter_clauses.append({"terms": {"owner_id": visible_scopes}})
        if app_id is not None:
            filter_clauses.append({"term": {"app_id": app_id}})
        if document_type:
            filter_clauses.append({"term": {"document_type": document_type}})
        if tags:
            filter_clauses.append({"terms": {"tags": tags}})

        search_body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses,
                },
            },
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 200,
                        "number_of_fragments": 3,
                    },
                    "title": {},
                },
                "pre_tags": ["**"],
                "post_tags": ["**"],
            },
            "size": limit,
            "_source": ["document_id", "title", "document_type", "source_path"],
        }

        response = self.client.search(index=self.index_name, body=search_body)

        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            highlights = hit.get("highlight", {})

            content_snippets = highlights.get("content", [])
            snippet = " ... ".join(content_snippets) if content_snippets else ""

            results.append(SearchResult(
                document_id=source["document_id"],
                title=source.get("title", ""),
                content_snippet=snippet,
                score=hit["_score"],
                document_type=source.get("document_type", ""),
                source_path=source.get("source_path", ""),
            ))

        logger.info("lexical_store.searched: query=%s hits=%s", query[:50], len(results))
        return results

    def search_exact(self, field: str, value: str, limit: int = 10) -> list[SearchResult]:
        """
        Exact-match search on a specific keyword field.
        Useful for finding documents by exact author name, error code, file path, etc.
        """
        search_body = {
            "query": {"term": {field: value}},
            "size": limit,
        }

        response = self.client.search(index=self.index_name, body=search_body)
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(SearchResult(
                document_id=source["document_id"],
                title=source.get("title", ""),
                content_snippet="",
                score=hit["_score"],
                document_type=source.get("document_type", ""),
                source_path=source.get("source_path", ""),
            ))

        return results


    def delete_document(self, document_id: str) -> bool:
        """Remove a document from the search index."""
        try:
            self.client.delete(index=self.index_name, id=document_id)
            logger.info("lexical_store.deleted: document_id=%s", document_id)
            return True
        except ESNotFoundError:
            return False

    def get_stats(self) -> dict:
        """Get index statistics."""
        try:
            stats = self.client.indices.stats(index=self.index_name)
            index_stats = stats["indices"].get(self.index_name, {}).get("primaries", {})
            return {
                "total_documents": index_stats.get("docs", {}).get("count", 0),
                "index_size_bytes": index_stats.get("store", {}).get("size_in_bytes", 0),
                "index_size_mb": round(
                    index_stats.get("store", {}).get("size_in_bytes", 0) / (1024 * 1024), 2
                ),
            }
        except Exception:
            return {"total_documents": 0, "index_size_bytes": 0, "index_size_mb": 0}

    def refresh_index(self):
        """Force-refresh the index to make recent writes searchable immediately."""
        self.client.indices.refresh(index=self.index_name)

    def clear_all(self):
        """⚠️ Delete and recreate the index."""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        self._ensure_index()
        logger.warning("lexical_store.cleared_all")
