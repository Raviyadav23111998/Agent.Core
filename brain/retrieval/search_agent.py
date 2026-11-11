from __future__ import annotations

"""
Agent Brain — Search Agent
==============================
Retrieval agent for the Lexical Search Index (Elasticsearch).

This is the CHEAPEST and FASTEST agent — no LLM calls needed.
Pure BM25 keyword matching with highlighting.

Use for:
  "Find all documents mentioning error code ERR-4092"
  "Search for JWT token refresh"
  "Find emails from mike@company.com"
"""

import logging

from brain.storage.lexical_store import LexicalStore
from core.tenant import TENANT_SCOPE
from brain.retrieval.base import BaseAgent, AgentResult
from brain import NLP_STOP_WORDS

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
    """
    Retrieval agent that queries the Elasticsearch lexical index.
    Falls back to PostgreSQL FTS when Elasticsearch is unavailable.
    No LLM required — direct BM25/FTS5 keyword search.
    """

    AGENT_NAME = "search_agent"

    RERANK_POOL_FACTOR = 3

    def __init__(self, lexical_store: LexicalStore, document_store=None,
                 reranker=None, embeddings=None):
        self.lexical = lexical_store
        self._document_store = document_store
        if embeddings is None:
            try:
                from core.embeddings import EmbeddingClient
                embeddings = EmbeddingClient()
            except Exception:
                embeddings = None
        self._embeddings = embeddings
        if reranker is None:
            try:
                from brain.retrieval.fusion import EmbeddingReranker
                reranker = EmbeddingReranker(embeddings)
            except Exception:
                reranker = None
        self._reranker = reranker

    def _hybrid_fetch(self, optimized: str, limit: int,
                      visible_scopes: list = None, app_id: int = None,
                      semantic_query: str = None) -> list:
        """Fetch documents via FTS, fused with vector search when available.

        Returns a list of Document objects. When embeddings are configured,
        BM25/FTS and dense-vector rankings are merged with Reciprocal Rank
        Fusion so exact-term and semantic matches both surface. When not,
        this is exactly the previous FTS-only behavior.

        `semantic_query` (C.5): a declarative full-sentence restatement used
        for the EMBEDDING arm only — keyword-stripped queries help BM25 but
        hurt dense retrieval, so each arm gets the query form it ranks best.
        """
        pool = max(limit * self.RERANK_POOL_FACTOR, limit)
        fts_docs = self._document_store.search_fts(
            optimized, limit=pool, visible_scopes=visible_scopes, app_id=app_id)

        if not (self._embeddings and getattr(self._embeddings, "available", False)):
            return fts_docs[:limit]
        try:
            qvec = self._embeddings.embed_one(semantic_query or optimized)
            vec_docs = self._vector_search(
                qvec, pool=pool, visible_scopes=visible_scopes, app_id=app_id)
        except Exception as e:
            logger.debug("search_agent.vector_arm_failed: %s", e)
            return fts_docs[:limit]
        if not vec_docs:
            return fts_docs[:limit]

        from brain.retrieval.fusion import reciprocal_rank_fusion
        by_id = {str(d.id): d for d in fts_docs}
        by_id.update({str(d.id): d for d in vec_docs})
        fused = reciprocal_rank_fusion(
            [[str(d.id) for d in fts_docs], [str(d.id) for d in vec_docs]])
        ordered = [by_id[doc_id] for doc_id, _ in fused if doc_id in by_id]
        return ordered[:limit]

    class _VectorHit:
        """Adapter giving a Document the same shape as an ES search hit, so
        vector-only candidates can flow through the shared formatting path."""

        __slots__ = ("document_id", "title", "score", "document_type",
                     "source_path", "content_snippet")

        def __init__(self, doc):
            self.document_id = str(doc.id)
            self.title = doc.title
            self.score = 0.0
            dt = getattr(doc, "document_type", "")
            self.document_type = dt.value if hasattr(dt, "value") else str(dt)
            self.source_path = getattr(doc, "source_path", "")
            content = getattr(doc, "content", "") or ""
            self.content_snippet = content[:300].replace("\n", " ").strip()

    def _fuse_es_with_vector(self, optimized: str, es_hits: list, pool: int,
                             visible_scopes: list = None,
                             app_id: int = None,
                             semantic_query: str = None) -> list:
        """RRF-fuse healthy-Elasticsearch BM25 hits with the vector arm.

        Previously the vector index was only consulted when ES was DOWN
        (the PG fallback path) — so with ES healthy, semantic-only matches
        could never enter the candidate pool and the reranker could only
        reorder BM25 hits. Degrades to the BM25 hits unchanged on any error
        or when embeddings are not configured.
        """
        if not (self._embeddings and getattr(self._embeddings, "available", False)
                and self._document_store):
            return es_hits
        try:
            qvec = self._embeddings.embed_one(semantic_query or optimized)
            vec_docs = self._vector_search(
                qvec, pool=pool, visible_scopes=visible_scopes, app_id=app_id)
            if not vec_docs:
                return es_hits

            from brain.retrieval.fusion import reciprocal_rank_fusion
            by_id = {str(h.document_id): h for h in es_hits}
            for d in vec_docs:
                by_id.setdefault(str(d.id), self._VectorHit(d))
            fused = reciprocal_rank_fusion([
                [str(h.document_id) for h in es_hits],
                [str(d.id) for d in vec_docs],
            ])
            return [by_id[doc_id] for doc_id, _ in fused if doc_id in by_id][:pool]
        except Exception as e:
            logger.debug("search_agent.es_vector_fuse_failed: %s", e)
            return es_hits

    def _vector_search(self, qvec: list, pool: int,
                       visible_scopes: list = None, app_id: int = None) -> list:
        """Dense vector arm — CHUNK-level (parent-child) preferred.

        Searches small embedded chunks for the precise passage and returns the
        parent Document (with the matched chunk as its snippet). Falls back to
        the coarse whole-doc embedding when no chunks are available (e.g. an
        un-backfilled corpus), so behaviour degrades, never breaks.
        """
        store = self._document_store
        chunk_search = getattr(store, "search_chunks_vector", None)
        if callable(chunk_search):
            try:
                docs = chunk_search(qvec, limit=pool,
                                    visible_scopes=visible_scopes, app_id=app_id)
                if docs:
                    return docs
            except Exception as e:
                logger.debug("search_agent.chunk_vector_failed: %s", e)
        return store.search_vector(
            qvec, limit=pool, visible_scopes=visible_scopes, app_id=app_id)

    def _rerank_hits(self, question: str, results: list, limit: int) -> list:
        """Rerank BM25 hits semantically; return the top `limit` in new order.

        BM25 finds exact-term matches; the reranker reorders by semantic
        relevance to the QUESTION (not just keyword overlap). Falls back to
        the original BM25 order on any error.
        """
        if not self._reranker or len(results) <= 1:
            return results[:limit]
        try:
            from brain.retrieval.fusion import ScoredDoc
            docs = [
                ScoredDoc(
                    doc_id=str(getattr(h, "document_id", i)),
                    text=f"{getattr(h, 'title', '')} {getattr(h, 'content_snippet', '') or ''}",
                )
                for i, h in enumerate(results)
            ]
            ranked = self._reranker.rerank(question, docs, top_n=limit)
            by_id = {str(getattr(h, "document_id", i)): h for i, h in enumerate(results)}
            reordered = [by_id[d.doc_id] for d in ranked if d.doc_id in by_id]
            return reordered or results[:limit]
        except Exception as e:
            logger.debug("search_agent.rerank_failed: %s", e)
            return results[:limit]

    def _optimize_query(self, question: str) -> str:
        """Strip stop words from natural language to improve BM25 relevance."""
        words = question.rstrip("?!.").split()
        keywords = [w for w in words if w.lower() not in NLP_STOP_WORDS and len(w) > 1]
        optimized = " ".join(keywords) if keywords else question
        logger.debug("search_agent.query_optimized: original=%s optimized=%s", question[:60], optimized[:60])
        return optimized

    async def query(
        self,
        question: str,
        **kwargs,
    ) -> AgentResult:
        """
        Search the lexical index with keywords.

        Args:
            question: Search query (keywords or natural language)
            **kwargs:
                document_type: Optional filter by type (email, code, policy, etc.)
                tags: Optional filter by tags
                limit: Max results (default 10)

        Returns:
            AgentResult with ranked document snippets
        """
        document_type = kwargs.get("document_type")
        tags = kwargs.get("tags")
        limit = kwargs.get("limit", 10)

        try:
            # `lexical` is None only when ES is not configured for this tenant.
            # A configured store is self-healing (it re-borrows the shared client
            # on use), so we no longer gate on the private `_client` handle — a
            # genuinely unreachable cluster surfaces as an exception below and
            # degrades to the Postgres FTS fallback.
            if not self.lexical:
                if self._document_store:
                    owner_id = kwargs.get("owner_id")
                    visible_scopes = kwargs.get("visible_scopes")
                    app_id = kwargs.get("app_id")
                    return await self._fts5_fallback(question, limit, owner_id=owner_id,
                                                     visible_scopes=visible_scopes,
                                                     app_id=app_id,
                                                     semantic_query=kwargs.get("semantic_query"))
                return AgentResult(
                    agent_name=self.AGENT_NAME,
                    content="Search index is currently unavailable.",
                    confidence=0.0,
                )

            optimized_query = self._optimize_query(question)

            visible_scopes = kwargs.get("visible_scopes")
            app_id = kwargs.get("app_id")
            fetch_limit = limit * self.RERANK_POOL_FACTOR if self._reranker else limit
            results = self.lexical.search(
                query=optimized_query,
                document_type=document_type,
                tags=tags,
                limit=fetch_limit,
                visible_scopes=visible_scopes,
                app_id=app_id,
            )

            pool = max(limit * self.RERANK_POOL_FACTOR, limit)
            results = self._fuse_es_with_vector(
                optimized_query, results, pool,
                visible_scopes=visible_scopes, app_id=app_id,
                semantic_query=kwargs.get("semantic_query"))
            if results:
                results = self._rerank_hits(question, results, limit)

            if not results:
                return AgentResult(
                    agent_name=self.AGENT_NAME,
                    content=f"No documents found matching: '{question}'",
                    sources=[],
                    confidence=0.1,
                    raw_data={"query": question, "hits": 0},
                )

            content_parts = [f"Found {len(results)} document(s) matching '{question}':\n"]

            sources = []
            for i, hit in enumerate(results, 1):
                content_parts.append(f"**{i}. {hit.title}** (score: {hit.score:.2f})")
                content_parts.append(f"   Type: {hit.document_type} | Source: {hit.source_path}")
                if hit.content_snippet:
                    content_parts.append(f"   Excerpt: ...{hit.content_snippet}...")
                content_parts.append("")

                sources.append({
                    "document_id": hit.document_id,
                    "title": hit.title,
                    "score": hit.score,
                    "type": hit.document_type,
                    "source_path": hit.source_path,
                    "snippet": (hit.content_snippet or "")[:300],
                })

            content = "\n".join(content_parts)

            top_score = results[0].score if results else 0
            confidence = min(0.3 + (top_score * 0.05) + (len(results) * 0.03), 0.95)

            return AgentResult(
                agent_name=self.AGENT_NAME,
                content=content,
                sources=sources,
                confidence=confidence,
                raw_data={
                    "query": question,
                    "hits": len(results),
                    "top_score": top_score,
                },
            )

        except Exception as e:
            # If Elasticsearch is genuinely unreachable, degrade to the Postgres
            # FTS fallback rather than returning an empty/zero-confidence error.
            if self._document_store:
                logger.warning("search_agent.es_unavailable_fallback_fts: question=%s error=%s",
                               question[:50], str(e))
                try:
                    return await self._fts5_fallback(
                        question, limit,
                        owner_id=kwargs.get("owner_id"),
                        visible_scopes=kwargs.get("visible_scopes"),
                        app_id=kwargs.get("app_id"),
                        semantic_query=kwargs.get("semantic_query"),
                    )
                except Exception as fe:
                    logger.error("search_agent.fts_fallback_failed: error=%s", str(fe))
            logger.error("search_agent.query_failed: question=%s error=%s", question[:50], str(e))
            return self._error_result(str(e))

    async def search_exact(
        self,
        field: str,
        value: str,
        limit: int = 10,
    ) -> AgentResult:
        """
        Exact-match search on a specific field.
        Useful for: error codes, email addresses, file paths.

        Args:
            field: Elasticsearch field name (e.g., "author", "document_type")
            value: Exact value to match
            limit: Max results

        Returns:
            AgentResult with exact matches
        """
        try:
            results = self.lexical.search_exact(field, value, limit=limit)

            if not results:
                return AgentResult(
                    agent_name=self.AGENT_NAME,
                    content=f"No exact matches found for {field}='{value}'",
                    confidence=0.1,
                )

            content_parts = [f"Found {len(results)} exact match(es) for {field}='{value}':\n"]
            sources = []

            for i, hit in enumerate(results, 1):
                content_parts.append(f"  {i}. {hit.title} ({hit.document_type})")
                sources.append({
                    "document_id": hit.document_id,
                    "title": hit.title,
                    "type": hit.document_type,
                })

            return AgentResult(
                agent_name=self.AGENT_NAME,
                content="\n".join(content_parts),
                sources=sources,
                confidence=0.9,
                raw_data={"field": field, "value": value, "hits": len(results)},
            )

        except Exception as e:
            return self._error_result(str(e))

    async def _fts5_fallback(self, question: str, limit: int = 10,
                              owner_id: str = None,
                              visible_scopes: list = None,
                              app_id: int = None,
                              semantic_query: str = None) -> AgentResult:
        """Search PostgreSQL when Elasticsearch is unavailable.

        Hybrid: BM25/FTS fused with dense-vector search via RRF when an
        embedding model is configured; pure FTS otherwise. (Method name kept
        for compatibility; "fts5" is historical.)
        """
        try:
            optimized = self._optimize_query(question)
            if visible_scopes:
                scopes = visible_scopes
            elif owner_id:
                scopes = [owner_id, TENANT_SCOPE]
            else:
                scopes = None
            docs = self._hybrid_fetch(optimized, limit=limit,
                                      visible_scopes=scopes, app_id=app_id,
                                      semantic_query=semantic_query)
            engine = "hybrid" if (
                self._embeddings and getattr(self._embeddings, "available", False)
            ) else "fts"

            if not docs:
                return AgentResult(
                    agent_name=self.AGENT_NAME,
                    content=f"No documents found matching: '{question}'",
                    sources=[],
                    confidence=0.1,
                    raw_data={"query": question, "hits": 0, "engine": engine},
                )

            content_parts = [f"Found {len(docs)} document(s) matching '{question}':\n"]
            sources = []
            for i, doc in enumerate(docs, 1):
                content_parts.append(f"**{i}. {doc.title}** ({doc.token_count} tokens)")
                snippet = doc.content[:300].replace('\n', ' ').strip()
                if len(doc.content) > 300:
                    snippet += "..."
                content_parts.append(f"   {snippet}")
                content_parts.append("")
                sources.append({
                    "document_id": doc.id,
                    "title": doc.title,
                    "type": doc.document_type.value if hasattr(doc.document_type, 'value') else str(doc.document_type),
                    "source_path": doc.source_path,
                    "snippet": snippet,
                })

            confidence = min(0.3 + len(docs) * 0.1, 0.85)

            return AgentResult(
                agent_name=self.AGENT_NAME,
                content="\n".join(content_parts),
                sources=sources,
                confidence=confidence,
                raw_data={"query": question, "hits": len(docs), "engine": engine},
            )
        except Exception as e:
            logger.error("search_agent.fts5_fallback_failed: %s", str(e))
            return self._error_result(f"FTS5 fallback failed: {e}")
