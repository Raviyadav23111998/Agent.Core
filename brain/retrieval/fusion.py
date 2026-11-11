from __future__ import annotations

"""
Agent Brain — Hybrid Fusion & Reranking
═══════════════════════════════════════════
Combines results from multiple retrievers (BM25 lexical, vector/semantic,
graph) into one ranked list, then optionally reranks the top candidates.

Why this matters (Phase-1 accuracy leap):
- BM25 nails exact terms/IDs but misses paraphrase/synonyms.
- Vector nails semantics but misses rare exact tokens.
- Reciprocal Rank Fusion (RRF) merges their rankings WITHOUT needing
  comparable scores — robust, parameter-light, the production default.

Everything here is pure logic (no infra), so it is fully unit-testable and
works whether or not the vector store is configured.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("agent.brain.fusion")


@dataclass
class ScoredDoc:
    """A retrieved item with an id, text, and provenance."""
    doc_id: str
    text: str = ""
    source: dict = field(default_factory=dict)
    score: float = 0.0


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """Fuse multiple ranked id-lists via Reciprocal Rank Fusion.

    RRF score for a doc = Σ_lists  w_l / (k + rank_in_list).
    Lower rank (earlier) → higher contribution. k dampens the tail.

    Args:
        ranked_lists: each is an ordered sequence of doc ids (best first).
        k: RRF constant (60 is the canonical default).
        weights: optional per-list weight (e.g. trust vector > lexical).

    Returns:
        [(doc_id, fused_score), …] sorted by score desc.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: Dict[str, float] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def fuse_docs(
    doc_lists: Sequence[Sequence[ScoredDoc]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
    top_n: Optional[int] = None,
) -> List[ScoredDoc]:
    """RRF-fuse lists of ScoredDocs, returning unique docs in fused order."""
    id_lists = [[d.doc_id for d in lst] for lst in doc_lists]
    fused = reciprocal_rank_fusion(id_lists, k=k, weights=weights)
    by_id: Dict[str, ScoredDoc] = {}
    for lst in doc_lists:
        for d in lst:
            by_id.setdefault(d.doc_id, d)
    out: List[ScoredDoc] = []
    for doc_id, score in fused:
        d = by_id.get(doc_id)
        if d is None:
            continue
        out.append(ScoredDoc(doc_id=d.doc_id, text=d.text, source=d.source, score=score))
    return out[:top_n] if top_n else out


class Reranker:
    """Interface: reorder candidates by relevance to the query."""

    def rerank(self, query: str, docs: List[ScoredDoc], top_n: int = 8) -> List[ScoredDoc]:
        raise NotImplementedError


class LexicalReranker(Reranker):
    """Zero-infra reranker: scores by query-term overlap (Jaccard-ish).

    A sane default when no embedding model is configured — still meaningfully
    better than raw fusion order for keyword-y queries.
    """

    @staticmethod
    def _tokens(text: str) -> set:
        return {t for t in ''.join(c.lower() if c.isalnum() else ' ' for c in text).split() if len(t) > 2}

    def rerank(self, query: str, docs: List[ScoredDoc], top_n: int = 8) -> List[ScoredDoc]:
        q = self._tokens(query)
        if not q:
            return docs[:top_n]
        scored = []
        for d in docs:
            dt = self._tokens(d.text)
            overlap = len(q & dt) / (len(q) or 1)
            scored.append((overlap, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ScoredDoc(d.doc_id, d.text, d.source, score) for score, d in scored[:top_n]]


class EmbeddingReranker(Reranker):
    """Semantic reranker: cosine(query_emb, doc_emb).

    Uses an EmbeddingClient; degrades to LexicalReranker when embeddings are
    unavailable, so callers can always rely on a result.
    """

    def __init__(self, embedding_client):
        self._emb = embedding_client
        self._fallback = LexicalReranker()

    def rerank(self, query: str, docs: List[ScoredDoc], top_n: int = 8) -> List[ScoredDoc]:
        if not docs:
            return []
        if not self._emb or not getattr(self._emb, "available", False):
            return self._fallback.rerank(query, docs, top_n)
        from core.embeddings import cosine_similarity
        try:
            vectors = self._emb.embed([query] + [d.text for d in docs])
        except Exception:
            vectors = []
        if not vectors or len(vectors) != len(docs) + 1:
            return self._fallback.rerank(query, docs, top_n)
        q_vec, doc_vecs = vectors[0], vectors[1:]
        scored = [
            (cosine_similarity(q_vec, dv), d)
            for dv, d in zip(doc_vecs, docs)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ScoredDoc(d.doc_id, d.text, d.source, score) for score, d in scored[:top_n]]
