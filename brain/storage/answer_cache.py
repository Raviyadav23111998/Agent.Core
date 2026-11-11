from __future__ import annotations

"""
Agent Brain — Semantic Answer Cache (PostgreSQL)
============================================================
Caches full LLM answers keyed by a normalized question hash.
When a similar question is asked again, the cached answer is
returned instantly at $0.00 cost and ~5ms latency.

Storage: PostgreSQL ag_brain.answer_cache (via TenantDB)

Normalization strategy:
  1. Lowercase the question
  2. Remove punctuation and stop words
  3. Sort remaining keywords alphabetically
  4. SHA-256 hash the sorted keyword string
"""

import hashlib
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "but", "and",
    "or", "if", "while", "about", "up", "down", "what", "which", "who",
    "this", "that", "these", "those", "it", "its", "i", "me", "my", "we",
    "our", "you", "your", "he", "she", "they", "them", "their", "please",
})

DEFAULT_TTL_SECONDS = 7200
FUZZY_MATCH_THRESHOLD = 0.70


class AnswerCache:
    """PostgreSQL-backed semantic answer cache.

    Flow:
      1. Normalize question → generate hash key
      2. Check PG for exact hash match
      3. If miss, check for fuzzy keyword-overlap match
      4. If hit: return cached answer (cost: $0.00, latency: ~5ms)
      5. If miss: return None (caller should invoke LLM)
    """

    def __init__(self, db=None):
        """
        Args:
            db: TenantDB instance for PostgreSQL access.
        """
        self._db = db
        if db is None:
            logger.warning("AnswerCache initialized without TenantDB — caching disabled")


    def lookup(self, question: str, bundle_name: str = "",
               visible_scopes: list = None) -> Optional[str]:
        """Check if we have a cached answer (exact hash, then fuzzy overlap)."""
        if not self._db:
            return None

        keywords = self._normalize(question)
        scope_key = self._scope_key(visible_scopes)
        question_hash = self._hash(keywords + scope_key)

        row = self._db.fetch_one(
            """SELECT * FROM ag_brain.answer_cache
               WHERE question_hash = %s
                 AND created_at + (ttl_seconds * INTERVAL '1 second') > NOW()""",
            (question_hash,),
        )

        if row:
            self._record_hit(question_hash)
            logger.info("answer_cache.exact_hit: question=%s hit_count=%s tokens_saved=%s",
                        question[:60], row.get("hit_count", 0) + 1, row.get("input_tokens_saved", 0))
            return row["answer"]

        if bundle_name:
            candidates = self._db.fetch_all(
                """SELECT * FROM ag_brain.answer_cache
                   WHERE bundle_name = %s
                     AND created_at + (ttl_seconds * INTERVAL '1 second') > NOW()
                   ORDER BY created_at DESC LIMIT 50""",
                (bundle_name,),
            )
        else:
            candidates = self._db.fetch_all(
                """SELECT * FROM ag_brain.answer_cache
                   WHERE created_at + (ttl_seconds * INTERVAL '1 second') > NOW()
                   ORDER BY created_at DESC LIMIT 50""",
            )

        for candidate in candidates:
            candidate_keywords = set(candidate["question_keywords"].split(","))
            overlap = self._keyword_overlap(set(keywords.split(",")), candidate_keywords)
            if overlap >= FUZZY_MATCH_THRESHOLD:
                self._record_hit(candidate["question_hash"])
                logger.info("answer_cache.fuzzy_hit: question=%s overlap=%s",
                            question[:60], round(overlap, 2))
                return candidate["answer"]

        return None

    def store(
        self,
        question: str,
        answer: str,
        bundle_name: str = "",
        model: str = "",
        input_tokens_saved: int = 0,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        visible_scopes: list = None,
    ):
        """Cache an LLM answer for future reuse."""
        if not self._db:
            return

        keywords = self._normalize(question)
        scope_key = self._scope_key(visible_scopes)
        question_hash = self._hash(keywords + scope_key)

        self._db.execute("""
            INSERT INTO ag_brain.answer_cache
                (question_hash, question_text, question_keywords, answer,
                 bundle_name, model, input_tokens_saved, ttl_seconds, hit_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0)
            ON CONFLICT (question_hash) DO UPDATE SET
                answer = EXCLUDED.answer,
                question_text = EXCLUDED.question_text,
                question_keywords = EXCLUDED.question_keywords,
                model = EXCLUDED.model,
                input_tokens_saved = EXCLUDED.input_tokens_saved,
                ttl_seconds = EXCLUDED.ttl_seconds,
                hit_count = 0,
                created_at = NOW()
        """, (question_hash, question, keywords, answer,
              bundle_name, model, input_tokens_saved, ttl_seconds))

    def invalidate_bundle(self, bundle_name: str) -> int:
        """Invalidate all cached answers for a specific bundle."""
        if not self._db:
            return 0
        count = self._db.execute(
            "DELETE FROM ag_brain.answer_cache WHERE bundle_name = %s",
            (bundle_name,),
        )
        if count:
            logger.info("answer_cache.bundle_invalidated: bundle=%s deleted=%s", bundle_name, count)
        return count

    def clear(self):
        """Clear the entire answer cache."""
        if not self._db:
            return
        self._db.execute("DELETE FROM ag_brain.answer_cache")
        logger.warning("answer_cache.cleared")

    def get_stats(self) -> dict:
        """Return cache statistics."""
        if not self._db:
            return {"total_entries": 0, "total_cache_hits": 0, "total_tokens_saved": 0,
                    "estimated_cost_saved_usd": 0, "top_hits": []}

        total = self._db.fetch_scalar(
            "SELECT COUNT(*) FROM ag_brain.answer_cache"
        ) or 0
        total_hits = self._db.fetch_scalar(
            "SELECT COALESCE(SUM(hit_count), 0) FROM ag_brain.answer_cache"
        ) or 0
        total_tokens_saved = self._db.fetch_scalar(
            "SELECT COALESCE(SUM(hit_count * input_tokens_saved), 0) FROM ag_brain.answer_cache"
        ) or 0

        top_hits = self._db.fetch_all(
            """SELECT question_text, hit_count, input_tokens_saved
               FROM ag_brain.answer_cache ORDER BY hit_count DESC LIMIT 5"""
        )

        return {
            "total_entries": total,
            "total_cache_hits": total_hits,
            "total_tokens_saved": total_tokens_saved,
            "estimated_cost_saved_usd": round(total_tokens_saved * 0.15 / 1_000_000, 5),
            "top_hits": top_hits,
        }


    def _normalize(self, question: str) -> str:
        """Normalize a question into sorted, deduplicated keywords."""
        text = re.sub(r'[^\w\s]', '', question.lower())
        words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
        unique_sorted = sorted(set(words))
        return ",".join(unique_sorted)

    def _hash(self, keywords: str) -> str:
        """SHA-256 hash of normalized keywords."""
        return hashlib.sha256(keywords.encode()).hexdigest()[:16]

    @staticmethod
    def _scope_key(visible_scopes: list = None) -> str:
        """Generate a stable key fragment from visible_scopes for cache isolation."""
        if not visible_scopes:
            return "|scope:__all__"
        return "|scope:" + ",".join(sorted(visible_scopes))

    def _keyword_overlap(self, set_a: set, set_b: set) -> float:
        """Calculate Jaccard similarity between two keyword sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    def _record_hit(self, question_hash: str):
        """Increment the hit counter for a cache entry."""
        try:
            self._db.execute(
                """UPDATE ag_brain.answer_cache
                   SET hit_count = hit_count + 1, last_hit_at = NOW()
                   WHERE question_hash = %s""",
                (question_hash,),
            )
        except Exception:
            pass

    def close(self):
        """No-op — connection lifecycle managed by TenantDB pool."""
        pass
