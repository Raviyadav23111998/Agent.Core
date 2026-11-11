"""
Agent — Memory Consolidator (PostgreSQL)
========================================
Lightweight, LLM-free memory maintenance engine.

Operations:
  - Temporal decay: reduce confidence of unaccessed memories, evict dead ones
  - Layer promotion: frequently-accessed session memories → semantic
  - Capacity management: evict lowest-importance memories when over limit
  - Expiry cleanup: drop memories past their expires_at

Designed to run on session expire and periodically via cron.

HISTORY: this module was originally written against the SQLite store
(`store._conn`, `?` placeholders). After the PostgreSQL migration it
crashed with AttributeError on every call — silently, because the gateway
wraps it in a debug-logged try/except. Rewritten 2026-06-12 for TenantDB.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

CONFIDENCE_DECAY = 0.95

MIN_CONFIDENCE = 0.1

PROMOTION_ACCESS_THRESHOLD = 3

LAYER_LIMITS = {
    "session": 50,
    "episodic": 200,
    "semantic": 500,
    "company": 1000,
}

DECAY_INACTIVITY_DAYS = 7


class MemoryConsolidator:
    """Heuristic memory maintenance — no LLM needed.

    Called at session expire and on cron. Works against the PostgreSQL
    MemoryStore/MemoryManager via its TenantDB handle (`store._db`).
    """

    def __init__(self, memory_store):
        """
        Args:
            memory_store: MemoryStore/MemoryManager instance (must expose _db).
        """
        self.store = memory_store
        self._db = getattr(memory_store, "_db", None)

    def consolidate(self, user_id: str = None) -> Dict[str, int]:
        """Run all consolidation passes.

        Args:
            user_id: If provided, only consolidate this user's memories.

        Returns:
            Dict with counts: decayed, promoted, evicted
        """
        stats = {"decayed": 0, "promoted": 0, "evicted": 0}
        if self._db is None:
            return stats

        stats["decayed"] = self._decay_stale(user_id)
        stats["promoted"] = self._promote_frequent(user_id)
        stats["evicted"] = self._evict_over_capacity(user_id)

        if any(v > 0 for v in stats.values()):
            logger.info("memory.consolidated user=%s %s", user_id, stats)

        return stats

    def _user_clause(self, user_id, params):
        if user_id:
            params.append(user_id)
            return " AND user_id = %s"
        return ""

    def _decay_stale(self, user_id: str = None) -> int:
        """Confidence-decay memories not accessed in DECAY_INACTIVITY_DAYS,
        then evict anything that decayed below the floor."""
        try:
            params = [CONFIDENCE_DECAY, DECAY_INACTIVITY_DAYS]
            clause = self._user_clause(user_id, params)
            params.append(MIN_CONFIDENCE)
            decayed = self._db.execute(f"""
                UPDATE ag_memory.memories
                SET confidence = confidence * %s
                WHERE last_accessed < NOW() - (%s || ' days')::interval
                  {clause}
                  AND confidence > %s
            """, tuple(params))

            self._db.execute("""
                DELETE FROM ag_memory.memories WHERE confidence <= %s
            """, (MIN_CONFIDENCE,))
            return decayed or 0
        except Exception as e:
            logger.warning("memory.decay_failed: %s", e)
            return 0

    def _promote_frequent(self, user_id: str = None) -> int:
        """Promote session memories with high access counts to semantic."""
        try:
            params = [PROMOTION_ACCESS_THRESHOLD]
            clause = self._user_clause(user_id, params)
            count = self._db.execute(f"""
                UPDATE ag_memory.memories
                SET layer = 'semantic', updated_at = NOW()
                WHERE layer = 'session' AND access_count >= %s
                  {clause}
            """, tuple(params))
            if count:
                logger.info("memory.promoted %d session→semantic", count)
            return count or 0
        except Exception as e:
            logger.warning("memory.promote_failed: %s", e)
            return 0

    def _evict_over_capacity(self, user_id: str = None) -> int:
        """Evict lowest-importance memories when a layer exceeds capacity."""
        total = 0
        try:
            for layer, limit in LAYER_LIMITS.items():
                params = [layer]
                clause = self._user_clause(user_id, params)
                params.append(limit)
                rows = self._db.fetch_all(f"""
                    SELECT user_id, COUNT(*) AS cnt FROM ag_memory.memories
                    WHERE layer = %s {clause}
                    GROUP BY user_id HAVING COUNT(*) > %s
                """, tuple(params))

                for row in rows or []:
                    overflow = row["cnt"] - limit
                    if overflow <= 0:
                        continue
                    self._db.execute("""
                        DELETE FROM ag_memory.memories WHERE memory_id IN (
                            SELECT memory_id FROM ag_memory.memories
                            WHERE user_id = %s AND layer = %s
                            ORDER BY confidence ASC, last_accessed ASC
                            LIMIT %s
                        )
                    """, (row["user_id"], layer, overflow))
                    total += overflow
        except Exception as e:
            logger.warning("memory.evict_failed: %s", e)
        return total

    def cleanup_expired(self) -> int:
        """Remove memories that have passed their expires_at timestamp."""
        if self._db is None:
            return 0
        try:
            return self._db.execute("""
                DELETE FROM ag_memory.memories
                WHERE expires_at IS NOT NULL AND expires_at < NOW()
            """) or 0
        except Exception as e:
            logger.warning("memory.cleanup_failed: %s", e)
            return 0
