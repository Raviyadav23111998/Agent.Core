from __future__ import annotations

"""
Agent Brain — Content Store (PostgreSQL TTL Cache)
==============================================================
LAYER 3: Temporary full-content cache with automatic expiration.

Enterprise pattern: Store summaries permanently (search index), cache full
content temporarily. On cache miss, fetch from source API and cache result.

Storage: PostgreSQL ag_brain.content_cache (via TenantDB)
"""

import hashlib
import logging
from typing import Optional, Callable

from core.tenant import TENANT_SCOPE

logger = logging.getLogger(__name__)


class ContentStore:
    """PostgreSQL-backed TTL content cache for full document text.

    Stores full email bodies, Jira descriptions, Slack threads temporarily.
    Auto-evicts expired items. Supports cache-through pattern.

    Usage:
        store = ContentStore(db=tenant_db)

        # Direct put/get
        store.put("jira://PROJ/JIRA-1234", full_text, ttl=86400)
        content = store.get("jira://PROJ/JIRA-1234")  # or None if expired

        # Cache-through (preferred)
        content = store.get_or_fetch(
            "jira://PROJ/JIRA-1234",
            fetch_fn=lambda: jira_api.get_issue("JIRA-1234").description
        )
    """

    DEFAULT_TTL = 86400

    def __init__(self, db=None):
        """
        Args:
            db: TenantDB instance for PostgreSQL access.
        """
        self._db = db
        self._hits = 0
        self._misses = 0
        if db is None:
            logger.warning("ContentStore initialized without TenantDB — caching disabled")

    def get(self, source_uri: str, owner_id: str = TENANT_SCOPE, app_id: int = None) -> Optional[str]:
        """Get cached content if not expired. Returns None on miss."""
        if not self._db:
            self._misses += 1
            return None

        if app_id:
            row = self._db.fetch_one(
                """SELECT content, fetched_at, ttl_seconds
                   FROM ag_brain.content_cache
                   WHERE source_uri = %s AND owner_id = %s AND app_id = %s
                     AND fetched_at + (ttl_seconds * INTERVAL '1 second') > NOW()""",
                (source_uri, owner_id, app_id),
            )
        else:
            row = self._db.fetch_one(
                """SELECT content, fetched_at, ttl_seconds
                   FROM ag_brain.content_cache
                   WHERE source_uri = %s AND owner_id = %s
                     AND fetched_at + (ttl_seconds * INTERVAL '1 second') > NOW()""",
                (source_uri, owner_id),
            )

        if not row:
            self._misses += 1
            return None

        self._hits += 1
        return row["content"]

    def put(
        self,
        source_uri: str,
        content: str,
        ttl: int = DEFAULT_TTL,
        source_type: str = "",
        owner_id: str = TENANT_SCOPE,
        app_id: int = 1,
    ):
        """Cache content with TTL (default 24h)."""
        if not self._db:
            return

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        size_bytes = len(content.encode("utf-8"))

        self._db.execute("""
            INSERT INTO ag_brain.content_cache
                (source_uri, owner_id, content, content_hash, fetched_at,
                 ttl_seconds, source_type, size_bytes, app_id)
            VALUES (%s, %s, %s, %s, NOW(), %s, %s, %s, %s)
            ON CONFLICT (source_uri, owner_id) DO UPDATE SET
                content = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                fetched_at = NOW(),
                ttl_seconds = EXCLUDED.ttl_seconds,
                source_type = EXCLUDED.source_type,
                size_bytes = EXCLUDED.size_bytes,
                app_id = EXCLUDED.app_id
        """, (source_uri, owner_id, content, content_hash, ttl, source_type, size_bytes, app_id))

        logger.debug("content_store.cached: uri=%s owner=%s app=%s size=%d ttl=%ds",
                     source_uri[:60], owner_id, app_id, size_bytes, ttl)

    def get_or_fetch(
        self,
        source_uri: str,
        fetch_fn: Callable[[], Optional[str]],
        ttl: int = DEFAULT_TTL,
        source_type: str = "",
        owner_id: str = TENANT_SCOPE,
        app_id: int = 1,
    ) -> Optional[str]:
        """Cache-through: return cached content or fetch, cache, and return."""
        cached = self.get(source_uri, owner_id=owner_id, app_id=app_id)
        if cached is not None:
            return cached

        try:
            content = fetch_fn()
            if content:
                self.put(source_uri, content, ttl=ttl, source_type=source_type,
                         owner_id=owner_id, app_id=app_id)
                return content
        except Exception as e:
            logger.error("content_store.fetch_failed: uri=%s error=%s",
                        source_uri[:60], str(e))
        return None

    def has(self, source_uri: str, owner_id: str = TENANT_SCOPE) -> bool:
        """Check if URI is cached and not expired (without loading content)."""
        if not self._db:
            return False
        return self._db.exists(
            """SELECT 1 FROM ag_brain.content_cache
               WHERE source_uri = %s AND owner_id = %s
                 AND fetched_at + (ttl_seconds * INTERVAL '1 second') > NOW()""",
            (source_uri, owner_id),
        )

    def invalidate(self, source_uri: str, owner_id: str = TENANT_SCOPE):
        """Explicitly remove a cached item."""
        if not self._db:
            return
        self._db.execute(
            "DELETE FROM ag_brain.content_cache WHERE source_uri = %s AND owner_id = %s",
            (source_uri, owner_id),
        )

    def evict_expired(self) -> int:
        """Bulk-remove all expired entries. Returns count evicted."""
        if not self._db:
            return 0
        count = self._db.execute(
            "DELETE FROM ag_brain.content_cache WHERE fetched_at + (ttl_seconds * INTERVAL '1 second') < NOW()"
        )
        if count > 0:
            logger.info("content_store.evicted: count=%d", count)
        return count

    def get_stats(self) -> dict:
        """Cache statistics: size, count, hit rate."""
        if not self._db:
            return {"total_items": 0, "total_bytes": 0, "total_mb": 0,
                    "expired_pending_eviction": 0, "hit_rate_percent": 0,
                    "hits": 0, "misses": 0, "by_source": {}}

        row = self._db.fetch_one(
            """SELECT COUNT(*) as count, COALESCE(SUM(size_bytes), 0) as total_bytes
               FROM ag_brain.content_cache"""
        )

        type_rows = self._db.fetch_all(
            """SELECT source_type, COUNT(*) as count, COALESCE(SUM(size_bytes), 0) as bytes
               FROM ag_brain.content_cache GROUP BY source_type"""
        )

        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

        expired = self._db.fetch_scalar(
            """SELECT COUNT(*) FROM ag_brain.content_cache
               WHERE fetched_at + (ttl_seconds * INTERVAL '1 second') < NOW()"""
        ) or 0

        return {
            "total_items": row["count"] if row else 0,
            "total_bytes": row["total_bytes"] if row else 0,
            "total_mb": round((row["total_bytes"] if row else 0) / (1024 * 1024), 2),
            "expired_pending_eviction": expired,
            "hit_rate_percent": round(hit_rate, 1),
            "hits": self._hits,
            "misses": self._misses,
            "by_source": {
                r.get("source_type") or "unknown": {
                    "count": r["count"],
                    "bytes": r["bytes"],
                }
                for r in type_rows
            },
        }

    def close(self):
        """No-op — connection lifecycle managed by TenantDB pool."""
        pass
