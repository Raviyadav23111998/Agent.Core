"""
Index state store — per-document, per-pillar ingestion state.
═══════════════════════════════════════════════════════════════════════════════
Backs the eventually-consistent ingestion pipeline (see V028_brain_index_state).
Each optional pillar (lexical/Elasticsearch, graph/Neo4j, vector/pgvector) records
its sync state for every document so the reconciler can re-attempt anything that
isn't `ok`. The Postgres doc store + FTS are atomic with the row and not tracked.

States: pending (needs indexing) · ok (done) · failed (errored, will retry) ·
skipped (pillar not configured — nothing to do).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("brain.index_state")

PILLARS: Tuple[str, ...] = ("lexical", "graph", "vector")

_BACKOFF_BASE_S = 60
_BACKOFF_CAP_S = 3600


def _backoff_seconds(attempts: int) -> int:
    return min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2 ** max(0, attempts - 1)))


class IndexStateStore:
    """CRUD over ag_brain.document_index_state."""

    def __init__(self, db):
        self._db = db

    def seed(self, doc_id: str, configured: Dict[str, bool]) -> None:
        """Seed a row per pillar for a new document.

        configured[pillar] True  → 'pending' (the writer/reconciler will index it)
        configured[pillar] False → 'skipped' (pillar not wired — nothing to do)

        Does NOT clobber an existing row (a re-ingest keeps prior state; the
        inline write in ingest_document re-marks it).
        """
        doc_id = int(doc_id)
        for pillar in PILLARS:
            state = "pending" if configured.get(pillar) else "skipped"
            self._db.execute(
                """
                INSERT INTO ag_brain.document_index_state (doc_id, pillar, state)
                VALUES (%s, %s, %s)
                ON CONFLICT (doc_id, pillar) DO NOTHING
                """,
                (doc_id, pillar, state),
            )

    def mark(self, doc_id: str, pillar: str, state: str, error: str = None) -> None:
        """Record the outcome of a pillar write.

        ok/skipped → retry cleared. failed → attempts++ and backoff scheduled.
        Upserts so it works whether or not seed() ran first.
        """
        doc_id = int(doc_id)
        if state == "failed":
            self._db.execute(
                """
                INSERT INTO ag_brain.document_index_state
                    (doc_id, pillar, state, attempts, last_error, next_retry_at, updated_at)
                VALUES (%s, %s, 'failed', 1, %s,
                        now() + (%s || ' seconds')::interval, now())
                ON CONFLICT (doc_id, pillar) DO UPDATE SET
                    state = 'failed',
                    attempts = ag_brain.document_index_state.attempts + 1,
                    last_error = EXCLUDED.last_error,
                    next_retry_at = now() + (%s || ' seconds')::interval,
                    updated_at = now()
                """,
                (doc_id, pillar, (error or "")[:1000],
                 _backoff_seconds(1), _backoff_seconds(self._attempts(doc_id, pillar) + 1)),
            )
        else:
            self._db.execute(
                """
                INSERT INTO ag_brain.document_index_state
                    (doc_id, pillar, state, last_error, next_retry_at, updated_at)
                VALUES (%s, %s, %s, NULL, now(), now())
                ON CONFLICT (doc_id, pillar) DO UPDATE SET
                    state = EXCLUDED.state,
                    last_error = NULL,
                    next_retry_at = now(),
                    updated_at = now()
                """,
                (doc_id, pillar, state),
            )

    def _attempts(self, doc_id, pillar: str) -> int:
        row = self._db.fetch_one(
            "SELECT attempts FROM ag_brain.document_index_state WHERE doc_id=%s AND pillar=%s",
            (int(doc_id), pillar),
        )
        return int(row["attempts"]) if row else 0

    def due(self, limit: int = 100) -> List[Tuple[str, str]]:
        """Return up to `limit` (doc_id, pillar) pairs needing (re)indexing now."""
        rows = self._db.fetch_all(
            """
            SELECT doc_id, pillar
            FROM ag_brain.document_index_state
            WHERE state IN ('pending', 'failed') AND next_retry_at <= now()
            ORDER BY next_retry_at
            LIMIT %s
            """,
            (limit,),
        )
        return [(r["doc_id"], r["pillar"]) for r in rows]

    def health(self) -> Dict[str, Dict[str, int]]:
        """Counts per pillar per state, e.g. {'vector': {'ok': 6, 'pending': 110}}."""
        rows = self._db.fetch_all(
            "SELECT pillar, state, count(*) AS n FROM ag_brain.document_index_state GROUP BY pillar, state",
            (),
        )
        out: Dict[str, Dict[str, int]] = {p: {} for p in PILLARS}
        for r in rows:
            out.setdefault(r["pillar"], {})[r["state"]] = int(r["n"])
        return out
