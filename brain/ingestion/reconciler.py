"""
Brain reconciler — heals incomplete multi-store ingestion.
═══════════════════════════════════════════════════════════════════════════════
Ingestion writes the optional pillars (Elasticsearch, Neo4j, pgvector) best-effort
and records each outcome in ag_brain.document_index_state. This reconciler is the
self-healing half: every tick it pulls documents whose pillars are 'pending' or
'failed' (and due) and re-runs JUST that pillar via StorageManager.reindex —
idempotent upserts, so re-running is always safe.

Why this exists (and a backfill script does not suffice): a backfill heals the
gap that exists today, but the NEXT time a pillar is briefly down at ingest time
the gap silently returns. The reconciler makes the pipeline eventually
consistent — pillars converge to 'ok' on their own whenever the service is
reachable, and 'failed' rows surface the real problem (e.g. an empty Neo4j
password) instead of vanishing.

Per-tenant: each tenant's StorageManager points at that tenant's DB + ES index +
graph key, so the gateway runs one tick per active tenant (like cron fan-out).
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("brain.reconciler")


class BrainReconciler:
    """Re-attempts non-'ok' pillars for a single tenant's StorageManager."""

    def __init__(self, storage_manager, batch_size: int = 50, extractor=None):
        """
        Args:
            storage_manager: a CONNECTED StorageManager for one tenant.
            batch_size: max (doc, pillar) units healed per tick.
            extractor: optional GraphExtractor so the graph pillar can re-extract
                entities on reindex. When None, graph reindex stores nothing new
                (still marked ok) — set it to actually rebuild the graph.
        """
        self.storage = storage_manager
        self.batch_size = batch_size
        if extractor is not None:
            self.storage.extractor = extractor

    async def tick(self) -> dict:
        """One reconciliation pass. Returns a small stats dict for logging/health.

        reindex() does blocking network I/O (ES / embedding endpoint / Neo4j), so
        each unit runs in a worker thread to keep the event loop free.
        """
        try:
            due = self.storage.index_state.due(self.batch_size)
        except Exception as e:
            logger.warning("reconciler.due_query_failed: %s", e)
            return {"processed": 0, "healed": 0, "failed": 0, "error": str(e)}

        if not due:
            return {"processed": 0, "healed": 0, "failed": 0}

        healed = failed = 0
        for doc_id, pillar in due:
            try:
                ok, _err = await asyncio.to_thread(self.storage.reindex, doc_id, pillar)
            except Exception as e:
                logger.warning("reconciler.reindex_crashed: doc=%s pillar=%s err=%s",
                               doc_id, pillar, e)
                ok = False
            healed += 1 if ok else 0
            failed += 0 if ok else 1

        if healed or failed:
            logger.info("reconciler.tick: processed=%d healed=%d failed=%d",
                        len(due), healed, failed)
        return {"processed": len(due), "healed": healed, "failed": failed}
