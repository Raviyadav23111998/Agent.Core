"""
Agent Brain — Sync Worker
======================================
Orchestrates incremental sync across all configured data sources.

Enterprise pattern:
1. Load last_sync_at per source from sync_state.json
2. For each configured connector: fetch_incremental(since=last_sync_at)
3. For each SyncItem:
   a. Store summary in DocumentStore (permanent search index)
   b. Cache full content in ContentStore (TTL=24h)
   c. Extract entities → EntityIndex (permanent relationships)
4. Update last_sync_at

Also handles on-demand content fetching for cache misses.
"""

from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from brain.models import DocumentType
from brain.storage.document_store import DocumentStore
from brain.storage.content_store import ContentStore
from brain.storage.entity_index import EntityIndex
from brain.ingestion.connectors.base_connector import BaseConnector, SyncItem
from core.tenant import TENANT_SCOPE

logger = logging.getLogger(__name__)


_SOURCE_TO_DOC_TYPE = {
    "jira": DocumentType.TICKET,
    "outlook": DocumentType.EMAIL,
    "slack": DocumentType.CHAT_MESSAGE,
}


@dataclass
class SyncReport:
    """Result of a sync cycle."""
    source_type: str
    items_synced: int = 0
    items_updated: int = 0
    items_new: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: int = 0


class SyncWorker:
    """Orchestrates incremental sync across all configured data sources.
    
    Usage:
        worker = SyncWorker(
            document_store=storage.documents,
            content_store=content_store,
            entity_index=entity_index,
            connectors=[jira_conn, outlook_conn, slack_conn],
        )
        
        # Manual sync
        reports = await worker.sync_all()
        
        # On-demand fetch (content cache miss)
        content = await worker.fetch_full_content("jira://PROJ-1234")
    """

    SYNC_STATE_FILE = ".ag_data/sync_state.json"
    DEFAULT_CONTENT_TTL = 86400

    def __init__(
        self,
        document_store: DocumentStore,
        content_store: ContentStore,
        entity_index: EntityIndex,
        connectors: list[BaseConnector] = None,
        llm_client=None,
        sync_state_path: str = None,
        db=None,
        default_app_id: int = 1,
    ):
        self._doc_store = document_store
        self._content_store = content_store
        self._entity_index = entity_index
        self._connectors: list[BaseConnector] = connectors or []
        self._llm = llm_client
        self._db = db
        self._default_app_id = default_app_id

        self._sync_state = self._load_state()


    def has_configured_sources(self) -> bool:
        """Check if any connectors are configured with credentials."""
        return any(c.is_configured for c in self._connectors)

    def get_configured_sources(self) -> list[str]:
        """Get list of configured source types."""
        return [c.source_type for c in self._connectors if c.is_configured]

    async def sync_all(self, max_results_per_source: int = 100) -> list[SyncReport]:
        """Run incremental sync for all configured sources.
        
        Returns list of SyncReport, one per source.
        """
        reports = []
        for connector in self._connectors:
            if not connector.is_configured:
                continue
            report = await self._sync_source(connector, max_results_per_source)
            reports.append(report)
        return reports

    async def sync_source(
        self, source_type: str, max_results: int = 100
    ) -> Optional[SyncReport]:
        """Sync a specific source by type name."""
        for connector in self._connectors:
            if connector.source_type == source_type and connector.is_configured:
                return await self._sync_source(connector, max_results)
        return None

    async def fetch_full_content(self, source_uri: str) -> Optional[str]:
        """On-demand fetch: check content cache → miss → fetch from API → cache.
        
        Used by Brain.ask() when SearchAgent returns a summary-only result
        and we need full content for LLM synthesis.
        """
        cached = self._content_store.get(source_uri, owner_id=TENANT_SCOPE)
        if cached:
            return cached

        source_type = source_uri.split("://")[0] if "://" in source_uri else None
        if not source_type:
            return None

        connector = self._find_connector(source_type)
        if not connector:
            return None

        try:
            item = connector.fetch_item(source_uri)
            if item and item.full_content:
                self._content_store.put(
                    source_uri, item.full_content,
                    ttl=self.DEFAULT_CONTENT_TTL,
                    source_type=source_type,
                )
                return item.full_content
        except Exception as e:
            logger.error("sync_worker.on_demand_fetch_failed: uri=%s error=%s",
                        source_uri[:60], str(e))
        return None

    def get_sync_status(self) -> dict:
        """Get comprehensive sync status for all sources."""
        status = {
            "configured_sources": self.get_configured_sources(),
            "sources": {},
        }
        for connector in self._connectors:
            st = connector.source_type
            source_state = self._sync_state.get(st, {})
            status["sources"][st] = {
                "configured": connector.is_configured,
                "last_sync_at": source_state.get("last_sync_at"),
                "total_items_synced": source_state.get("total_items_synced", 0),
                "last_sync_items": source_state.get("last_sync_items", 0),
                "last_sync_errors": source_state.get("last_sync_errors", 0),
                "last_sync_duration_ms": source_state.get("last_sync_duration_ms", 0),
            }

        status["content_cache"] = self._content_store.get_stats()
        status["entity_index"] = self._entity_index.get_stats()

        return status


    async def _sync_source(
        self, connector: BaseConnector, max_results: int
    ) -> SyncReport:
        """Run incremental sync for a single source."""
        source_type = connector.source_type
        report = SyncReport(source_type=source_type)

        source_state = self._sync_state.get(source_type, {})
        since = source_state.get("last_sync_at")

        start_time = _now_ms()
        logger.info("sync_worker.sync_start: source=%s since=%s",
                    source_type, since or "full")

        try:
            items = await asyncio.to_thread(
                connector.fetch_incremental,
                since=since, max_results=max_results
            )
            report.items_synced = len(items)

            for item in items:
                try:
                    is_new = self._process_sync_item(item)
                    if is_new:
                        report.items_new += 1
                    else:
                        report.items_updated += 1
                except Exception as e:
                    error_msg = f"{item.source_uri}: {str(e)[:100]}"
                    report.errors.append(error_msg)
                    logger.error("sync_worker.item_failed: %s", error_msg)

        except Exception as e:
            report.errors.append(str(e))
            logger.error("sync_worker.sync_failed: source=%s error=%s",
                        source_type, str(e))

        report.duration_ms = _now_ms() - start_time

        if report.items_synced > 0:
            self._sync_state[source_type] = {
                "last_sync_at": datetime.now(timezone.utc).isoformat(),
                "total_items_synced": source_state.get("total_items_synced", 0) + report.items_synced,
                "last_sync_items": report.items_synced,
                "last_sync_errors": len(report.errors),
                "last_sync_duration_ms": report.duration_ms,
            }
        else:
            existing_state = self._sync_state.get(source_type, {})
            existing_state["last_sync_items"] = 0
            existing_state["last_sync_errors"] = len(report.errors)
            existing_state["last_sync_duration_ms"] = report.duration_ms
            self._sync_state[source_type] = existing_state
        self._save_state()

        logger.info(
            "sync_worker.sync_complete: source=%s new=%d updated=%d errors=%d duration=%dms",
            source_type, report.items_new, report.items_updated,
            len(report.errors), report.duration_ms,
        )
        return report

    def _process_sync_item(self, item: SyncItem) -> bool:
        """Process a single SyncItem across all storage layers.
        
        Returns True if this is a new item, False if update.
        """
        doc_type = _SOURCE_TO_DOC_TYPE.get(item.source_type, DocumentType.DOCUMENT)

        existing = self._doc_store.find_by_source_path(item.source_uri)
        is_new = existing is None

        if is_new:
            doc = self._doc_store.add_document(
                content=item.summary,
                source_path=item.source_uri,
                document_type=doc_type,
                title=item.title,
                metadata={
                    **item.metadata,
                    "source_type": item.source_type,
                    "people": item.people,
                    "status": item.status,
                    "updated_at": item.updated_at,
                    "full_content_size": item.content_size,
                },
                deduplicate=False,
                owner_id=TENANT_SCOPE,
                app_id=self._default_app_id,
            )
            
            from brain.models import IngestionStatus
            self._doc_store.update_status(doc.id, IngestionStatus.COMPLETED)
        else:
            self._doc_store.update_document(
                doc_id=existing.id,
                content=item.summary,
                title=item.title,
                metadata={
                    **item.metadata,
                    "source_type": item.source_type,
                    "people": item.people,
                    "status": item.status,
                    "updated_at": item.updated_at,
                    "full_content_size": item.content_size,
                },
            )
            
            from brain.models import IngestionStatus
            self._doc_store.update_status(existing.id, IngestionStatus.COMPLETED)

        self._content_store.put(
            item.source_uri,
            item.full_content,
            ttl=self.DEFAULT_CONTENT_TTL,
            source_type=item.source_type,
            owner_id=TENANT_SCOPE,
            app_id=self._default_app_id,
        )

        self._entity_index.bulk_ingest_from_sync_item(
            source_uri=item.source_uri,
            people=item.people,
            title=item.title,
            source_type=item.source_type,
            metadata=item.metadata,
            app_id=self._default_app_id,
        )

        return is_new

    def _find_connector(self, source_type: str) -> Optional[BaseConnector]:
        """Find a connector by source type."""
        for c in self._connectors:
            if c.source_type == source_type and c.is_configured:
                return c
        return None


    def _load_state(self) -> dict:
        """Load sync state from PostgreSQL (ag_brain.sync_state)."""
        if not self._db:
            return {}
        try:
            rows = self._db.fetch_all(
                "SELECT source_key, last_sync_at, docs_synced, status FROM ag_brain.sync_state"
            )
            state = {}
            for row in rows:
                state[row["source_key"]] = {
                    "last_sync_at": str(row["last_sync_at"]) if row.get("last_sync_at") else None,
                    "total_items_synced": row.get("docs_synced", 0),
                }
            return state
        except Exception as e:
            logger.warning("sync_worker.state_load_failed: %s", e)
            return {}

    def _save_state(self):
        """Persist sync state to PostgreSQL (ag_brain.sync_state)."""
        if not self._db:
            return
        try:
            for source_type, state in self._sync_state.items():
                self._db.execute("""
                    INSERT INTO ag_brain.sync_state (source_key, last_sync_at, docs_synced, status)
                    VALUES (%s, %s, %s, 'idle')
                    ON CONFLICT (source_key) DO UPDATE SET
                        last_sync_at = EXCLUDED.last_sync_at,
                        docs_synced = EXCLUDED.docs_synced,
                        status = 'idle'
                """, (
                    source_type,
                    state.get("last_sync_at"),
                    state.get("total_items_synced", 0),
                ))
        except Exception as e:
            logger.error("sync_worker.state_save_failed: %s", e)


def _now_ms() -> int:
    """Current time in milliseconds."""
    import time
    return int(time.time() * 1000)
