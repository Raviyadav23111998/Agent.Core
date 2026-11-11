"""
Agent Brain — Base Connector + SyncItem
====================================================
Standard interface for all data source connectors.

Every connector (Jira, Outlook, Slack) implements BaseConnector to provide:
- Incremental sync (fetch only new/updated items since last sync)
- On-demand single-item fetch (for content cache misses)
- Standardized output format (SyncItem)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SyncItem:
    """Standardized item from any data source.
    
    The connector normalizes source-specific formats into this common structure.
    The SyncWorker writes different parts to different storage layers:
      - summary + metadata → Search Index (permanent)
      - full_content → Content Cache (TTL)
      - people + relations → Entity Index (permanent)
    """
    source_uri: str
    source_type: str
    title: str
    summary: str
    full_content: str
    people: list[str] = field(default_factory=list)
    status: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def content_size(self) -> int:
        """Size of full content in bytes."""
        return len(self.full_content.encode("utf-8"))

    @property
    def summary_size(self) -> int:
        """Size of summary in bytes."""
        return len(self.summary.encode("utf-8"))


class BaseConnector(ABC):
    """Standard interface for data source connectors.
    
    All connectors must implement:
    - source_type: identifier string
    - is_configured: whether credentials are present
    - fetch_incremental: fetch items updated since a timestamp
    - fetch_item: fetch a single item by URI
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Source identifier: 'jira', 'outlook', 'slack'."""
        ...

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Whether credentials/tokens are present and valid."""
        ...

    @abstractmethod
    def fetch_incremental(
        self,
        since: Optional[str] = None,
        max_results: int = 100,
    ) -> list[SyncItem]:
        """Fetch items created/updated since the given timestamp.
        
        Args:
            since: ISO timestamp — only fetch items updated after this.
                   If None, fetches the most recent `max_results` items.
            max_results: Maximum items to return per sync cycle.
            
        Returns:
            List of SyncItems, newest first.
        """
        ...

    @abstractmethod
    def fetch_item(self, source_uri: str) -> Optional[SyncItem]:
        """Fetch a single item by its source URI.
        
        Used for on-demand content cache refills when a cached item expires
        and the user queries it.
        
        Args:
            source_uri: The unique source identifier (e.g., "jira://PROJ/JIRA-1234")
            
        Returns:
            SyncItem if found, None if not found or access denied.
        """
        ...

    def get_sync_state(self) -> dict:
        """Optional: return connector-specific sync state info."""
        return {
            "source_type": self.source_type,
            "is_configured": self.is_configured,
        }

    @staticmethod
    def _extract_summary(text: str, max_chars: int = 300) -> str:
        """Extract a summary from raw text (first N chars, cleaned up)."""
        if not text:
            return ""
        cleaned = " ".join(text.split())
        if len(cleaned) <= max_chars:
            return cleaned
        truncated = cleaned[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > max_chars * 0.7:
            truncated = truncated[:last_space]
        return truncated + "..."
