"""
Agent Brain — Slack Connector (Enterprise)
======================================================
Ingests messages from Slack channels using the Slack Web API.

Implements BaseConnector protocol:
- fetch_incremental(since) → SyncItems with summaries + full content
- fetch_item(uri) → single message on-demand

Credentials resolved via CredentialRegistry (vault + bindings).
Fallback: SLACK_BOT_TOKEN env var.
Scopes: channels:history, channels:read, groups:read, users:read
"""

import os
from typing import Optional
import logging

from brain.models import DocumentType
from brain.ingestion.parsers.base import ParsedDocument
from brain.ingestion.connectors.base_connector import BaseConnector, SyncItem

logger = logging.getLogger(__name__)


class SlackConnector(BaseConnector):
    """Connects to Slack via the Web API for incremental sync."""

    BASE_URL = "https://slack.com/api"

    def __init__(self, credential_registry=None):
        self._registry = credential_registry
        self.token = ""
        if credential_registry:
            try:
                cred = credential_registry.resolve("slack")
                self.token = cred.secret
            except Exception as e:
                logger.debug("slack._init registry failed: %s", e)
        if not self.token:
            self.token = os.getenv("SLACK_BOT_TOKEN", "")
        self._users_cache: dict[str, str] = {}
        self._channels_cache: dict[str, str] = {}
        self.sync_channels: list[str] = []


    @property
    def source_type(self) -> str:
        return "slack"

    @property
    def is_configured(self) -> bool:
        return bool(self.token)

    def fetch_incremental(
        self,
        since: Optional[str] = None,
        max_results: int = 100,
    ) -> list[SyncItem]:
        """Fetch messages from all configured channels since timestamp.
        
        Args:
            since: ISO timestamp — fetch messages after this time.
            max_results: Max messages per channel
        """
        if not self.is_configured:
            return []

        channels = self._get_sync_channels()
        if not channels:
            logger.warning("slack_connector.no_channels")
            return []

        oldest = None
        if since:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
                oldest = str(dt.timestamp())
            except (ValueError, AttributeError):
                oldest = None

        items = []
        for channel_id, channel_name in channels:
            channel_items = self._fetch_channel_messages(
                channel_id, channel_name, oldest=oldest,
                max_results=max_results
            )
            items.extend(channel_items)

        logger.info("slack_connector.synced: total=%d channels=%d since=%s",
                    len(items), len(channels), since or "full")
        return items

    def fetch_item(self, source_uri: str) -> Optional[SyncItem]:
        """Fetch a single message by URI (e.g., 'slack://C0123/1234567890.123456')."""
        if not self.is_configured:
            return None

        parts = source_uri.replace("slack://", "").split("/", 1)
        if len(parts) != 2:
            return None
        channel_id, ts = parts

        try:
            data = self._api_get("conversations.history", {
                "channel": channel_id,
                "latest": ts,
                "inclusive": "true",
                "limit": "1",
            })
            if not data.get("ok"):
                return None

            messages = data.get("messages", [])
            if not messages:
                return None

            channel_name = self._resolve_channel(channel_id)
            return self._message_to_sync_item(
                messages[0], channel_id, channel_name
            )
        except Exception as e:
            logger.error("slack_connector.fetch_item_failed: uri=%s error=%s",
                        source_uri, str(e))
            return None


    def _api_get(self, endpoint: str, params: dict = None) -> dict:
        """Make a GET request to the Slack API."""
        import httpx
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                f"{self.BASE_URL}/{endpoint}",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                params=params or {},
            )
            data = resp.json()
            if not data.get("ok"):
                logger.error("slack_connector.api_error: endpoint=%s error=%s",
                           endpoint, data.get("error"))
            return data

    def _get_sync_channels(self) -> list[tuple[str, str]]:
        """Get channels to sync. Returns list of (channel_id, name)."""
        if self.sync_channels:
            return [(cid, self._resolve_channel(cid)) 
                    for cid in self.sync_channels]

        channels = []
        cursor = ""
        while True:
            params = {"types": "public_channel,private_channel", "limit": 200}
            if cursor:
                params["cursor"] = cursor

            data = self._api_get("conversations.list", params)
            if not data.get("ok"):
                break

            for ch in data.get("channels", []):
                cid = ch["id"]
                name = ch.get("name", "unknown")
                self._channels_cache[cid] = name
                channels.append((cid, name))

            cursor = data.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break

        return channels

    def _resolve_channel(self, channel_id: str) -> str:
        """Resolve channel ID to name (cached)."""
        if channel_id in self._channels_cache:
            return self._channels_cache[channel_id]
        data = self._api_get("conversations.info", {"channel": channel_id})
        if data.get("ok"):
            name = data.get("channel", {}).get("name", channel_id)
            self._channels_cache[channel_id] = name
            return name
        return channel_id

    def _resolve_user(self, user_id: str) -> str:
        """Resolve a Slack user ID to display name (cached)."""
        if user_id in self._users_cache:
            return self._users_cache[user_id]

        data = self._api_get("users.info", {"user": user_id})
        if data.get("ok"):
            user = data.get("user", {})
            name = (
                user.get("profile", {}).get("display_name")
                or user.get("real_name")
                or user.get("name", user_id)
            )
            self._users_cache[user_id] = name
            return name

        self._users_cache[user_id] = user_id
        return user_id

    def _fetch_channel_messages(
        self,
        channel_id: str,
        channel_name: str,
        oldest: str = None,
        max_results: int = 100,
    ) -> list[SyncItem]:
        """Fetch messages from a single channel."""
        items = []
        fetched = 0
        cursor = ""

        while fetched < max_results:
            params = {
                "channel": channel_id,
                "limit": str(min(200, max_results - fetched)),
            }
            if oldest:
                params["oldest"] = oldest
            if cursor:
                params["cursor"] = cursor

            data = self._api_get("conversations.history", params)
            if not data.get("ok"):
                break

            for msg in data.get("messages", []):
                subtype = msg.get("subtype", "")
                if subtype in ("bot_message", "channel_join", "channel_leave",
                              "channel_topic", "channel_purpose"):
                    continue

                item = self._message_to_sync_item(msg, channel_id, channel_name)
                if item:
                    items.append(item)
                    fetched += 1

            cursor = data.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break

        return items

    def _message_to_sync_item(
        self,
        msg: dict,
        channel_id: str,
        channel_name: str,
    ) -> Optional[SyncItem]:
        """Convert a Slack message to SyncItem."""
        text = msg.get("text", "").strip()
        if not text:
            return None

        user_id = msg.get("user", "unknown")
        sender_name = self._resolve_user(user_id)
        ts = msg.get("ts", "")

        try:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            updated_at = dt.isoformat()
        except (ValueError, TypeError):
            updated_at = ""

        content_parts = [
            f"Channel: #{channel_name}",
            f"Author: {sender_name}",
            f"Timestamp: {updated_at}",
            "\n--- Message ---",
            text,
        ]

        thread_content = []
        reply_count = msg.get("reply_count", 0)
        if reply_count > 0:
            thread_data = self._api_get("conversations.replies", {
                "channel": channel_id,
                "ts": ts,
                "limit": "20",
            })
            if thread_data.get("ok"):
                for reply in thread_data.get("messages", [])[1:]:
                    reply_user = self._resolve_user(reply.get("user", ""))
                    reply_text = reply.get("text", "")
                    thread_content.append(f"  {reply_user}: {reply_text}")

        if thread_content:
            content_parts.append(f"\n--- Thread ({reply_count} replies) ---")
            content_parts.extend(thread_content)

        for att in msg.get("attachments", []):
            title = att.get("title", att.get("fallback", "attachment"))
            content_parts.append(f"\n[Attachment: {title}]")
            if att.get("text"):
                content_parts.append(att["text"])

        full_content = "\n".join(content_parts)

        summary = self._extract_summary(
            f"#{channel_name} — {sender_name}: {text}"
        )

        people = [sender_name]
        import re
        mention_ids = re.findall(r'<@([A-Z0-9]+)>', text)
        for mid in mention_ids:
            people.append(self._resolve_user(mid))

        return SyncItem(
            source_uri=f"slack://{channel_id}/{ts}",
            source_type="slack",
            title=f"Slack: #{channel_name} — {sender_name}",
            summary=summary,
            full_content=full_content,
            people=list(set(people)),
            status="thread" if reply_count > 0 else "message",
            updated_at=updated_at,
            metadata={
                "channel_id": channel_id,
                "channel_name": channel_name,
                "user_id": user_id,
                "author": sender_name,
                "reply_count": reply_count,
                "timestamp": ts,
            },
        )


    def list_channels(self) -> list[dict]:
        """List all accessible channels (legacy method)."""
        channels_raw = self._get_sync_channels()
        return [{"id": cid, "name": name} for cid, name in channels_raw]

    def ingest_channel(
        self,
        channel_id: str,
        max_results: int = 100,
        include_threads: bool = True,
    ) -> list[ParsedDocument]:
        """Fetch messages from a channel (legacy method)."""
        channel_name = self._resolve_channel(channel_id)
        items = self._fetch_channel_messages(
            channel_id, channel_name, max_results=max_results
        )
        documents = []
        for item in items:
            doc = ParsedDocument(
                content=item.full_content,
                source_path=item.source_uri,
                document_type=DocumentType.CHAT_MESSAGE,
                title=item.title,
                metadata=item.metadata,
                raw_size_bytes=item.content_size,
            )
            documents.append(doc)
        return documents
