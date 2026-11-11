"""
Agent Brain — Outlook Connector (Enterprise)
========================================================
Connects to Outlook Mail via Microsoft Graph API for incremental sync.

Implements BaseConnector protocol:
- fetch_incremental(since) → SyncItems with summaries + full content
- fetch_item(uri) → single email on-demand

Architecture:
    Uses raw `requests` with OAuth2 Client Credentials flow against
    Microsoft Graph API. No azure-identity or httpx dependencies.
    Credentials resolved via AppConfig (PostgreSQL-backed), matching
    the same pattern used by tools/microsoft365.py.

Credentials: the system ``m365`` credential in the vault (the same app
registration the chat tools use). Resolved via CredentialRegistry.
"""

import logging
import time
from typing import Optional

import requests as http_requests

from brain.ingestion.connectors.base_connector import BaseConnector, SyncItem

logger = logging.getLogger(__name__)


class OutlookConnector(BaseConnector):
    """Connects to Outlook Mail via Microsoft Graph for incremental sync.

    Credentials are resolved via AppConfig at call time. Authentication
    uses OAuth2 Client Credentials flow (app-only), identical to the
    working tools/microsoft365.py implementation.
    """

    _cached_token: Optional[str] = None
    _token_expiry: float = 0

    def __init__(self, app_config=None, credential_registry=None):
        """Initialize the Outlook connector.

        Args:
            app_config: AppConfig instance (kept for compatibility; the M365
                        app is no longer sourced from config). If None, falls
                        back to the global singleton.
            credential_registry: CredentialRegistry — the single source for the
                        M365 app registration (vault). Without it, the connector
                        reports not-configured and ingestion skips gracefully.
        """
        if app_config is None:
            from core.config import app_config as _cfg
            app_config = _cfg
        self._app_config = app_config
        self._registry = credential_registry


    def _resolve_credentials(self) -> dict:
        """Resolve the M365 app registration from the vault (single source).

        The system m365 credential in the vault is the one authoritative
        source for the app registration — the same one the chat tools use.
        Returns an empty dict (→ ``is_configured`` False) when unavailable,
        so ingestion degrades gracefully instead of crashing.
        """
        empty = {"tenant_id": "", "client_id": "", "client_secret": "", "user_email": ""}
        if self._registry is None:
            return empty
        try:
            cred = self._registry.resolve("m365")
            cfg = dict(cred.config)
            cfg["client_secret"] = cred.secret
            return cfg
        except Exception as e:
            logger.debug("outlook_connector.vault_resolve_failed: %s", e)
            return empty

    def _get_access_token(self, config: dict) -> str:
        """Fetch an OAuth2 token using the Client Credentials Flow.

        Caches the token and only refreshes when it's about to expire
        (5-minute buffer). This avoids hitting the Azure AD token endpoint
        on every Graph API call during a sync cycle.

        Raises:
            requests.HTTPError: If token acquisition fails.
        """
        now = time.time()
        if self._cached_token and now < self._token_expiry - 300:
            return self._cached_token

        token_url = (
            f"https://login.microsoftonline.com/"
            f"{config['tenant_id']}/oauth2/v2.0/token"
        )
        payload = {
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials",
        }

        resp = http_requests.post(token_url, data=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        self._cached_token = data["access_token"]
        self._token_expiry = now + data.get("expires_in", 3600)

        logger.info(
            "outlook_connector.token_acquired: expires_in=%ds",
            data.get("expires_in", 0),
        )
        return self._cached_token


    @property
    def source_type(self) -> str:
        return "outlook"

    @property
    def is_configured(self) -> bool:
        config = self._resolve_credentials()
        return bool(
            config["tenant_id"]
            and config["client_id"]
            and config["client_secret"]
            and config["user_email"]
        )

    def fetch_incremental(
        self,
        since: Optional[str] = None,
        max_results: int = 100,
    ) -> list[SyncItem]:
        """Fetch emails received/updated since the given timestamp.

        Args:
            since: ISO timestamp (e.g., "2026-05-28T10:00:00Z").
                   If None, fetches the most recent max_results emails.
            max_results: Max emails per sync cycle.
        """
        config = self._resolve_credentials()
        if not all(config.values()):
            logger.debug("outlook_connector.not_configured")
            return []

        try:
            token = self._get_access_token(config)
        except Exception as e:
            logger.error("outlook_connector.auth_failed: %s", str(e))
            return []

        try:
            logger.info(
                "outlook_connector.incremental_sync: since=%s max=%d",
                since or "full", max_results,
            )

            url = (
                f"https://graph.microsoft.com/v1.0"
                f"/users/{config['user_email']}/messages"
            )
            headers = {
                "Authorization": f"Bearer {token}",
                "Prefer": 'outlook.body-content-type="text"',
            }
            params = {
                "$top": max_results,
                "$orderby": "receivedDateTime desc",
                "$select": (
                    "id,subject,sender,receivedDateTime,bodyPreview,"
                    "body,toRecipients,ccRecipients,importance,isRead"
                ),
            }
            if since:
                try:
                    from datetime import datetime, timezone
                    since_clean = since
                    if since_clean.endswith('+00'): since_clean += ':00'
                    dt = datetime.fromisoformat(since_clean)
                    dt = dt.astimezone(timezone.utc)
                    since_iso = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                except Exception:
                    since_iso = since.replace(' ', 'T')
                    if not since_iso.endswith('Z') and '+' not in since_iso:
                        since_iso += 'Z'
                params["$filter"] = f"receivedDateTime gt {since_iso}"

            resp = http_requests.get(
                url, headers=headers, params=params, timeout=30,
            )
            resp.encoding = "utf-8"

            if resp.status_code != 200:
                logger.error(
                    "outlook_connector.api_error: status=%s body=%s",
                    resp.status_code, resp.text[:300],
                )
                return []

            messages = resp.json().get("value", [])

            items = []
            for msg in messages:
                item = self._message_to_sync_item(msg)
                if item:
                    items.append(item)

            logger.info(
                "outlook_connector.synced: count=%d since=%s",
                len(items), since or "full",
            )
            return items

        except Exception as e:
            logger.error("outlook_connector.incremental_failed: %s", str(e))
            return []

    def fetch_item(self, source_uri: str) -> Optional[SyncItem]:
        """Fetch a single email by URI (e.g., 'outlook://msgid123')."""
        config = self._resolve_credentials()
        if not all(config.values()):
            return None

        msg_id = source_uri.replace("outlook://", "")

        try:
            token = self._get_access_token(config)
            url = (
                f"https://graph.microsoft.com/v1.0"
                f"/users/{config['user_email']}/messages/{msg_id}"
            )
            headers = {
                "Authorization": f"Bearer {token}",
                "Prefer": 'outlook.body-content-type="text"',
            }
            params = {
                "$select": (
                    "id,subject,sender,receivedDateTime,bodyPreview,"
                    "body,toRecipients,ccRecipients,importance,isRead"
                ),
            }

            resp = http_requests.get(
                url, headers=headers, params=params, timeout=15,
            )
            resp.encoding = "utf-8"

            if resp.status_code != 200:
                logger.error(
                    "outlook_connector.fetch_item_error: status=%s",
                    resp.status_code,
                )
                return None

            msg = resp.json()
            return self._message_to_sync_item(msg)

        except Exception as e:
            logger.error(
                "outlook_connector.fetch_item_failed: id=%s error=%s",
                msg_id[:20], str(e),
            )
            return None


    def _message_to_sync_item(self, msg: dict) -> Optional[SyncItem]:
        """Convert a Graph API message JSON to a SyncItem."""
        try:
            msg_id = msg.get("id", "")
            subject = msg.get("subject", "No Subject")
            date = msg.get("receivedDateTime", "")

            sender_dict = msg.get("sender", {}).get("emailAddress", {})
            sender_name = sender_dict.get("name", "")
            sender_email = sender_dict.get("address", "")
            sender = (
                f"{sender_name} <{sender_email}>"
                if sender_name else sender_email
            )

            to_recipients = []
            for r in msg.get("toRecipients", []):
                addr = r.get("emailAddress", {})
                name = addr.get("name", addr.get("address", ""))
                to_recipients.append(name)

            cc_recipients = []
            for r in msg.get("ccRecipients", []):
                addr = r.get("emailAddress", {})
                name = addr.get("name", addr.get("address", ""))
                cc_recipients.append(name)

            body_content = msg.get("body", {}).get("content", "")
            body_type = msg.get("body", {}).get("contentType", "text")
            if body_type == "html":
                body_text = self._strip_html(body_content)
            else:
                body_text = body_content

            content_parts = [
                f"From: {sender}",
                f"To: {', '.join(to_recipients)}",
                f"Subject: {subject}",
                f"Date: {date}",
            ]
            if cc_recipients:
                content_parts.append(f"CC: {', '.join(cc_recipients)}")
            content_parts.extend(["\n--- Body ---", body_text])

            full_content = "\n".join(content_parts)

            body_preview = msg.get("bodyPreview", "")
            summary = self._extract_summary(
                f"{subject}. {body_preview}. "
                f"From {sender_name or sender_email}."
            )

            people = [sender_name or sender_email] + to_recipients + cc_recipients
            people = [p for p in people if p]

            is_read = msg.get("isRead", False)
            importance = msg.get("importance", "normal")

            return SyncItem(
                source_uri=f"outlook://{msg_id}",
                source_type="outlook",
                title=f"Email: {subject}",
                summary=summary,
                full_content=full_content,
                people=people,
                status="read" if is_read else "unread",
                updated_at=date,
                metadata={
                    "message_id": msg_id,
                    "subject": subject,
                    "sender": sender,
                    "sender_email": sender_email,
                    "importance": importance,
                    "to": to_recipients,
                    "cc": cc_recipients,
                },
            )
        except Exception as e:
            logger.error("outlook_connector.parse_failed: %s", str(e))
            return None

    @staticmethod
    def _strip_html(html: str) -> str:
        """Strip HTML tags to plain text."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            import re
            text = re.sub(r'<[^>]+>', '', html)
            return text.strip()
