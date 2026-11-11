import requests
import logging
from typing import Callable
from .native_mcp import NativeMCP
from core.tenant_db import TenantDB

logger = logging.getLogger(__name__)

class Microsoft365Tools:
    """
    Direct, dependency-free Microsoft Graph integration using REST APIs.
    Credentials resolved exclusively via Connections (ag_cred). No fallback.
    """

    def __init__(self, credential_registry=None, user_context_resolver=None, app_id: str = None,
                 db_resolver: Callable[[], TenantDB] = None):
        self._registry = credential_registry
        self._user_context_resolver = user_context_resolver
        self._app_id = app_id
        self._db_resolver = db_resolver

    def _resolve_ctx(self):
        """Resolve (user_id, app_id) from the active UserContext."""
        user_id, app_id = None, self._app_id
        if self._user_context_resolver:
            try:
                ctx = self._user_context_resolver()
                if ctx and getattr(ctx, "user_id", None):
                    uid = str(ctx.user_id).split("_")[-1]
                    user_id = int(uid) if uid.isdigit() else None
                if not app_id and ctx and getattr(ctx, "app_id", None):
                    app_id = ctx.app_id
            except Exception:
                pass
        return user_id, app_id

    def _get_config(self) -> dict:
        """Resolve M365 config (app registration + mailbox + secret) via
        Connections. Returns {} when not configured — callers treat empty as
        'integration not set up'. There is no other source."""
        if not self._registry:
            return {}
        user_id, app_id = self._resolve_ctx()
        try:
            cred = self._registry.resolve("m365", user_id=user_id, app_id=app_id)
            config = dict(cred.config)
            config["client_secret"] = cred.secret
            return config
        except Exception as e:
            logger.debug("m365._get_config: not configured (%s)", e)
            return {}

    def _get_access_token(self, config: dict) -> str:
        """Fetches an OAuth2 token using the Client Credentials Flow."""
        token_url = f"https://login.microsoftonline.com/{config['tenant_id']}/oauth2/v2.0/token"
        payload = {
            "client_id": config['client_id'],
            "client_secret": config['client_secret'],
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials"
        }
        resp = requests.post(token_url, data=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["access_token"]
        
    def register_tools(self, mcp_server: NativeMCP):
        
        @mcp_server.tool()
        def fetch_recent_emails(queue_id: str, limit: int = 5) -> str:
            """Fetch UNPROCESSED emails from a preconfigured M365 processing queue. Requires a valid queue_id set up by an admin (for automated email-triage cron jobs). Do NOT use this for an ad-hoc 'check my email' request — if you have no queue_id, use search_emails (M365) or fetch_emails_imap (IMAP mailboxes) instead.

            Args:
                queue_id: The configured M365 queue identifier. If you don't have one, do not call this tool.
                limit: Max emails to fetch (default 5).
            """
            try:
                config = self._get_config()
                if not config.get("tenant_id") or not config.get("client_secret"):
                    return "✗ Failed: M365 Outlook integration is not configured for this user."

                token = self._get_access_token(config)
                url = f"https://graph.microsoft.com/v1.0/users/{config['user_email']}/messages"
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Prefer": 'outlook.body-content-type="text"'
                }
                params = {
                    "$top": max(limit * 2, 20),
                    "$select": "id,subject,from,receivedDateTime,bodyPreview,body"
                }
                
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                resp.raise_for_status()
                resp.encoding = 'utf-8'
                messages = resp.json().get("value", [])
                
                if not messages: return "Inbox is empty."
                
                if self._db_resolver:
                    try:
                        db = self._db_resolver()
                        with db.connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "SELECT message_id FROM ag_system.processed_emails WHERE queue_id = %s",
                                    (queue_id,)
                                )
                                processed_ids = {row[0] for row in cur.fetchall()}
                                
                                messages = [m for m in messages if m.get('id') not in processed_ids]
                    except Exception as e:
                        logger.error(f"Error checking idempotency table: {e}")
                
                messages = messages[:limit]
                
                if not messages: return "No new emails to process."
                
                output = []
                for msg in messages:
                    sender = msg.get("from", {}).get("emailAddress", {}).get("address", "Unknown")
                    body_content = msg.get("body", {}).get("content", "").strip()
                    output.append(f"ID: {msg.get('id', 'N/A')} | From: {sender} | Subject: {msg.get('subject')} | Preview: {msg.get('bodyPreview')} | Body: {body_content}")
                    
                return "\n".join(output)
            except Exception as e:
                return f"MS Graph API Error: {str(e)}"

        @mcp_server.tool()
        def mark_email_processed(message_id: str, queue_id: str) -> str:
            """Marks an email as processed in the database idempotency table for a specific queue. MUST be called after processing any email to prevent duplicate processing."""
            try:
                if self._db_resolver:
                    db = self._db_resolver()
                    with db.cursor() as cur:
                        cur.execute(
                            "INSERT INTO ag_system.processed_emails (message_id, queue_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (message_id, queue_id)
                        )
                    return f"✓ Successfully marked email {message_id} as processed in queue '{queue_id}'."
                else:
                    return "✗ Failed: Database connection not available for idempotency tracking."
            except Exception as e:
                return f"Database Error: {str(e)}"

        @mcp_server.tool()
        def search_emails(query: str = "", limit: int = 5) -> str:
            """Search the user's Outlook/M365 mailbox. This is the go-to tool for ad-hoc email requests like "check my latest email", "find emails about X", or "what did Y send me". Pass keywords to search, or leave empty to fetch the most recent emails. No queue_id needed.

            Args:
                query: Keywords to search subjects/bodies. Leave empty for latest/recent emails.
                limit: Max results (default 5).
            """
            try:
                config = self._get_config()
                if not config.get("tenant_id") or not config.get("client_secret"):
                    return "✗ Failed: M365 Outlook integration is not configured for this user."

                token = self._get_access_token(config)
                url = f"https://graph.microsoft.com/v1.0/users/{config['user_email']}/messages"
                headers = {"Authorization": f"Bearer {token}"}
                params = {"$top": limit, "$select": "id,subject,from,receivedDateTime,bodyPreview"}

                if query and query.strip():
                    params["$search"] = f'"{query.strip()}"'
                else:
                    params["$orderby"] = "receivedDateTime desc"

                resp = requests.get(url, headers=headers, params=params, timeout=10)
                resp.raise_for_status()
                resp.encoding = 'utf-8'
                messages = resp.json().get("value", [])
                
                label = f"matching '{query}'" if query and query.strip() else "in inbox"
                if not messages: return f"No emails found {label}."
                
                output = []
                for msg in messages:
                    sender = msg.get("from", {}).get("emailAddress", {}).get("address", "Unknown")
                    output.append(f"ID: {msg.get('id', 'N/A')} | From: {sender} | Subject: {msg.get('subject')} | Date: {msg.get('receivedDateTime', 'N/A')} | Preview: {msg.get('bodyPreview')}")
                    
                return "\n".join(output)
            except Exception as e:
                return f"MS Graph API Error: {str(e)}"
