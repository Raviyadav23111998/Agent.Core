"""
Agent — Outlook Action Tools
=========================================
Email write operations via Microsoft Graph REST API.

Tools:
- outlook_reply:   Reply to an email by message ID
- outlook_send:    Send a new email
- outlook_forward: Forward an email to another recipient

Credentials resolved exclusively via Connections (ag_cred). No fallback.
"""

import requests
import logging

logger = logging.getLogger(__name__)


class OutlookActionTools:
    """Email reply/send/forward via Microsoft Graph API.
    Credentials resolved exclusively via Connections (ag_cred). No fallback.
    """

    def __init__(self, credential_registry=None, user_context_resolver=None, app_id: str = None):
        self._registry = credential_registry
        self._user_context_resolver = user_context_resolver
        self._app_id = app_id

    def _get_config(self) -> dict:
        """Resolve M365 config via Connections. Returns {} when not configured."""
        if not self._registry:
            return {}
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
        try:
            cred = self._registry.resolve("m365", user_id=user_id, app_id=app_id)
            config = dict(cred.config)
            config["client_secret"] = cred.secret
            return config
        except Exception as e:
            logger.debug("outlook._get_config: not configured (%s)", e)
            return {}

    @property
    def is_configured(self) -> bool:
        config = self._get_config()
        return bool(config.get("tenant_id") and config.get("client_id")
                    and config.get("client_secret") and config.get("user_email"))

    def _get_access_token(self, config: dict) -> str:
        """Fetch OAuth2 token using client credentials flow."""
        token_url = f"https://login.microsoftonline.com/{config['tenant_id']}/oauth2/v2.0/token"
        payload = {
            "client_id": config['client_id'],
            "client_secret": config['client_secret'],
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials",
        }
        resp = requests.post(token_url, data=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["access_token"]

    def _graph_url(self, path: str, config: dict) -> str:
        """Build Graph API URL for user's mailbox."""
        return f"https://graph.microsoft.com/v1.0/users/{config['user_email']}/{path}"

    def register_tools(self, mcp_server):
        instance = self

        @mcp_server.tool()
        def outlook_reply(message_id: str, body: str) -> str:
            """Reply to an email by its message ID.

            Sends a reply to the sender of the specified email.

            Args:
                message_id: The email message ID (from fetch_recent_emails)
                body: The reply body text (HTML supported)
            """
            try:
                config = instance._get_config()
                if not config.get("tenant_id") or not config.get("client_secret"):
                    return "✗ Reply failed: M365 Outlook integration is not configured for this user."
                
                token = instance._get_access_token(config)
                url = instance._graph_url(f"messages/{message_id}/reply", config)
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=utf-8",
                }
                html_body = body.replace('\n', '<br>') if '<br' not in body.lower() else body
                payload = {
                    "message": {
                        "body": {
                            "contentType": "HTML",
                            "content": html_body
                        }
                    }
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=15)
                resp.raise_for_status()
                return f"✓ Reply sent to message {message_id[:20]}..."
            except requests.HTTPError as e:
                return f"✗ Reply failed: {e.response.status_code} — {e.response.text[:200]}"
            except Exception as e:
                return f"✗ Reply error: {str(e)}"

        @mcp_server.tool()
        def outlook_send(to: str, subject: str, body: str) -> str:
            """Send a new email.

            Args:
                to: Recipient email address
                subject: Email subject line
                body: Email body text (HTML supported)
            """
            try:
                config = instance._get_config()
                if not config.get("tenant_id") or not config.get("client_secret"):
                    return "✗ Send failed: M365 Outlook integration is not configured for this user."

                token = instance._get_access_token(config)
                url = instance._graph_url("sendMail", config)
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=utf-8",
                }
                html_body = body.replace('\n', '<br>') if '<br' not in body.lower() else body
                payload = {
                    "message": {
                        "subject": subject,
                        "body": {
                            "contentType": "HTML",
                            "content": html_body,
                        },
                        "toRecipients": [
                            {"emailAddress": {"address": to}}
                        ],
                    }
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=15)
                resp.raise_for_status()
                return f"✓ Email sent to {to}: \"{subject}\""
            except requests.HTTPError as e:
                return f"✗ Send failed: {e.response.status_code} — {e.response.text[:200]}"
            except Exception as e:
                return f"✗ Send error: {str(e)}"

        @mcp_server.tool()
        def outlook_forward(message_id: str, to: str, comment: str = "") -> str:
            """Forward an email to another recipient.

            Args:
                message_id: The email message ID to forward
                to: Recipient email address
                comment: Optional comment to include above the forwarded content
            """
            try:
                config = instance._get_config()
                if not config.get("tenant_id") or not config.get("client_secret"):
                    return "✗ Forward failed: M365 Outlook integration is not configured for this user."

                token = instance._get_access_token(config)
                url = instance._graph_url(f"messages/{message_id}/forward", config)
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=utf-8",
                }
                payload = {
                    "comment": comment,
                    "toRecipients": [
                        {"emailAddress": {"address": to}}
                    ],
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=15)
                resp.raise_for_status()
                return f"✓ Email forwarded to {to}"
            except requests.HTTPError as e:
                return f"✗ Forward failed: {e.response.status_code} — {e.response.text[:200]}"
            except Exception as e:
                return f"✗ Forward error: {str(e)}"
