"""
Agent — Teams Messaging Tools
=========================================
Proactive Teams messaging via Bot Framework REST + Microsoft Graph API.

Tools:
- teams_send_message:    Send a 1:1 Teams message to any org member
- teams_post_channel:    Post a message to a Teams channel
- teams_list_channels:   List channels in a team

Architecture:
    Uses SYNC HTTP calls (requests library) — same pattern as
    Microsoft365Tools, OutlookActionTools, JiraActionTools.
    No async bridges, no Bot Framework SDK dependency.

    Bot Framework REST API (direct HTTP):
        POST {service_url}/v3/conversations       → Create 1:1 chat
        POST {service_url}/v3/conversations/{id}/activities → Send message

    Microsoft Graph API:
        GET  /v1.0/users?$filter=...              → Resolve user by name/email
        GET  /v1.0/groups?$filter=...              → Find team by name
        GET  /v1.0/teams/{id}/channels             → List channels
        POST /v1.0/teams/{id}/channels/{id}/messages → Post to channel

Credentials resolved via CredentialRegistry (vault + bindings).
"""

import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_URL = "https://smba.trafficmanager.net/emea/"

BOT_TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
BOT_TOKEN_SCOPE = "https://api.botframework.com/.default"

GRAPH_TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
GRAPH_TOKEN_SCOPE = "https://graph.microsoft.com/.default"


class TeamsMessagingTools:
    """Proactive Teams messaging: send DMs and post to channels.

    Follows the same pattern as Microsoft365Tools / OutlookActionTools:
    - Sync HTTP calls (requests library)
    - Credentials via CredentialRegistry
    - UserContext resolver for per-user scoping
    - Registered via NativeMCP
    """

    def __init__(self, credential_registry=None, user_context_resolver=None,
                 app_id: str = None):
        self._registry = credential_registry
        self._user_context_resolver = user_context_resolver
        self._app_id = app_id

        self._cached_config = None


    def _get_config(self) -> dict:
        """Resolve Teams bot config: client_id, client_secret, tenant_id.

        Source: CredentialRegistry (provider 'm365') — per-user or system.
        The vault is the single source; an empty config means not configured.
        """
        if self._registry:
            user_id = None
            app_id = self._app_id
            if self._user_context_resolver:
                try:
                    ctx = self._user_context_resolver()
                    if ctx and hasattr(ctx, "user_id") and ctx.user_id:
                        uid_str = str(ctx.user_id).split("_")[-1]
                        user_id = int(uid_str) if uid_str.isdigit() else None
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
                logger.debug("teams_messaging._get_config registry failed: %s", e)

        return {"client_id": "", "client_secret": "", "tenant_id": ""}


    def _get_bot_token(self, config: dict) -> str:
        """Get a Bot Framework token for proactive messaging.

        Uses the Microsoft Identity Platform (v2.0) endpoint with
        client_credentials grant — same as Bot Framework SDK does internally.
        """
        tenant_id = config.get("tenant_id", "")
        if not tenant_id:
            raise ValueError("tenant_id not configured")
        resp = requests.post(BOT_TOKEN_URL.format(tenant_id=tenant_id), data={
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "scope": BOT_TOKEN_SCOPE,
            "grant_type": "client_credentials",
        }, timeout=10)
        resp.raise_for_status()
        return resp.json()["access_token"]

    def _get_graph_token(self, config: dict) -> str:
        """Get a Microsoft Graph API token for user/team resolution."""
        tenant_id = config.get("tenant_id", "")
        if not tenant_id:
            raise ValueError("tenant_id not configured")
        resp = requests.post(GRAPH_TOKEN_URL.format(tenant_id=tenant_id), data={
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "scope": GRAPH_TOKEN_SCOPE,
            "grant_type": "client_credentials",
        }, timeout=10)
        resp.raise_for_status()
        return resp.json()["access_token"]


    def _resolve_user(self, identifier: str, graph_token: str) -> Optional[dict]:
        """Resolve a user by email or display name via Graph API.

        Args:
            identifier: Email address or display name
            graph_token: Graph API access token

        Returns:
            dict with 'id' (AAD Object ID), 'displayName', 'mail' or None
        """
        headers = {"Authorization": f"Bearer {graph_token}"}

        if "@" in identifier:
            resp = requests.get(
                f"https://graph.microsoft.com/v1.0/users/{identifier}",
                headers=headers,
                params={"$select": "id,displayName,mail,userPrincipalName"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "id": data["id"],
                    "displayName": data.get("displayName", ""),
                    "mail": data.get("mail") or data.get("userPrincipalName", identifier),
                }

        resp = requests.get(
            "https://graph.microsoft.com/v1.0/users",
            headers=headers,
            params={
                "$filter": f"startswith(displayName,'{identifier}')",
                "$select": "id,displayName,mail,userPrincipalName",
                "$top": 5,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            users = resp.json().get("value", [])
            if users:
                for u in users:
                    if u.get("displayName", "").lower() == identifier.lower():
                        return {
                            "id": u["id"],
                            "displayName": u.get("displayName", ""),
                            "mail": u.get("mail") or u.get("userPrincipalName", ""),
                        }
                u = users[0]
                return {
                    "id": u["id"],
                    "displayName": u.get("displayName", ""),
                    "mail": u.get("mail") or u.get("userPrincipalName", ""),
                }

        return None


    def _send_dm(self, config: dict, recipient: str, message: str) -> str:
        """Resolve a recipient and send them a 1:1 Teams message.

        Shared by teams_send_message and teams_notify_me. Returns a ✓/✗ line.
        Raises requests.HTTPError on transport failure (callers convert it).
        """
        try:
            graph_token = self._get_graph_token(config)
            user = self._resolve_user(recipient, graph_token)
            if not user:
                return (f"✗ Could not find user '{recipient}' in your organization. "
                        f"Try using their full email address.")

            aad_id = user["id"]
            display_name = user["displayName"]
            user_mail = user["mail"]

            logger.info("[TeamsMsg] Sending to %s (aad=%s, mail=%s)",
                        display_name, aad_id[:8], user_mail)

            bot_token = self._get_bot_token(config)

            tenant_id = config.get("tenant_id", "")
            create_resp = requests.post(
                f"{DEFAULT_SERVICE_URL}v3/conversations",
                headers={
                    "Authorization": f"Bearer {bot_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "bot": {"id": config["client_id"]},
                    "members": [{"id": aad_id}],
                    "channelData": {"tenant": {"id": tenant_id}},
                    "isGroup": False,
                },
                timeout=15,
            )
            create_resp.raise_for_status()
            conversation_id = create_resp.json().get("id")

            if not conversation_id:
                return f"✗ Failed to create conversation with {display_name}"

            send_resp = requests.post(
                f"{DEFAULT_SERVICE_URL}v3/conversations/{conversation_id}/activities",
                headers={
                    "Authorization": f"Bearer {bot_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "type": "message",
                    "text": message,
                },
                timeout=15,
            )
            send_resp.raise_for_status()

            return f"✓ Teams message sent to {display_name} ({user_mail})"
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            body = e.response.text[:200] if e.response is not None else ""
            logger.error("[TeamsMsg] DM failed: %s %s", status, body)
            return f"✗ Teams send failed ({status}): {body}"


    def register_tools(self, mcp_server):
        instance = self

        @mcp_server.tool()
        def teams_send_message(recipient: str, message: str) -> str:
            """Send a direct Teams message to a person by name or email.

            Creates a 1:1 chat with the recipient and sends the message.
            The message appears as a personal chat from the AI Agent bot.

            Args:
                recipient: Person's email (e.g. 'user@company.com') or display name (e.g. 'Krishan Kant')
                message: The message text to send
            """
            try:
                config = instance._get_config()
                if not config.get("client_id") or not config.get("client_secret"):
                    return "✗ Teams bot not configured. Set Azure app credentials in Credential Vault."
                return instance._send_dm(config, recipient, message)
            except Exception as e:
                logger.error("[TeamsMsg] Send failed to %s: %s", recipient, e, exc_info=True)
                return f"✗ Teams send failed: {str(e)}"

        @mcp_server.tool()
        def teams_notify_me(message: str) -> str:
            """Send a direct Teams message to the CURRENT user (yourself).

            Use this to notify the person you're working for — e.g. "ping me on
            Teams when the deploy finishes" or "DM me a summary". It resolves the
            current user's own mailbox automatically; no recipient needed.

            Args:
                message: The message text to send to the current user
            """
            try:
                config = instance._get_config()
                if not config.get("client_id") or not config.get("client_secret"):
                    return "✗ Teams bot not configured. Set Azure app credentials in Credential Vault."

                recipient = config.get("user_email") or config.get("email") or ""
                if not recipient and instance._user_context_resolver:
                    try:
                        ctx = instance._user_context_resolver()
                        name = getattr(ctx, "user_name", "") or ""
                        if "@" in name:
                            recipient = name
                    except Exception:
                        pass
                if not recipient:
                    return ("✗ Could not determine your own address to DM. Connect "
                            "your Microsoft 365 mailbox under Settings → Connections.")

                return instance._send_dm(config, recipient, message)
            except Exception as e:
                logger.error("[TeamsMsg] notify_me failed: %s", e, exc_info=True)
                return f"✗ Teams notify failed: {str(e)}"

        @mcp_server.tool()
        def teams_post_channel(team_name: str, channel_name: str, message: str) -> str:
            """Post a message to a Microsoft Teams channel.

            Finds the team and channel by name and posts the message.
            Supports HTML formatting in the message body.

            Args:
                team_name: The team name (e.g. 'My Software Team')
                channel_name: The channel name (e.g. 'General')
                message: The message to post (HTML supported)
            """
            try:
                config = instance._get_config()
                if not config.get("client_id") or not config.get("client_secret"):
                    return "✗ Teams bot not configured. Set Azure app credentials in Credential Vault."

                graph_token = instance._get_graph_token(config)
                headers = {
                    "Authorization": f"Bearer {graph_token}",
                    "Content-Type": "application/json",
                }

                resp = requests.get(
                    "https://graph.microsoft.com/v1.0/groups",
                    headers=headers,
                    params={
                        "$filter": (f"displayName eq '{team_name}' and "
                                    "resourceProvisioningOptions/Any(x:x eq 'Team')"),
                        "$select": "id,displayName",
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                teams = resp.json().get("value", [])
                if not teams:
                    return f"✗ Team '{team_name}' not found in your organization."
                team_id = teams[0]["id"]

                resp = requests.get(
                    f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels",
                    headers=headers,
                    params={"$filter": f"displayName eq '{channel_name}'"},
                    timeout=10,
                )
                resp.raise_for_status()
                channels = resp.json().get("value", [])
                if not channels:
                    resp2 = requests.get(
                        f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels",
                        headers=headers,
                        params={"$select": "displayName"},
                        timeout=10,
                    )
                    available = [c["displayName"] for c in resp2.json().get("value", [])]
                    return (f"✗ Channel '{channel_name}' not found in '{team_name}'. "
                            f"Available: {', '.join(available)}")
                channel_id = channels[0]["id"]

                html_body = message.replace('\n', '<br>') if '<br' not in message.lower() else message
                resp = requests.post(
                    f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages",
                    headers=headers,
                    json={
                        "body": {
                            "contentType": "html",
                            "content": html_body,
                        }
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                return f"✓ Posted to {team_name} / #{channel_name}"

            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                body = e.response.text[:200] if e.response is not None else ""
                logger.error("[TeamsMsg] Channel post failed: %s %s", status, body)
                return f"✗ Channel post failed ({status}): {body}"
            except Exception as e:
                logger.error("[TeamsMsg] Channel post failed: %s", e, exc_info=True)
                return f"✗ Channel post failed: {str(e)}"

        @mcp_server.tool()
        def teams_list_channels(team_name: str) -> str:
            """List all channels in a Microsoft Teams team.

            Args:
                team_name: The team name (e.g. 'My Software Team')
            """
            try:
                config = instance._get_config()
                if not config.get("client_id") or not config.get("client_secret"):
                    return "✗ Teams bot not configured."

                graph_token = instance._get_graph_token(config)
                headers = {"Authorization": f"Bearer {graph_token}"}

                resp = requests.get(
                    "https://graph.microsoft.com/v1.0/groups",
                    headers=headers,
                    params={
                        "$filter": (f"displayName eq '{team_name}' and "
                                    "resourceProvisioningOptions/Any(x:x eq 'Team')"),
                        "$select": "id,displayName",
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                teams = resp.json().get("value", [])
                if not teams:
                    return f"✗ Team '{team_name}' not found."
                team_id = teams[0]["id"]

                resp = requests.get(
                    f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels",
                    headers=headers,
                    params={"$select": "displayName,description,membershipType"},
                    timeout=10,
                )
                resp.raise_for_status()
                channels = resp.json().get("value", [])
                if not channels:
                    return f"No channels found in '{team_name}'."

                lines = [f"**Channels in {team_name}:**\n"]
                for ch in channels:
                    desc = ch.get("description") or ""
                    mtype = ch.get("membershipType", "standard")
                    lines.append(f"• **#{ch['displayName']}** ({mtype})"
                                 + (f" — {desc}" if desc else ""))
                return "\n".join(lines)

            except Exception as e:
                return f"✗ Error listing channels: {str(e)}"
