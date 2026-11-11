"""
Credential verification — "Test connection" for stored credentials.
═══════════════════════════════════════════════════════════════════

Given a resolved credential (config + decrypted secret), make a minimal live
call to the provider to confirm it actually works. Returns (ok, message) so
the UI can show a green/red badge instead of the user discovering a bad
credential only when a tool fails mid-task.

Every check is best-effort and bounded (short timeouts); a failure returns a
human-readable reason, never raises.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

logger = logging.getLogger("agent.credential_verify")

_TIMEOUT = 10


def verify_credential(provider: str, config: Dict, secret: str,
                      connections=None) -> Tuple[bool, str]:
    """Test a credential against its live service. Returns (ok, message).

    `connections` (the Connections engine) is passed through for dynamic API
    integrations, which need a tenant-routed DB to look up the integration's
    probe operation. Static providers ignore it.
    """
    try:
        if provider.startswith("api_"):
            return _verify_dynamic(provider, config or {}, secret or "", connections)
        fn = _VERIFIERS.get(provider)
        if fn is None:
            return False, f"Connection test isn't available for '{provider}'."
        return fn(config or {}, secret or "")
    except Exception as e:
        logger.debug("verify.%s failed: %s", provider, e)
        return False, f"Connection failed: {e}"


def _verify_dynamic(provider: str, config: Dict, secret: str,
                    connections=None) -> Tuple[bool, str]:
    """Generic 'Test connection' for a dynamic API integration ('api_{slug}').

    Delegates to the integrations runtime, which builds an authenticated probe
    request through the egress guard and reports the HTTP outcome. Degrades to a
    clear, non-failing message (so the credential can still be SAVED) when the
    runtime or a DB context is unavailable.
    """
    try:
        from core.integrations.runtime import verify_integration_credential
    except Exception:
        return True, ("Saved. Live connection test for API integrations is not "
                      "available in this build.")
    return verify_integration_credential(provider, config, secret,
                                         connections=connections, timeout=_TIMEOUT)


def _verify_jira(config: Dict, secret: str) -> Tuple[bool, str]:
    import requests
    server = (config.get("server_url") or config.get("server") or "").rstrip("/")
    email = config.get("email") or ""
    if not (server and email and secret):
        return False, "Missing site URL, email, or API token."
    r = requests.get(f"{server}/rest/api/3/myself",
                     auth=(email, secret), timeout=_TIMEOUT,
                     headers={"Accept": "application/json"})
    if r.status_code == 200:
        name = (r.json() or {}).get("displayName", email)
        return True, f"Connected to Jira as {name}."
    if r.status_code in (401, 403):
        return False, "Authentication failed — check the email and API token."
    return False, f"Jira returned HTTP {r.status_code}."


def _verify_m365(config: Dict, secret: str) -> Tuple[bool, str]:
    import requests
    tenant = config.get("tenant_id")
    client = config.get("client_id")
    if not (tenant and client and secret):
        return False, "Missing app registration (tenant/client id/secret) — set by an admin."
    resp = requests.post(
        f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
        data={"client_id": client, "client_secret": secret,
              "scope": "https://graph.microsoft.com/.default",
              "grant_type": "client_credentials"},
        timeout=_TIMEOUT)
    if resp.status_code == 200 and resp.json().get("access_token"):
        mbox = config.get("user_email")
        return True, f"Microsoft 365 app authorized{(' for ' + mbox) if mbox else ''}."
    detail = ""
    try:
        detail = resp.json().get("error_description", "")[:120]
    except Exception:
        pass
    return False, f"M365 token request failed (HTTP {resp.status_code}). {detail}"


def _verify_smtp(config: Dict, secret: str) -> Tuple[bool, str]:
    import smtplib
    server = config.get("server")
    port = int(config.get("port", 587) or 587)
    username = config.get("username")
    if not (server and username and secret):
        return False, "Missing SMTP server, username, or password."
    with smtplib.SMTP(server, port, timeout=_TIMEOUT) as conn:
        if config.get("use_tls", True):
            conn.starttls()
        conn.login(username, secret)
    return True, f"SMTP login succeeded on {server}:{port}."


def _verify_imap(config: Dict, secret: str) -> Tuple[bool, str]:
    import imaplib
    server = config.get("server")
    port = int(config.get("port", 993) or 993)
    username = config.get("username")
    if not (server and username and secret):
        return False, "Missing IMAP server, username, or password."
    conn = imaplib.IMAP4_SSL(server, port, timeout=_TIMEOUT)
    try:
        conn.login(username, secret)
    finally:
        try:
            conn.logout()
        except Exception:
            pass
    return True, f"IMAP login succeeded on {server}."


def _verify_telegram(config: Dict, secret: str) -> Tuple[bool, str]:
    """Validate a personal Telegram bot token via getMe. The token is the
    single secret; a 200 with ok=true confirms @BotFather issued it."""
    import requests
    token = (secret or "").strip()
    if not token:
        return False, "Missing bot token — create one with @BotFather."
    r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=_TIMEOUT)
    if r.status_code == 200 and (r.json() or {}).get("ok"):
        me = (r.json().get("result") or {})
        uname = me.get("username") or me.get("first_name") or "bot"
        return True, f"Connected as @{uname}."
    if r.status_code == 401:
        return False, "Invalid bot token — re-copy it from @BotFather."
    return False, f"Telegram returned HTTP {r.status_code}."


_VERIFIERS = {
    "jira": _verify_jira,
    "m365": _verify_m365,
    "smtp": _verify_smtp,
    "imap": _verify_imap,
    "telegram": _verify_telegram,
}
