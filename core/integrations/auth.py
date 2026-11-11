"""
core.integrations.auth — pluggable authentication strategies.
═══════════════════════════════════════════════════════════════════════════════
One AuthStrategy per `auth_type`. A strategy turns a resolved credential into
request auth and (if token-based) mints / caches / refreshes the token. This
replaces the inline `static | oauth2 | hmac` switch in the executor so new schemes
(login_grant, oauth2_password, token_exchange) are added by REGISTERING a strategy,
not by editing the executor.

`auth_type` is authoritative. When an integration predates it (only `security_scheme`
is stored) the strategy is derived from that — so existing integrations keep working
with no migration. The heavy primitives (OAuth token mint, HMAC signing, static
header/query application) stay on the executor / runtime module and are reused 1:1.

See docs/INTEGRATION_AUTH_ARCHITECTURE.md and docs/INTEGRATION_AUTH_IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

API_KEY = "api_key"
BEARER_STATIC = "bearer_static"
BASIC = "basic"
CUSTOM = "custom"
HMAC = "hmac"
OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"
OAUTH2_PASSWORD = "oauth2_password"
LOGIN_GRANT = "login_grant"


def _auth_config(op: dict) -> dict:
    ac = op.get("auth_config") or {}
    if isinstance(ac, str):
        try:
            ac = json.loads(ac)
        except Exception:
            ac = {}
    return ac if isinstance(ac, dict) else {}


class AuthStrategy:
    """Apply credentials to a request. A `token_based` strategy is refreshed once
    on a 401 (the access token may simply be stale)."""

    token_based = False

    def apply(self, executor, op: dict, req, cred_config: dict, secret: str,
              *, principal=None, force_refresh: bool = False, connection_id=None) -> None:
        raise NotImplementedError

    def is_token_based(self) -> bool:
        return self.token_based


class StaticStrategy(AuthStrategy):
    """apiKey / bearer / basic / custom — the op's `auth_config` already encodes
    where the secret goes ({in: header|query|basic, name, fmt})."""

    def apply(self, executor, op, req, cred_config, secret, *, principal=None,
              force_refresh: bool = False, connection_id=None):
        from core.integrations.runtime import _apply_auth
        _apply_auth(req, _auth_config(op), cred_config, secret)


class HmacStrategy(AuthStrategy):
    """Sign the request with the integration's HMAC recipe."""

    def apply(self, executor, op, req, cred_config, secret, *, principal=None,
              force_refresh: bool = False, connection_id=None):
        from core.integrations.runtime import _apply_hmac
        _apply_hmac(req, _auth_config(op), secret)


def _bearer(req, token):
    from core.integrations.runtime import _apply_auth
    _apply_auth(req, {"in": "header", "name": "Authorization", "fmt": "Bearer {token}"}, {}, token)


class OAuth2ClientCredentialsStrategy(AuthStrategy):
    """Server-to-server OAuth2: mint (and cache/refresh) an access token, then send
    it as a Bearer header. Token minting lives on the executor (shared, tenant-keyed
    cache + egress-guarded fetch)."""

    token_based = True

    def apply(self, executor, op, req, cred_config, secret, *, principal=None,
              force_refresh: bool = False, connection_id=None):
        token = executor._oauth_token(op, cred_config, secret, _auth_config(op),
                                      force_refresh=force_refresh)
        _bearer(req, token)


class LoginGrantStrategy(AuthStrategy):
    """Live-token via the API's OWN login endpoint (e.g. fleet /auth/Login). The
    stored secret is a {access_token, refresh_token} bundle from the connect-time
    exchange — never the password. The executor mints/refreshes/rotates and caches
    per (tenant, provider, cred, principal); we just attach the Bearer header."""

    token_based = True

    def apply(self, executor, op, req, cred_config, secret, *, principal=None,
              force_refresh: bool = False, connection_id=None):
        token = executor._login_grant_token(op, cred_config, secret, _auth_config(op),
                                            principal=principal, force_refresh=force_refresh,
                                            connection_id=connection_id)
        _bearer(req, token)


class OAuth2PasswordStrategy(AuthStrategy):
    """OAuth2 password grant — like client-credentials but per-user; refreshes with
    grant_type=refresh_token from the stored bundle."""

    token_based = True

    def apply(self, executor, op, req, cred_config, secret, *, principal=None,
              force_refresh: bool = False, connection_id=None):
        token = executor._oauth_password_token(op, cred_config, secret, _auth_config(op),
                                               principal=principal, force_refresh=force_refresh,
                                               connection_id=connection_id)
        _bearer(req, token)


_STATIC = StaticStrategy()
STRATEGIES: Dict[str, AuthStrategy] = {
    API_KEY: _STATIC,
    BEARER_STATIC: _STATIC,
    BASIC: _STATIC,
    CUSTOM: _STATIC,
    HMAC: HmacStrategy(),
    OAUTH2_CLIENT_CREDENTIALS: OAuth2ClientCredentialsStrategy(),
    OAUTH2_PASSWORD: OAuth2PasswordStrategy(),
    LOGIN_GRANT: LoginGrantStrategy(),
}

_SCHEME_TO_TYPE = {
    "apikey": API_KEY, "bearer": BEARER_STATIC, "basic": BASIC,
    "custom": CUSTOM, "hmac": HMAC, "oauth2": OAUTH2_CLIENT_CREDENTIALS,
}


def resolve(auth_type: Optional[str], security_scheme: Optional[str] = None) -> AuthStrategy:
    """The strategy for an op. `auth_type` wins; else derive from the legacy
    `security_scheme`; else static (an op that still names an auth_provider but
    declares no special scheme — e.g. a plain header key)."""
    if auth_type and auth_type in STRATEGIES:
        return STRATEGIES[auth_type]
    if security_scheme:
        t = _SCHEME_TO_TYPE.get(security_scheme.lower())
        if t:
            return STRATEGIES[t]
    return _STATIC
