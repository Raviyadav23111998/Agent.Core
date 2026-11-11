"""
Phase 4 regression — auth depth (OAuth2 client-credentials + HMAC). No network.

  1. OAuth2: token fetched, attached as Bearer, cached, refreshed on 401.
  2. HMAC: deterministic signature header from the configured recipe.

    python -m pytest tests/test_integrations_phase4.py -q
"""

import hashlib
import hmac as _hmac
import json
import pytest

from core.integrations import runtime as rt
from core.integrations.runtime import DynamicHttpExecutor, _Req, _apply_hmac


class _Ctx:
    user_id, app_id = "user_5", "4"


class _Cred:
    def __init__(self, config, secret):
        self.config, self.secret = config, secret


class _Conn:
    def __init__(self, cred):
        self._cred, self._db = cred, object()
    def resolve_or_none(self, provider, *, user_id=None, app_id=None):
        return self._cred


class _Store:
    def __init__(self, op):
        self._op = op
    def get_op_for_execution(self, tool_name):
        return self._op if tool_name == self._op["tool_name"] else None


class _Sender:
    """Returns a queue of statuses (to simulate 401-then-200); records requests."""
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = []
    def __call__(self, method, url, req):
        self.calls.append((method, url, dict(req.headers), dict(req.params)))
        st = self.statuses.pop(0) if self.statuses else 200
        return st, {}, b'{"ok":true}'


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    monkeypatch.setattr(rt.socket, "getaddrinfo",
                        lambda host, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])


_OAUTH_OP = {
    "operation_id": "list", "tool_name": "acme__list", "http_method": "GET",
    "path_template": "/things", "base_url": "https://api.acme.example",
    "request_map": {"path": [], "query": [], "header": [], "has_body": False},
    "parameters_schema": {"type": "object", "properties": {}},
    "side_effect": "read", "requires_approval": False, "slug": "acme",
    "auth_provider": "api_acme", "security_scheme": "oauth2",
    "auth_config": {"flow": "client_credentials", "token_url": "https://auth.acme.example/token",
                    "scopes": ["read"]},
    "egress_allow": [], "allow_private": False,
}


def _oauth_exec(statuses, token_calls):
    sender = _Sender(statuses)
    def token_fetch(url, data):
        token_calls.append((url, dict(data)))
        return f"TOKEN{len(token_calls)}", 3600
    ex = DynamicHttpExecutor(
        db=object(), connections=_Conn(_Cred({"client_id": "cid", "token_url": ""}, "csecret")),
        http=sender, token_fetch=token_fetch, store=_Store(_OAUTH_OP))
    return ex, sender


def test_oauth2_fetches_and_attaches_bearer():
    tokens = []
    ex, sender = _oauth_exec([200], tokens)
    out = ex.call("acme__list", {}, ctx=_Ctx())
    assert "HTTP 200" in out
    assert tokens[0][0] == "https://auth.acme.example/token"
    assert tokens[0][1]["grant_type"] == "client_credentials"
    assert tokens[0][1]["client_id"] == "cid"
    assert tokens[0][1]["client_secret"] == "csecret"
    assert tokens[0][1]["scope"] == "read"
    assert sender.calls[0][2]["Authorization"] == "Bearer TOKEN1"


def test_oauth2_token_is_cached():
    tokens = []
    ex, _ = _oauth_exec([200, 200], tokens)
    ex.call("acme__list", {}, ctx=_Ctx())
    ex.call("acme__list", {}, ctx=_Ctx())
    assert len(tokens) == 1


def test_oauth2_token_cache_is_per_credential():
    """Two principals holding DIFFERENT client credentials for the SAME provider
    must NOT share a cached token — otherwise the first caller's token (identity)
    would be served as everyone else's within the tenant."""
    tokens = []
    def token_fetch(url, data):
        tokens.append(data.get("client_id"))
        return f"TOK{len(tokens)}", 3600
    sender = _Sender([200, 200, 200])
    conn = _Conn(_Cred({"client_id": "cidA", "token_url": ""}, "secA"))
    ex = DynamicHttpExecutor(db=object(), connections=conn, http=sender,
                             token_fetch=token_fetch, store=_Store(_OAUTH_OP))
    ex.call("acme__list", {}, ctx=_Ctx())
    conn._cred = _Cred({"client_id": "cidB", "token_url": ""}, "secB")
    ex.call("acme__list", {}, ctx=_Ctx())
    conn._cred = _Cred({"client_id": "cidA", "token_url": ""}, "secA")
    ex.call("acme__list", {}, ctx=_Ctx())
    assert tokens == ["cidA", "cidB"]


def test_oauth2_refreshes_on_401():
    tokens = []
    ex, sender = _oauth_exec([401, 200], tokens)
    out = ex.call("acme__list", {}, ctx=_Ctx())
    assert "HTTP 200" in out
    assert len(tokens) == 2
    assert sender.calls[1][2]["Authorization"] == "Bearer TOKEN2"


def test_oauth2_missing_token_url_errors():
    sender = _Sender([200])
    ex = DynamicHttpExecutor(
        db=object(), connections=_Conn(_Cred({"client_id": "c"}, "s")),
        http=sender, token_fetch=lambda u, d: ("T", 3600),
        store=_Store(dict(_OAUTH_OP, auth_config={"flow": "client_credentials"})))
    out = ex.call("acme__list", {}, ctx=_Ctx())
    assert "could not prepare authentication" in out.lower()
    assert "token_url" in out.lower()


def test_hmac_signature_default_recipe():
    req = _Req(method="POST", url="https://api.acme.example/things",
               json_body={"b": 2, "a": 1})
    _apply_hmac(req, {}, "shhh")
    body = json.dumps({"b": 2, "a": 1}, separators=(",", ":"), sort_keys=True)
    expect = _hmac.new(b"shhh", f"POST\n/things\n{body}".encode(), hashlib.sha256).hexdigest()
    assert req.headers["X-Signature"] == expect


def test_hmac_custom_header_and_fmt():
    req = _Req(method="GET", url="https://api.acme.example/x")
    _apply_hmac(req, {"header": "X-Sig", "fmt": "HMAC {sig}", "alg": "sha1"}, "k")
    assert req.headers["X-Sig"].startswith("HMAC ")
    expect = _hmac.new(b"k", b"GET\n/x\n", hashlib.sha1).hexdigest()
    assert req.headers["X-Sig"] == f"HMAC {expect}"


def test_hmac_end_to_end_via_call():
    op = dict(_OAUTH_OP, tool_name="acme__sign", security_scheme="hmac",
              http_method="POST", path_template="/sign",
              request_map={"path": [], "query": [], "header": [], "has_body": True},
              auth_config={"header": "X-Signature"})
    sender = _Sender([200])
    ex = DynamicHttpExecutor(db=object(), connections=_Conn(_Cred({}, "secret")),
                             http=sender, store=_Store(op))
    ex.call("acme__sign", {"body": {"x": 1}}, ctx=_Ctx())
    assert "X-Signature" in sender.calls[0][2]


class _LCred:
    def __init__(self, config, secret, connection_id):
        self.config, self.secret, self.connection_id = config, secret, connection_id


class _LConn:
    def __init__(self, cred):
        self._cred, self._db = cred, object()
        self.updates = []
    def resolve_or_none(self, provider, *, user_id=None, app_id=None):
        return self._cred
    def update(self, connection_id, *, fields=None, **kw):
        self.updates.append((connection_id, dict(fields or {})))


_LOGIN_OP = {
    "operation_id": "list", "tool_name": "fleet__list", "http_method": "GET",
    "path_template": "/things", "base_url": "https://api.fleet.example",
    "request_map": {"path": [], "query": [], "header": [], "has_body": False},
    "parameters_schema": {"type": "object", "properties": {}},
    "side_effect": "read", "requires_approval": False, "slug": "fleet",
    "auth_provider": "api_fleet", "security_scheme": "bearer", "auth_type": "login_grant",
    "auth_config": {
        "login_url": "https://api.fleet.example/auth/Login",
        "body_map": {"email": "{username}", "password": "{password}"},
        "token_path": "$.accessToken", "refresh_token_path": "$.refreshToken",
        "refresh": {"url": "https://api.fleet.example/auth/RefreshToken",
                    "body_map": {"accessToken": "{access_token}", "refreshToken": "{refresh_token}"}},
    },
    "egress_allow": [], "allow_private": False,
}


def _bundle(access, refresh):
    return json.dumps({"access_token": access, "refresh_token": refresh})


def test_login_exchange_returns_bundle_and_sends_password_once():
    from core.integrations.runtime import login_exchange
    ac = {"login_url": "https://api.fleet.example/auth/Login",
          "body_map": {"email": "{username}", "password": "{password}"},
          "token_path": "$.accessToken", "refresh_token_path": "$.refreshToken"}
    seen = {}
    def jp(url, body, headers=None):
        seen.update(body)
        return {"accessToken": "AAA", "refreshToken": "RRR"}
    b = login_exchange(jp, ac, "u@x.io", "pw")
    assert b == {"access_token": "AAA", "refresh_token": "RRR"}
    assert seen == {"email": "u@x.io", "password": "pw"}


def test_login_grant_refreshes_caches_and_rotates():
    sender = _Sender([200, 200])
    posts = []
    def jp(url, body, headers=None):
        posts.append((url, dict(body)))
        return {"accessToken": f"NEW{len(posts)}", "refreshToken": f"R{len(posts) + 1}"}
    conn = _LConn(_LCred({}, _bundle("old", "R1"), 7))
    ex = DynamicHttpExecutor(db=object(), connections=conn, http=sender,
                             json_post=jp, store=_Store(_LOGIN_OP))
    out = ex.call("fleet__list", {}, ctx=_Ctx())
    assert "HTTP 200" in out
    assert posts[0][0] == "https://api.fleet.example/auth/RefreshToken"
    assert posts[0][1] == {"accessToken": "old", "refreshToken": "R1"}
    assert sender.calls[0][2]["Authorization"] == "Bearer NEW1"
    ex.call("fleet__list", {}, ctx=_Ctx())
    assert len(posts) == 1
    assert conn.updates and json.loads(conn.updates[0][1]["password"])["refresh_token"] == "R2"


def test_login_grant_per_principal_isolation():
    posts = []
    def jp(url, body, headers=None):
        posts.append(body.get("refreshToken"))
        return {"accessToken": f"A{len(posts)}", "refreshToken": "R1"}
    conn = _LConn(_LCred({}, _bundle("old", "R1"), 7))
    ex = DynamicHttpExecutor(db=object(), connections=conn, http=_Sender([200, 200]),
                             json_post=jp, store=_Store(_LOGIN_OP))
    class _CtxA: user_id, app_id = "user_5", "4"
    class _CtxB: user_id, app_id = "user_9", "4"
    ex.call("fleet__list", {}, ctx=_CtxA())
    ex.call("fleet__list", {}, ctx=_CtxB())
    assert len(posts) == 2


def test_login_grant_needs_reauth_when_bundle_empty():
    ex = DynamicHttpExecutor(db=object(), connections=_LConn(_LCred({}, "{}", 7)),
                             http=_Sender([200]), json_post=lambda u, b, h=None: {}, store=_Store(_LOGIN_OP))
    out = ex.call("fleet__list", {}, ctx=_Ctx())
    assert "reconnect" in out.lower()


def test_default_headers_sent_on_call_and_refresh():
    """A static header (e.g. X-Tenant-Domain) configured on the integration must
    ride on every op call AND on the token-refresh POST."""
    sender = _Sender([200])
    seen = {}
    def jp(url, body, headers=None):
        seen["headers"] = dict(headers or {})
        return {"data": {"accessToken": "NEW", "refreshToken": "R2"}}
    ac = json.loads(json.dumps(_LOGIN_OP["auth_config"]))
    ac["refresh"]["token_path"] = ["$.data.accessToken"]
    ac["refresh"]["refresh_token_path"] = ["$.data.refreshToken"]
    op = dict(_LOGIN_OP, auth_config=ac,
              default_headers={"X-Tenant-Domain": "fleet.storeglance.com"})
    conn = _LConn(_LCred({}, _bundle("old", "R1"), 7))
    ex = DynamicHttpExecutor(db=object(), connections=conn, http=sender,
                             json_post=jp, store=_Store(op))
    ex.call("fleet__list", {}, ctx=_Ctx())
    assert sender.calls[0][2].get("X-Tenant-Domain") == "fleet.storeglance.com"
    assert seen["headers"].get("X-Tenant-Domain") == "fleet.storeglance.com"
