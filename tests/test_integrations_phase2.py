"""
Phase 2 regression — dynamic runtime (no real network/DB).

  1. principal_from_ctx mirrors the hand-coded tools.
  2. egress guard (SSRF): allowlist, private/metadata/http rules.
  3. request building: path subst, query, body, auth (header/bearer/query/basic).
  4. call() end-to-end via a fake HTTP sender (+ not-configured, missing param).
  5. normalize: JSON pretty, error status, truncation.
  6. IntegrationToolProvider.register_session_direct through the REAL registry
     (json_schema pass-through + kwargs callback + method policy) — incl. flag gate.
  7. effective_allowed_tools grant expansion (flag off = inert).

    python -m pytest tests/test_integrations_phase2.py -q
"""

import pytest

from core.integrations import runtime as rt
from core.integrations.runtime import (
    DynamicHttpExecutor, IntegrationToolProvider, principal_from_ctx,
    egress_check, EgressBlocked, _Req,
)
from tools.registry import ToolRegistry
from tools.tool_policy import READ


class _Ctx:
    def __init__(self, user_id="user_5", app_id="4"):
        self.user_id = user_id
        self.app_id = app_id


class _Cred:
    def __init__(self, config, secret):
        self.config = config
        self.secret = secret


class _Conn:
    def __init__(self, cred=None):
        self._cred = cred
        self._db = object()

    def resolve_or_none(self, provider, *, user_id=None, app_id=None):
        return self._cred


_GET_OP = {
    "operation_id": "getLead", "tool_name": "acme__getlead", "http_method": "GET",
    "path_template": "/leads/{id}", "base_url": "https://api.acme.example/v2",
    "request_map": {"path": ["id"], "query": ["expand"], "header": [], "has_body": False},
    "parameters_schema": {"type": "object", "properties": {
        "id": {"type": "integer"}, "expand": {"type": "string"}}, "required": ["id"]},
    "side_effect": "read", "requires_approval": False, "slug": "acme",
    "auth_provider": "api_acme", "security_scheme": "apiKey",
    "auth_config": {"in": "header", "name": "X-Api-Key"},
    "egress_allow": [], "allow_private": False,
}
_POST_OP = dict(_GET_OP, operation_id="createLead", tool_name="acme__createlead",
                http_method="POST", path_template="/leads", side_effect="write",
                requires_approval=True,
                request_map={"path": [], "query": [], "header": [], "has_body": True},
                parameters_schema={"type": "object", "properties": {
                    "body": {"type": "object"}}, "required": ["body"]})


class _Store:
    def __init__(self, ops):
        self.ops = {o["tool_name"]: o for o in ops}

    def get_op_for_execution(self, tool_name):
        return self.ops.get(tool_name)

    def catalog_for_principal(self, *, app_id="", persona_id="", embed_safe_only=False):
        return [dict(o) for o in self.ops.values()]


class _Sender:
    """Records the last request; returns a canned (status, headers, body)."""
    def __init__(self, status=200, headers=None, body=b'{"ok": true}'):
        self.status, self.headers, self.body = status, headers or {}, body
        self.last = None

    def __call__(self, method, url, req: _Req):
        self.last = (method, url, req)
        return self.status, self.headers, self.body


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    """Make the egress guard see a public IP for example hosts (no real DNS)."""
    def fake_getaddrinfo(host, *a, **k):
        ip = {"api.acme.example": "93.184.216.34",
              "evil.example": "10.0.0.5",
              "metadata.example": "169.254.169.254"}.get(host, "93.184.216.34")
        return [(2, 1, 6, "", (ip, 0))]
    monkeypatch.setattr(rt.socket, "getaddrinfo", fake_getaddrinfo)


def _exec(ops, cred=None, sender=None):
    sender = sender or _Sender()
    ex = DynamicHttpExecutor(db=object(), connections=_Conn(cred),
                             http=sender, store=_Store(ops))
    return ex, sender


def test_principal_from_ctx():
    assert principal_from_ctx(_Ctx("user_5", "4")) == (5, "4")
    assert principal_from_ctx(_Ctx("embed_abc", None)) == (None, None)
    assert principal_from_ctx(None) == (None, None)


def test_egress_allows_declared_host():
    ips = egress_check("https://api.acme.example/v2/leads",
                       base_url="https://api.acme.example/v2", egress_allow=[],
                       allow_private=False)
    assert ips == ["93.184.216.34"]


def test_egress_blocks_foreign_host():
    with pytest.raises(EgressBlocked):
        egress_check("https://evil.example/x", base_url="https://api.acme.example/v2",
                     egress_allow=[], allow_private=False)


def test_egress_blocks_private_unless_opted_in():
    with pytest.raises(EgressBlocked):
        egress_check("https://evil.example/x", base_url="https://evil.example",
                     egress_allow=[], allow_private=False)
    ips = egress_check("https://evil.example/x", base_url="https://evil.example",
                       egress_allow=[], allow_private=True)
    assert ips == ["10.0.0.5"]


def test_egress_metadata_never_allowed():
    with pytest.raises(EgressBlocked):
        egress_check("https://metadata.example/latest", base_url="https://metadata.example",
                     egress_allow=[], allow_private=True)


def test_egress_http_requires_private():
    with pytest.raises(EgressBlocked):
        egress_check("http://api.acme.example/x", base_url="http://api.acme.example",
                     egress_allow=[], allow_private=False)


def test_build_get_request_path_query_auth():
    ex, sender = _exec([_GET_OP], cred=_Cred({}, "SECRET"))
    out = ex.call("acme__getlead", {"id": 7, "expand": "owner"}, ctx=_Ctx())
    method, url, req = sender.last
    assert method == "GET"
    assert url == "https://api.acme.example/v2/leads/7"
    assert req.params == {"expand": "owner"}
    assert req.headers["X-Api-Key"] == "SECRET"
    assert "HTTP 200" in out


def test_bearer_and_query_and_basic_auth():
    op = dict(_GET_OP, auth_config={"in": "header", "name": "Authorization", "fmt": "Bearer {token}"})
    ex, sender = _exec([op], cred=_Cred({}, "TOK"))
    ex.call("acme__getlead", {"id": 1}, ctx=_Ctx())
    assert sender.last[2].headers["Authorization"] == "Bearer TOK"

    op = dict(_GET_OP, auth_config={"in": "query", "name": "api_key"})
    ex, sender = _exec([op], cred=_Cred({}, "K"))
    ex.call("acme__getlead", {"id": 1}, ctx=_Ctx())
    assert sender.last[2].params["api_key"] == "K"

    op = dict(_GET_OP, auth_config={"in": "basic"})
    ex, sender = _exec([op], cred=_Cred({"username": "u"}, "p"))
    ex.call("acme__getlead", {"id": 1}, ctx=_Ctx())
    assert sender.last[2].basic_auth == ("u", "p")


def test_post_body():
    ex, sender = _exec([_POST_OP], cred=_Cred({}, "K"))
    ex.call("acme__createlead", {"body": {"name": "Ada"}}, ctx=_Ctx())
    assert sender.last[2].json_body == {"name": "Ada"}
    assert sender.last[0] == "POST"


def test_missing_required_path_param():
    ex, _ = _exec([_GET_OP], cred=_Cred({}, "K"))
    out = ex.call("acme__getlead", {"expand": "owner"}, ctx=_Ctx())
    assert "missing required parameter" in out.lower()


def test_unknown_tool():
    ex, _ = _exec([_GET_OP], cred=_Cred({}, "K"))
    assert "not available" in ex.call("acme__nope", {}, ctx=_Ctx()).lower()


def test_not_configured_when_no_credential():
    ex, _ = _exec([_GET_OP], cred=None)
    out = ex.call("acme__getlead", {"id": 1}, ctx=_Ctx())
    assert "not connected" in out.lower()


def test_normalize_pretty_and_error_and_truncate():
    ex, _ = _exec([_GET_OP], cred=_Cred({}, "K"),
                  sender=_Sender(status=404, body=b'{"error":"nope"}'))
    out = ex.call("acme__getlead", {"id": 1}, ctx=_Ctx())
    assert "HTTP 404 (error)" in out and "nope" in out

    big = b'{"x":"' + b"a" * (rt.MAX_RESP_BYTES + 50) + b'"}'
    ex, _ = _exec([_GET_OP], cred=_Cred({}, "K"), sender=_Sender(body=big))
    out = ex.call("acme__getlead", {"id": 1}, ctx=_Ctx())
    assert "truncated" in out.lower()


def test_register_session_direct_and_invoke(monkeypatch):
    from core.config import app_config
    monkeypatch.setattr(app_config, "get_bool", lambda k, d=False: True)

    sender = _Sender()
    ex = DynamicHttpExecutor(db=object(), connections=_Conn(_Cred({}, "K")),
                             http=sender, store=_Store([_GET_OP, _POST_OP]))
    prov = IntegrationToolProvider(connections=_Conn(_Cred({}, "K")),
                                   embeddings=None, executor=ex, store=ex._store)
    reg = ToolRegistry()
    n = prov.register_session_direct(reg, _Ctx())
    assert n == 2
    assert reg.get_policy("acme__getlead").side_effect == READ
    assert reg.get_policy("acme__getlead").requires_approval is False
    assert reg.get_policy("acme__createlead").requires_approval is True
    fn = next(t["function"] for t in reg.get_openai_tools()
              if t["function"]["name"] == "acme__createlead")
    assert fn["parameters"]["properties"]["body"]["type"] == "object"
    out = reg.execute("acme__getlead", {"id": 9})
    assert "HTTP 200" in out
    assert sender.last[1].endswith("/leads/9")


def test_register_noop_when_flag_off(monkeypatch):
    from core.config import app_config
    monkeypatch.setattr(app_config, "get_bool", lambda k, d=False: False)
    ex = DynamicHttpExecutor(db=object(), connections=_Conn(), http=_Sender(),
                             store=_Store([_GET_OP]))
    prov = IntegrationToolProvider(_Conn(), None, ex, store=ex._store)
    reg = ToolRegistry()
    assert prov.register_session_direct(reg, _Ctx()) == 0
    assert reg.get_openai_tools() == []


def test_effective_allowed_tools(monkeypatch):
    from core.integrations import grants
    from core.integrations.store import IntegrationStore
    monkeypatch.setattr(grants, "_enabled", lambda: True)
    monkeypatch.setattr(IntegrationStore, "catalog_for_principal",
                        lambda self, **k: [{"tool_name": "acme__getlead"}])
    out = grants.effective_allowed_tools(object(), ["brain_ask"], app_id="4")
    assert out == {"brain_ask", "acme__getlead"}


def test_effective_allowed_tools_inert_when_off(monkeypatch):
    from core.integrations import grants
    monkeypatch.setattr(grants, "_enabled", lambda: False)
    assert grants.effective_allowed_tools(object(), ["brain_ask"], app_id="4") == {"brain_ask"}
