"""
Phase 6 regression — MCP source + response→brain / transform. No network.

  1. MCP: fetch_tools, normalize_mcp, call_envelope, parse_mcp_result.
  2. Ingest via MCP (fake JSON-RPC transport) → ops with spec_kind='mcp'.
  3. Executor MCP call end-to-end (envelope out, result unwrapped).
  4. response_transform: JSONPath subtree / trim; brain sink (read-only, pointer).

    python -m pytest tests/test_integrations_phase6.py -q
"""

import json
import pytest

from core.integrations import runtime as rt, mcp
from core.integrations.runtime import DynamicHttpExecutor
from core.integrations.ingest import SpecIngestor
from core.integrations.openapi import SpecError


_TOOLS = [
    {"name": "search", "description": "Search docs",
     "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}},
    {"name": "create", "description": "Create a doc",
     "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}}},
]


def test_fetch_tools_via_jsonrpc():
    def post(url, body, headers):
        assert body["method"] == "tools/list"
        return {"jsonrpc": "2.0", "id": 1, "result": {"tools": _TOOLS}}
    tools = mcp.fetch_tools("https://mcp.acme/rpc", http_post=post)
    assert [t["name"] for t in tools] == ["search", "create"]


def test_fetch_tools_errors():
    with pytest.raises(SpecError):
        mcp.fetch_tools("https://x", http_post=lambda u, b, h: {"error": {"code": -1}})
    with pytest.raises(SpecError):
        mcp.fetch_tools("https://x", http_post=lambda u, b, h: {"result": {}})


def test_normalize_mcp():
    integ, req_maps = mcp.normalize_mcp(_TOOLS, "acme_mcp", "https://mcp.acme/rpc")
    assert integ.spec_kind == "mcp" and integ.auth_provider is None
    ops = {o.operation_id: o for o in integ.operations}
    assert ops["search"].tool_name == "acme_mcp__search"
    assert ops["search"].http_method == "POST"
    assert ops["search"].side_effect == "write" and ops["search"].requires_approval is True
    assert ops["search"].parameters_schema["properties"]["q"]["type"] == "string"
    assert req_maps["search"]["mcp"] is True


def test_call_envelope_and_parse():
    env = mcp.call_envelope("search", {"q": "hi"})
    assert env["method"] == "tools/call"
    assert env["params"] == {"name": "search", "arguments": {"q": "hi"}}
    assert mcp.parse_mcp_result(
        {"result": {"content": [{"type": "text", "text": "found 3"}]}}) == "found 3"
    assert "tool error" in mcp.parse_mcp_result(
        {"result": {"isError": True, "content": [{"type": "text", "text": "boom"}]}}).lower()
    assert mcp.parse_mcp_result({"error": {"message": "nope"}}).startswith("MCP error")


class _Store:
    def __init__(self): self.integ = {}; self.ops = {}; self._n = 1
    def get_integration_by_slug(self, slug): return self.integ.get(slug)
    def upsert_integration(self, integ, *, spec_raw, spec_hash, created_by):
        iid = self._n; self._n += 1
        self.integ[integ.slug] = {"integration_id": iid, "spec_hash": spec_hash,
                                  "spec_kind": integ.spec_kind}
        return iid
    def replace_operations(self, iid, ops, req_maps):
        self.ops[iid] = list(ops)
        return {"added": len(ops), "updated": 0, "deprecated": 0}
    def set_embeddings(self, iid, vectors): return 0
    def list_operations(self, iid): return self.ops.get(iid, [])


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    monkeypatch.setattr(rt.socket, "getaddrinfo",
                        lambda host, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    from core.integrations import ingest as ing_mod
    monkeypatch.setattr(ing_mod.socket, "getaddrinfo",
                        lambda host, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])


def test_ingest_mcp_creates_ops():
    store = _Store()
    ing = SpecIngestor(db=None, connections=object(), embeddings=None, store=store,
                       mcp_post=lambda u, b, h: {"result": {"tools": _TOOLS}})
    rep = ing.ingest("mcp://mcp.acme/rpc", slug="acme_mcp")
    assert rep.integration_id == 1 and rep.operation_count == 2
    assert store.integ["acme_mcp"]["spec_kind"] == "mcp"
    rep2 = ing.ingest("mcp://mcp.acme/rpc", slug="acme_mcp")
    assert rep2.unchanged is True


_MCP_OP = {
    "operation_id": "search", "tool_name": "acme__search", "http_method": "POST",
    "path_template": "", "base_url": "https://mcp.acme/rpc", "spec_kind": "mcp",
    "request_map": {"mcp": True}, "parameters_schema": {"type": "object"},
    "side_effect": "write", "requires_approval": True, "slug": "acme",
    "auth_provider": None, "security_scheme": None, "auth_config": {},
    "egress_allow": [], "allow_private": False,
}


class _Conn:
    _db = object()
    def resolve_or_none(self, *a, **k): return None


def test_mcp_call_end_to_end():
    captured = {}
    def sender(method, url, req):
        captured["body"] = req.json_body
        return 200, {}, json.dumps(
            {"result": {"content": [{"type": "text", "text": "3 hits"}]}}).encode()
    ex = DynamicHttpExecutor(db=object(), connections=_Conn(), http=sender,
                             store=type("S", (), {"get_op_for_execution":
                                                  lambda s, n: _MCP_OP})())
    out = ex.call("acme__search", {"q": "hello"}, ctx=None)
    assert out == "3 hits"
    assert captured["body"]["method"] == "tools/call"
    assert captured["body"]["params"]["arguments"] == {"q": "hello"}


def _exec_with_transform(op, brain=None, body=b'{}'):
    sender = lambda m, u, r: (200, {}, body)
    return DynamicHttpExecutor(db=object(), connections=_Conn(), http=sender,
                               brain_ingest=brain,
                               store=type("S", (), {"get_op_for_execution":
                                                    lambda s, n: op})())


def test_apply_transform_path_and_trim():
    ex = DynamicHttpExecutor(db=object(), connections=_Conn(), http=lambda *a: None,
                             store=object())
    body = json.dumps({"data": {"items": [1, 2, 3, 4, 5]}})
    out = ex._apply_transform(body, {"path": "data.items", "max_items": 2})
    assert json.loads(out) == [1, 2]
    assert ex._apply_transform(body, {"path": "nope.x"}) == "null"
    assert ex._apply_transform("not json", {"max_chars": 3}) == "not"


def test_brain_sink_for_read_returns_pointer():
    op = dict(_MCP_OP, tool_name="acme__list", spec_kind="openapi", side_effect="read",
              http_method="GET", request_map={"path": [], "query": [], "header": [], "has_body": False},
              response_transform={"sink": "brain"})
    seen = {}
    def brain(title, content): seen["t"] = title; seen["c"] = content; return "doc42"
    ex = _exec_with_transform(op, brain=brain, body=b'{"big":"payload"}')
    out = ex.call("acme__list", {}, ctx=None)
    assert "doc42" in out and "knowledge base" in out.lower()
    assert seen["t"] == "acme:search" and "payload" in seen["c"]


def test_brain_sink_skipped_for_write():
    op = dict(_MCP_OP, tool_name="acme__create", spec_kind="openapi", side_effect="write",
              http_method="POST", request_map={"path": [], "query": [], "header": [], "has_body": False},
              response_transform={"sink": "brain"}, auth_provider=None)
    called = {"n": 0}
    def brain(t, c): called["n"] += 1; return "x"
    ex = _exec_with_transform(op, brain=brain, body=b'{"ok":1}')
    out = ex.call("acme__create", {}, ctx=None)
    assert called["n"] == 0
    assert "HTTP 200" in out
