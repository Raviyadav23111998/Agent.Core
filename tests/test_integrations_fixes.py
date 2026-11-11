"""
Regression tests for the senior-review hardening pass (rev-3 fixes):
  - DELETE per-op approval opt-out actually relaxes the gate
  - tool_name truncation gets a disambiguating hash tail
  - egress guard blocks IPv6 + IPv4-mapped cloud-metadata
  - DNS pinning context maps host → validated IP for the calling thread
  - redirect handling resolves a RELATIVE Location and re-guards each hop
  - generic fixed-window rate limiter
"""

import pytest

from core.integrations import runtime as rt
from core.integrations.runtime import egress_check, EgressBlocked, _Req, _pin_dns
from core.integrations.model import policy_for_method, tool_name_for, _MAX_TOOL_NAME
from tools.tool_policy import READ, WRITE, IRREVERSIBLE


def test_delete_default_requires_approval():
    pol = policy_for_method("DELETE")
    assert pol.side_effect == IRREVERSIBLE
    assert pol.requires_approval is True


def test_delete_optout_relaxes_to_write():
    pol = policy_for_method("DELETE", requires_approval=False)
    assert pol.side_effect == WRITE
    assert pol.requires_approval is False


def test_get_is_read_and_post_override():
    assert policy_for_method("GET").side_effect == READ
    assert policy_for_method("POST").requires_approval is True
    assert policy_for_method("POST", requires_approval=False).side_effect == WRITE


def test_long_tool_names_do_not_collide():
    a = "a" * 200
    b = "a" * 199 + "b"
    na, nb = tool_name_for("slug", a), tool_name_for("slug", b)
    assert len(na) <= _MAX_TOOL_NAME and len(nb) <= _MAX_TOOL_NAME
    assert na != nb


def _patch_resolve(monkeypatch, mapping):
    def fake(host, *a, **k):
        ip = mapping.get(host, "93.184.216.34")
        fam = 10 if ":" in ip else 2
        sa = (ip, 0, 0, 0) if fam == 10 else (ip, 0)
        return [(fam, 1, 6, "", sa)]
    monkeypatch.setattr(rt.socket, "getaddrinfo", fake)


def test_egress_blocks_ipv6_metadata_even_private(monkeypatch):
    _patch_resolve(monkeypatch, {"imds.example": "fd00:ec2::254"})
    with pytest.raises(EgressBlocked):
        egress_check("https://imds.example/x", base_url="https://imds.example",
                     egress_allow=[], allow_private=True)


def test_egress_blocks_ipv4_mapped_metadata(monkeypatch):
    _patch_resolve(monkeypatch, {"m.example": "::ffff:169.254.169.254"})
    with pytest.raises(EgressBlocked):
        egress_check("https://m.example/x", base_url="https://m.example",
                     egress_allow=[], allow_private=True)


def test_pin_dns_maps_host_to_ip():
    host, ip = "api.acme.example", "93.184.216.34"
    with _pin_dns(host, ip):
        infos = rt.socket.getaddrinfo(host, None)
    assert any(ai[4][0] == ip for ai in infos)


def test_pin_dns_noop_without_ip():
    with _pin_dns("api.acme.example", None):
        pass


def test_send_follows_relative_redirect(monkeypatch):
    _patch_resolve(monkeypatch, {"api.acme.example": "93.184.216.34"})
    calls = []

    def sender(method, url, req):
        calls.append(url)
        if len(calls) == 1:
            return 302, {"location": "/v2/next"}, b""
        return 200, {}, b'{"ok":true}'

    ex = rt.DynamicHttpExecutor(db=object(), connections=object(),
                                http=sender, store=object())
    op = {"base_url": "https://api.acme.example", "egress_allow": [],
          "allow_private": False}
    status, body = ex._send(op, _Req(method="GET", url="https://api.acme.example/v2/start"))
    assert status == 200 and b"ok" in body
    assert calls[1] == "https://api.acme.example/v2/next"


def test_check_rate_allows_then_blocks():
    from core.rate_limit import check_rate
    bucket = "test:unique-bucket-xyz"
    assert check_rate(bucket, 2, 60)[0] is True
    assert check_rate(bucket, 2, 60)[0] is True
    ok, retry = check_rate(bucket, 2, 60)
    assert ok is False and retry >= 0


def test_check_rate_disabled_when_zero():
    from core.rate_limit import check_rate
    for _ in range(5):
        assert check_rate("test:zero", 0, 60)[0] is True


def test_keyword_pick_matches_dynamic_op_tokens():
    avail = {"crm__create_lead", "crm__delete_invoice", "view_file", "brain_search"}
    picked = rt._keyword_pick("please create a new lead for acme", avail, k=8)
    assert picked == {"crm__create_lead"}


def test_keyword_pick_ignores_builtins_and_empty():
    avail = {"crm__create_lead", "view_file"}
    assert rt._keyword_pick("", avail, k=8) == set()
    assert rt._keyword_pick("view the file", avail, k=8) == set()


def test_make_picker_uses_keyword_fallback_when_no_embeddings():
    prov = rt.IntegrationToolProvider(
        connections=object(), embeddings=None,
        executor=rt.DynamicHttpExecutor(db=object(), connections=object(),
                                        http=lambda *a: (200, {}, b""), store=object()))
    pick = prov.make_picker(_make_ctx("user_1", "4"))
    got = pick("delete the invoice now", {"crm__delete_invoice", "crm__create_lead"})
    assert got == {"crm__delete_invoice"}


def _make_ctx(uid, app):
    import types
    return types.SimpleNamespace(user_id=uid, app_id=app)


def test_extract_base_url_resolves_relative_server_against_source():
    from core.integrations import openapi as oa
    spec = {"openapi": "3.0.0", "servers": [{"url": "/api/v3"}], "paths": {}}
    assert oa.extract_base_url(
        spec, source_url="https://petstore3.swagger.io/api/v3/openapi.json"
    ) == "https://petstore3.swagger.io/api/v3"


def test_extract_base_url_no_server_falls_back_to_source_origin():
    from core.integrations import openapi as oa
    spec = {"openapi": "3.0.0", "paths": {}}
    assert oa.extract_base_url(
        spec, source_url="https://api.example.com/specs/openapi.json"
    ) == "https://api.example.com"


def test_extract_base_url_relative_without_source_still_errors():
    from core.integrations import openapi as oa
    with pytest.raises(oa.SpecError):
        oa.extract_base_url({"servers": [{"url": "/api/v3"}]})


def test_normalize_threads_source_url_for_relative_server():
    from core.integrations import openapi as oa
    spec = {"openapi": "3.0.0", "servers": [{"url": "/v1"}],
            "paths": {"/ping": {"get": {"operationId": "ping", "responses": {}}}}}
    integ, _ = oa.normalize(spec, "demo", source_url="https://demo.test/openapi.json")
    assert integ.base_url == "https://demo.test/v1"


def test_path_placeholder_reconciled_when_spec_omits_param():
    """A spec that puts {driverId} in the path but declares no parameter must
    still expose driverId as a required arg + map it for substitution."""
    from core.integrations import openapi as oa
    spec = {"openapi": "3.0.0", "servers": [{"url": "https://x.test"}],
            "paths": {"/driver/{driverId}/trips": {
                "get": {"operationId": "DriverTrips", "responses": {"200": {"description": "ok"}}}}}}
    integ, rmaps = oa.normalize(spec, "f")
    op = integ.operations[0]
    assert "driverId" in op.parameters_schema["properties"]
    assert "driverId" in op.parameters_schema.get("required", [])
    assert "driverId" in rmaps[op.operation_id]["path"]


def test_executor_rejects_unfilled_placeholder():
    """Defense in depth: a leftover {id} in the path must raise (→ clean error),
    never be sent as a literal."""
    ex = rt.DynamicHttpExecutor(db=object(), connections=object(),
                                http=lambda *a: (200, {}, b""), store=object())
    op = {"spec_kind": "openapi", "http_method": "GET",
          "base_url": "https://x.test", "path_template": "/driver/{id}",
          "request_map": {"path": [], "query": [], "header": [], "has_body": False}}
    import pytest
    with pytest.raises(KeyError):
        ex._build_request(op, {})


class _RecEmb:
    available = True
    def __init__(self): self.calls = []
    def embed(self, texts):
        self.calls.append(len(texts))
        return [[float(len(t))] + [0.0] * 767 for t in texts]


class _RecStore:
    def __init__(self, missing=()):
        self.saved = {}
        self._missing = list(missing)
    def set_embeddings(self, iid, vectors):
        self.saved.update(vectors)
        return len([v for v in vectors.values() if v])
    def operations_missing_embeddings(self, iid):
        return [r for r in self._missing if r["operation_id"] not in self.saved]


def test_embed_items_chunks_and_persists_partial():
    from core.integrations.ingest import SpecIngestor, IngestReport
    emb, store = _RecEmb(), _RecStore()
    ing = SpecIngestor(db=None, connections=object(), embeddings=emb, store=store)
    ing.EMBED_BATCH = 10
    items = [(f"op{i}", f"text {i}") for i in range(25)]
    rep = IngestReport(slug="x")
    n = ing._embed_items(1, items, 25, rep, budget_s=60)
    assert n == 25
    assert emb.calls == [10, 10, 5]
    assert len(store.saved) == 25


def test_backfill_only_embeds_missing():
    from core.integrations.ingest import SpecIngestor
    missing = [{"operation_id": f"op{i}", "http_method": "GET",
                "path_template": f"/x{i}", "summary": "", "description": ""} for i in range(3)]
    emb, store = _RecEmb(), _RecStore(missing=missing)
    ing = SpecIngestor(db=None, connections=object(), embeddings=emb, store=store)
    res = ing.backfill_embeddings(1)
    assert res["embedded"] == 3
    assert res["remaining"] == 0


def test_embed_text_combines_signals_when_summary_blank():
    from core.integrations.ingest import _embed_text
    t = _embed_text("GET", "/driver/{driverId}/trips", "Driver_get_trips", "", "")
    assert "GET /driver/{driverId}/trips" in t
    assert "Driver get trips" in t


def test_auth_resolve_by_type_and_scheme():
    from core.integrations import auth
    assert auth.resolve("oauth2_client_credentials").is_token_based() is True
    assert auth.resolve("api_key").is_token_based() is False
    assert auth.resolve("hmac").is_token_based() is False
    assert auth.resolve(None, "oauth2").is_token_based() is True
    assert auth.resolve(None, "apiKey").is_token_based() is False
    assert auth.resolve(None, "bearer").is_token_based() is False
    assert auth.resolve(None, "hmac").is_token_based() is False
    assert auth.resolve(None, None).is_token_based() is False
    assert auth.resolve("nonsense", "nonsense").is_token_based() is False


def test_static_strategy_applies_apikey_header_via_dispatch():
    ex = rt.DynamicHttpExecutor(db=object(), connections=object(),
                                http=lambda *a: (200, {}, b""), store=object())
    op = {"auth_provider": "api_acme", "security_scheme": "apiKey",
          "auth_config": {"in": "header", "name": "X-API-Key"}}
    req = rt._Req(method="GET", url="https://x.test")
    ex._apply_auth_for(op, req, {}, "sekret")
    assert req.headers.get("X-API-Key") == "sekret"


def test_dispatch_honors_auth_type_over_scheme():
    from core.integrations import auth
    op_type = {"auth_type": "bearer_static", "security_scheme": None}
    assert auth.resolve(op_type["auth_type"], op_type["security_scheme"]).is_token_based() is False


def _bearer_login_spec():
    return {
        "openapi": "3.0.0", "servers": [{"url": "https://api.fleet.test/api"}],
        "components": {
            "securitySchemes": {"Bearer": {"type": "http", "scheme": "bearer"}},
            "schemas": {
                "LoginRequest": {"type": "object", "required": ["email", "password"],
                                 "properties": {"email": {"type": "string"},
                                                "password": {"type": "string"}}},
                "RefreshTokenRequest": {"type": "object",
                                        "properties": {"accessToken": {"type": "string"},
                                                       "refreshToken": {"type": "string"}}},
            },
        },
        "paths": {
            "/auth/Login": {"post": {"operationId": "Auth_login",
                "requestBody": {"content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/LoginRequest"}}}},
                "responses": {"200": {"description": "ok"}}}},
            "/auth/RefreshToken": {"post": {"operationId": "Auth_refresh",
                "requestBody": {"content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/RefreshTokenRequest"}}}},
                "responses": {"200": {"description": "ok"}}}},
            "/ledger/Refresh": {"post": {"operationId": "Ledger_refresh",
                "requestBody": {"content": {"application/json": {"schema": {"type": "object",
                    "properties": {"refreshToken": {"type": "string"}}}}}},
                "responses": {"200": {"description": "ok"}}}},
            "/driver/List": {"get": {"operationId": "Driver_list", "responses": {"200": {"description": "ok"}}}},
        },
    }


def test_discover_auth_apikey_bearer_basic_oauth():
    from core.integrations import openapi as oa
    ak = oa.discover_auth({"components": {"securitySchemes": {"k": {"type": "apiKey", "in": "header", "name": "X-Key"}}}}, "s")
    assert ak["auth_type"] == "api_key" and ak["fields"][0]["key"] == "api_key"
    bs = oa.discover_auth({"components": {"securitySchemes": {"b": {"type": "http", "scheme": "basic"}}}}, "s")
    assert bs["auth_type"] == "basic" and {f["key"] for f in bs["fields"]} == {"username", "password"}
    cc = oa.discover_auth({"components": {"securitySchemes": {"o": {"type": "oauth2",
        "flows": {"clientCredentials": {"tokenUrl": "https://t/x", "scopes": {}}}}}}}, "s")
    assert cc["auth_type"] == "oauth2_client_credentials" and cc["scope_default"] == "org"
    pw = oa.discover_auth({"components": {"securitySchemes": {"o": {"type": "oauth2",
        "flows": {"password": {"tokenUrl": "https://t/x", "scopes": {}}}}}}}, "s")
    assert pw["auth_type"] == "oauth2_password" and pw["scope_default"] == "user"


def test_find_login_op_detects_fleet_style_login():
    from core.integrations import openapi as oa
    spec = oa.coerce_to_openapi3(_bearer_login_spec())
    login = oa.find_login_op(spec, base_url="https://api.fleet.test/api")
    assert login is not None
    assert login["login_url"] == "https://api.fleet.test/api/auth/Login"
    assert login["body_map"] == {"email": "{username}", "password": "{password}"}
    assert "$.accessToken" in login["token_path"] and "$.data.accessToken" in login["token_path"]
    assert login["refresh"]["url"] == "https://api.fleet.test/api/auth/RefreshToken"
    assert login["refresh"]["body_map"] == {"accessToken": "{access_token}",
                                            "refreshToken": "{refresh_token}"}


def test_extract_token_handles_wrapped_envelope():
    from core.integrations.runtime import _extract_token
    cands = ["$.accessToken", "$.data.accessToken", "$.token"]
    assert _extract_token({"data": {"accessToken": "ZZZ"}}, cands) == "ZZZ"
    assert _extract_token({"accessToken": "TOP"}, cands) == "TOP"
    assert _extract_token({"nope": 1}, cands) is None


def test_bearer_plus_login_op_upgrades_to_login_grant():
    from core.integrations import openapi as oa
    d = oa.discover_auth(oa.coerce_to_openapi3(_bearer_login_spec()), "fleet",
                         base_url="https://api.fleet.test/api")
    assert d["auth_type"] == "login_grant"
    assert d["scope_default"] == "user"
    assert {f["key"] for f in d["fields"]} == {"username", "password"}
    assert d["auth_config"]["login_url"].endswith("/auth/Login")
    assert any("login" in w.lower() for w in d["warnings"])


def test_plain_bearer_without_login_stays_static():
    from core.integrations import openapi as oa
    d = oa.discover_auth({"components": {"securitySchemes": {"b": {"type": "http", "scheme": "bearer"}}},
                          "paths": {}}, "s")
    assert d["auth_type"] == "bearer_static"


def test_extract_security_backcompat_tuple():
    from core.integrations import openapi as oa
    scheme, cfg, fields = oa.extract_security(
        {"components": {"securitySchemes": {"k": {"type": "apiKey", "in": "header", "name": "X"}}}}, "s")
    assert scheme == "apiKey" and isinstance(cfg, dict) and isinstance(fields, list)


def test_global_header_promoted_to_default_headers():
    """A header on (nearly) every op — e.g. x-tenant-domain — is promoted to the
    integration's default_headers and removed from per-op args (the LLM can't fill
    a tenant header). A genuinely per-op param stays exposed."""
    from core.integrations import openapi as oa
    P = {"$ref": "#/components/parameters/x-tenant-domain"}
    spec = {"openapi": "3.0.0", "servers": [{"url": "https://x.test"}],
            "components": {"parameters": {"x-tenant-domain": {
                "in": "header", "name": "x-tenant-domain", "required": True}}},
            "paths": {
                "/a": {"get": {"operationId": "A", "responses": {"200": {"description": "ok"}},
                    "parameters": [P, {"name": "q", "in": "query", "schema": {"type": "string"}}]}},
                "/b": {"get": {"operationId": "B", "responses": {"200": {"description": "ok"}},
                    "parameters": [P]}},
            }}
    integ, rmaps = oa.normalize(spec, "f")
    assert "x-tenant-domain" in integ.default_headers
    for op in integ.operations:
        assert "x-tenant-domain" not in (op.parameters_schema.get("properties") or {})
        assert "x-tenant-domain" not in rmaps[op.operation_id]["header"]
    a = next(o for o in integ.operations if o.operation_id == "A")
    assert "q" in a.parameters_schema["properties"]
