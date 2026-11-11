"""
Phase 1 regression — spec ingestion (no DB/network).

  1. OpenAPI 3.x normalize: $ref inlining, param+body merge, request_map.
  2. method → side_effect/approval; tool_name namespacing.
  3. security scheme → ag_cred provider fields.
  4. Swagger 2.0 coercion → same normalized shape.
  5. Ingest orchestration (fake store/conn/embeddings): persist, diff,
     unchanged short-circuit, auth materialization, embedding.
  6. SSRF host validation (private blocked unless allow_private; metadata never).

    python -m pytest tests/test_integrations_phase1.py -q
"""

import pytest

from core.integrations import openapi
from core.integrations.ingest import SpecIngestor
from tools.tool_policy import READ, IRREVERSIBLE


PETSTORE = {
    "openapi": "3.0.0",
    "info": {"title": "Acme CRM"},
    "servers": [{"url": "https://api.acme.example/v2"}],
    "paths": {
        "/leads/{id}": {
            "get": {
                "operationId": "getLead",
                "parameters": [
                    {"name": "id", "in": "path", "required": True,
                     "schema": {"type": "integer"}},
                    {"name": "expand", "in": "query",
                     "schema": {"type": "string", "enum": ["owner", "tasks"]}},
                ],
            },
            "delete": {"operationId": "deleteLead",
                       "parameters": [{"name": "id", "in": "path", "required": True,
                                       "schema": {"type": "integer"}}]},
        },
        "/leads": {
            "post": {
                "operationId": "createLead",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {
                        "schema": {"$ref": "#/components/schemas/Lead"}}}},
            },
        },
    },
    "components": {
        "securitySchemes": {"key": {"type": "apiKey", "in": "header", "name": "X-Api-Key"}},
        "schemas": {
            "Lead": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "owner": {"$ref": "#/components/schemas/User"},
                },
            },
            "User": {"type": "object", "properties": {"email": {"type": "string"}}},
        },
    },
}


def _ops(integ):
    return {op.operation_id: op for op in integ.operations}


def test_normalize_operations_and_refs():
    integ, req_maps = openapi.normalize(PETSTORE, "acme_crm")
    ops = _ops(integ)
    assert set(ops) == {"getLead", "deleteLead", "createLead"}

    get = ops["getLead"]
    assert get.tool_name == "acme_crm__getlead"
    assert get.side_effect == READ and get.requires_approval is False
    props = get.parameters_schema["properties"]
    assert props["id"]["type"] == "integer"
    assert props["expand"]["enum"] == ["owner", "tasks"]
    assert get.parameters_schema["required"] == ["id"]
    assert req_maps["getLead"] == {"path": ["id"], "query": ["expand"],
                                   "header": [], "has_body": False}

    post = ops["createLead"]
    body = post.parameters_schema["properties"]["body"]
    assert body["properties"]["name"]["type"] == "string"
    assert body["properties"]["owner"]["properties"]["email"]["type"] == "string"
    assert "body" in post.parameters_schema["required"]
    assert req_maps["createLead"]["has_body"] is True

    dele = ops["deleteLead"]
    assert dele.side_effect == IRREVERSIBLE and dele.requires_approval is True


def test_security_to_provider_fields():
    scheme, cfg, fields = openapi.extract_security(PETSTORE, "acme_crm")
    assert scheme == "apiKey"
    assert cfg == {"in": "header", "name": "X-Api-Key"}
    assert fields[0]["key"] == "api_key" and fields[0]["secret"] is True
    assert fields[0]["scope"] == "identity"


def test_no_servers_raises():
    with pytest.raises(openapi.SpecError):
        openapi.normalize({"openapi": "3.0.0", "paths": {}}, "x")


def test_no_operations_raises():
    spec = dict(PETSTORE, paths={})
    with pytest.raises(openapi.SpecError):
        openapi.normalize(spec, "acme_crm")


def test_duplicate_operation_ids_are_made_unique():
    """A non-compliant spec reusing an operationId must NOT lose an endpoint to
    the (integration_id, operation_id) upsert — every op gets a unique id/tool."""
    spec = {
        "openapi": "3.0.0", "info": {"title": "Dup"},
        "servers": [{"url": "https://api.dup.example"}],
        "paths": {
            "/a": {"get": {"operationId": "list"}},
            "/b": {"get": {"operationId": "list"}},
            "/c": {"get": {"operationId": "list"}},
        },
    }
    integ, req_maps = openapi.normalize(spec, "dup")
    op_ids = [o.operation_id for o in integ.operations]
    tool_names = [o.tool_name for o in integ.operations]
    assert len(integ.operations) == 3
    assert len(set(op_ids)) == 3
    assert len(set(tool_names)) == 3
    assert set(req_maps.keys()) == set(op_ids)


SWAGGER2 = {
    "swagger": "2.0",
    "host": "api.acme.example",
    "basePath": "/v1",
    "schemes": ["https"],
    "securityDefinitions": {"k": {"type": "apiKey", "in": "header", "name": "X-Token"}},
    "paths": {
        "/things": {
            "post": {
                "operationId": "createThing",
                "parameters": [
                    {"name": "body", "in": "body", "required": True,
                     "schema": {"$ref": "#/definitions/Thing"}},
                    {"name": "dry", "in": "query", "type": "boolean"},
                ],
            },
        },
    },
    "definitions": {"Thing": {"type": "object",
                              "properties": {"label": {"type": "string"}}}},
}


def test_swagger2_coercion():
    assert openapi.detect_kind(SWAGGER2) == "swagger"
    integ, req_maps = openapi.normalize(SWAGGER2, "acme")
    assert integ.base_url == "https://api.acme.example/v1"
    assert integ.spec_kind == "swagger"
    op = _ops(integ)["createThing"]
    assert op.parameters_schema["properties"]["body"]["properties"]["label"]["type"] == "string"
    assert op.parameters_schema["properties"]["dry"]["type"] == "boolean"
    assert req_maps["createThing"] == {"path": [], "query": ["dry"],
                                       "header": [], "has_body": True}
    scheme, cfg, _ = openapi.extract_security(openapi.coerce_to_openapi3(SWAGGER2), "acme")
    assert scheme == "apiKey" and cfg["name"] == "X-Token"


class _FakeStore:
    def __init__(self):
        self.integrations = {}
        self.ops = {}
        self.embeddings = {}
        self._next = 1

    def get_integration_by_slug(self, slug):
        return self.integrations.get(slug)

    def upsert_integration(self, integ, *, spec_raw, spec_hash, created_by):
        row = self.integrations.get(integ.slug)
        iid = row["integration_id"] if row else self._next
        if not row:
            self._next += 1
        self.integrations[integ.slug] = {"integration_id": iid, "spec_hash": spec_hash}
        return iid

    def replace_operations(self, integration_id, ops, req_maps):
        existing = self.ops.setdefault(integration_id, {})
        added = updated = 0
        for op in ops:
            if op.operation_id in existing:
                updated += 1
            else:
                added += 1
            existing[op.operation_id] = op
        incoming = {op.operation_id for op in ops}
        stale = [o for o in existing if o not in incoming]
        return {"added": added, "updated": updated, "deprecated": len(stale)}

    def set_embeddings(self, integration_id, vectors):
        self.embeddings[integration_id] = vectors
        return len([v for v in vectors.values() if v])

    def list_operations(self, integration_id):
        return list(self.ops.get(integration_id, {}).values())

    def operations_missing_embeddings(self, integration_id):
        have = set((self.embeddings.get(integration_id) or {}).keys())
        return [{"operation_id": op.operation_id, "http_method": op.http_method,
                 "path_template": op.path_template, "summary": op.summary,
                 "description": op.description}
                for op in self.ops.get(integration_id, {}).values()
                if op.operation_id not in have]


class _FakeConnections:
    def __init__(self):
        self.providers = {}

    def upsert_provider(self, key, name, **kw):
        self.providers[key] = (name, kw)


class _FakeEmbeddings:
    available = True

    def embed(self, texts):
        return [[float(len(t)), 0.0, 1.0] + [0.0] * 765 for t in texts]


def test_ingest_end_to_end():
    store, conn, emb = _FakeStore(), _FakeConnections(), _FakeEmbeddings()
    ing = SpecIngestor(db=None, connections=conn, embeddings=emb, store=store)
    rep = ing.ingest(PETSTORE, slug="Acme CRM", created_by=7)

    assert rep.slug == "acme_crm"
    assert rep.integration_id == 1
    assert rep.operation_count == 3
    assert rep.added == 3 and rep.updated == 0 and rep.deprecated == 0
    assert rep.embedded == 3
    assert rep.security_scheme == "apiKey"
    assert rep.auth_provider == "api_acme_crm"
    assert "api_acme_crm" in conn.providers
    fields = conn.providers["api_acme_crm"][1]["fields"]
    assert fields[0]["key"] == "api_key"


def test_ingest_unchanged_short_circuits():
    store, conn, emb = _FakeStore(), _FakeConnections(), _FakeEmbeddings()
    ing = SpecIngestor(db=None, connections=conn, embeddings=emb, store=store)
    ing.ingest(PETSTORE, slug="acme_crm")
    conn.providers.clear()
    rep2 = ing.ingest(PETSTORE, slug="acme_crm")
    assert rep2.unchanged is True
    assert rep2.operation_count == 3
    assert conn.providers == {}


def test_ingest_reimport_diff_counts_updated():
    store, conn, emb = _FakeStore(), _FakeConnections(), _FakeEmbeddings()
    ing = SpecIngestor(db=None, connections=conn, embeddings=emb, store=store)
    ing.ingest(PETSTORE, slug="acme_crm")
    spec2 = {**PETSTORE, "paths": {"/leads": PETSTORE["paths"]["/leads"]}}
    rep = ing.ingest(spec2, slug="acme_crm")
    assert rep.unchanged is False
    assert rep.updated == 1
    assert rep.deprecated == 2


def _ingest_with_base(url, allow_private):
    spec = {"openapi": "3.0.0", "servers": [{"url": url}],
            "paths": {"/x": {"get": {"operationId": "x"}}}}
    ing = SpecIngestor(db=None, connections=_FakeConnections(),
                       embeddings=None, store=_FakeStore())
    return ing.ingest(spec, slug="t", allow_private=allow_private)


def test_private_host_blocked_by_default():
    with pytest.raises(openapi.SpecError):
        _ingest_with_base("https://127.0.0.1/v1", allow_private=False)


def test_private_host_allowed_when_opted_in():
    rep = _ingest_with_base("http://127.0.0.1/v1", allow_private=True)
    assert rep.integration_id is not None


def test_metadata_ip_never_allowed():
    with pytest.raises(openapi.SpecError):
        _ingest_with_base("http://169.254.169.254/latest", allow_private=True)


_RELATIVE_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Rel", "version": "1.0.0"},
    "servers": [{"url": "/api/v3"}],
    "paths": {"/pets": {"get": {"operationId": "listPets", "responses": {"200": {"description": "ok"}}}}},
}


def test_ingest_dict_source_with_source_url_resolves_relative_base():
    store, conn, emb = _FakeStore(), _FakeConnections(), _FakeEmbeddings()
    ing = SpecIngestor(db=None, connections=conn, embeddings=emb, store=store)
    rep = ing.ingest(_RELATIVE_SPEC, slug="rel",
                     source_url="https://petstore3.swagger.io/api/v3")
    assert rep.operation_count == 1
    assert rep.added == 1


def test_ingest_dict_relative_without_source_url_raises():
    import pytest
    from core.integrations import openapi as oa
    store, conn, emb = _FakeStore(), _FakeConnections(), _FakeEmbeddings()
    ing = SpecIngestor(db=None, connections=conn, embeddings=emb, store=store)
    with pytest.raises(oa.SpecError):
        ing.ingest(_RELATIVE_SPEC, slug="rel")


def test_force_reimport_rederives_even_when_unchanged():
    """force=True must bypass the spec_hash short-circuit so normalizer upgrades
    (or repairs) re-derive an already-imported integration."""
    store, conn, emb = _FakeStore(), _FakeConnections(), _FakeEmbeddings()
    ing = SpecIngestor(db=None, connections=conn, embeddings=emb, store=store)
    ing.ingest(PETSTORE, slug="acme_crm")
    rep_default = ing.ingest(PETSTORE, slug="acme_crm")
    assert rep_default.unchanged is True
    rep_forced = ing.ingest(PETSTORE, slug="acme_crm", force=True)
    assert rep_forced.unchanged is False
    assert rep_forced.operation_count > 0


def test_normalizer_version_in_spec_hash():
    """Changing NORMALIZER_VERSION must change the stored hash so an unchanged
    spec re-derives once after an ingestor upgrade."""
    import core.integrations.ingest as ing_mod
    h1 = ing_mod._spec_hash({"a": 1})
    orig = ing_mod.NORMALIZER_VERSION
    try:
        ing_mod.NORMALIZER_VERSION = orig + "-test"
        h2 = ing_mod._spec_hash({"a": 1})
    finally:
        ing_mod.NORMALIZER_VERSION = orig
    assert h1 != h2
