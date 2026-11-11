"""
Phase 3 regression — per-turn tool-RAG selection (no DB/network).

  1. IntegrationToolProvider.make_picker: embedding-ranks granted ops, scopes by
     principal + embed_safe, degrades to empty when embeddings unavailable.
  2. Selection integration: dynamic ops (identified by json_schema, as
     _select_turn_tools does) are excluded from the keyword default-keep and
     re-added only by the picker — so a large catalog stays flat per turn.

    python -m pytest tests/test_integrations_phase3.py -q
"""

from core.integrations.runtime import IntegrationToolProvider, DynamicHttpExecutor
from core.tool_selection import select_tool_names
from tools.registry import ToolRegistry


class _Ctx:
    def __init__(self, user_id="user_5", app_id="4"):
        self.user_id, self.app_id = user_id, app_id


class _Emb:
    available = True
    def embed_one(self, text):
        return [0.1, 0.2, 0.3]


class _SearchStore:
    def __init__(self, result):
        self.result = result
        self.kwargs = None
    def search_catalog(self, vec, **kw):
        self.kwargs = kw
        return list(self.result)


def _provider(embeddings, store):
    ex = DynamicHttpExecutor(db=object(), connections=object(), store=store)
    return IntegrationToolProvider(connections=object(), embeddings=embeddings,
                                   executor=ex, store=store)


def test_picker_ranks_and_scopes():
    store = _SearchStore(["acme__getlead", "acme__list"])
    pick = _provider(_Emb(), store).make_picker(
        _Ctx(app_id="4"), persona_id="2", embed_safe_only=True, k=5)
    out = pick("show my leads", available={"acme__getlead", "view_file"})
    assert out == {"acme__getlead"}
    assert store.kwargs["app_id"] == "4"
    assert store.kwargs["persona_id"] == "2"
    assert store.kwargs["embed_safe_only"] is True
    assert store.kwargs["k"] == 5


def test_picker_empty_without_embeddings():
    pick = _provider(None, _SearchStore(["acme__x"])).make_picker(_Ctx())
    assert pick("q", {"acme__x"}) == set()


def test_picker_empty_when_embeddings_unavailable():
    class _Off:
        available = False
        def embed_one(self, t): return [0.0]
    pick = _provider(_Off(), _SearchStore(["acme__x"])).make_picker(_Ctx())
    assert pick("q", {"acme__x"}) == set()


def test_picker_swallows_errors():
    class _Boom:
        available = True
        def embed_one(self, t): raise RuntimeError("nope")
    pick = _provider(_Boom(), _SearchStore(["acme__x"])).make_picker(_Ctx())
    assert pick("q", {"acme__x"}) == set()


def _registry_with_dynamic():
    reg = ToolRegistry()
    reg.register("view_file", "Read", {"path": "string"}, lambda **k: "ok")
    for op in ("acme__a", "acme__b", "acme__c"):
        reg.register(op, op, {}, lambda **k: "ok",
                     json_schema={"type": "object", "properties": {}})
    return reg


def test_dynamic_tools_identified_by_json_schema():
    reg = _registry_with_dynamic()
    dynamic = {n for n, sch in reg._tools.items()
               if isinstance(sch, dict) and sch.get("json_schema")}
    assert dynamic == {"acme__a", "acme__b", "acme__c"}


def test_large_catalog_stays_flat_per_turn():
    reg = _registry_with_dynamic()
    names = {t["function"]["name"] for t in reg.get_openai_tools()}
    dynamic = {n for n, sch in reg._tools.items() if sch.get("json_schema")}

    keep = select_tool_names(
        "anything", names, dynamic_tools=dynamic,
        dynamic_picker=lambda text, avail: {"acme__b"})

    assert "view_file" in keep
    assert "acme__b" in keep
    assert "acme__a" not in keep and "acme__c" not in keep


def test_no_picker_means_no_dynamic_ops():
    reg = _registry_with_dynamic()
    names = {t["function"]["name"] for t in reg.get_openai_tools()}
    dynamic = {n for n, sch in reg._tools.items() if sch.get("json_schema")}
    keep = select_tool_names("anything", names, dynamic_tools=dynamic, dynamic_picker=None)
    assert "view_file" in keep
    assert not ({"acme__a", "acme__b", "acme__c"} & keep)
