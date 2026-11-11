"""
Phase 0 regression — dynamic API integrations spine (no DB/network).

Covers the four back-compatible primitives the later phases depend on:
  1. ToolRegistry JSON-Schema pass-through (nested OpenAPI bodies survive).
  2. ToolRegistry.execute passes nested args straight to the callback.
  3. HTTP method → ToolPolicy / side-effect / approval mapping.
  4. tool_selection excludes dynamic ops from default-keep; picker re-adds.

    python -m pytest tests/test_integrations_phase0.py -q
"""

from tools.registry import ToolRegistry
from tools.tool_policy import READ, WRITE, IRREVERSIBLE
from core.integrations.model import (
    policy_for_method, side_effect_for_method, requires_approval_for_method,
    tool_name_for, IntegrationOperation,
)
from core.tool_selection import select_tool_names


_NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "lead": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["new", "won", "lost"]},
            },
            "required": ["name"],
        },
    },
    "required": ["id", "lead"],
}


def test_full_json_schema_emitted_verbatim():
    reg = ToolRegistry()
    reg.register("acme__create_lead", "Create a lead", {}, lambda **k: "ok",
                 json_schema=_NESTED_SCHEMA)
    tools = reg.get_openai_tools()
    fn = next(t["function"] for t in tools if t["function"]["name"] == "acme__create_lead")
    assert fn["parameters"] == _NESTED_SCHEMA
    assert fn["parameters"]["properties"]["lead"]["properties"]["status"]["enum"] == \
        ["new", "won", "lost"]


def test_builtin_flat_tool_unaffected():
    reg = ToolRegistry()
    reg.register("view_file", "Read a file", {"path": "string", "n?": "int"},
                 lambda **k: "ok")
    fn = next(t["function"] for t in reg.get_openai_tools()
              if t["function"]["name"] == "view_file")
    assert fn["parameters"]["type"] == "object"
    assert fn["parameters"]["properties"]["path"] == {"type": "string", "description": "path"}
    assert fn["parameters"]["properties"]["n"]["type"] == "integer"
    assert fn["parameters"]["required"] == ["path"]


def test_execute_passes_nested_args():
    seen = {}
    reg = ToolRegistry()
    reg.register("acme__create_lead", "Create a lead", {},
                 lambda **kwargs: seen.update(kwargs) or "done",
                 json_schema=_NESTED_SCHEMA)
    out = reg.execute("acme__create_lead",
                      {"id": 7, "lead": {"name": "Ada", "tags": ["vip"], "status": "new"}})
    assert out == "done"
    assert seen["id"] == 7
    assert seen["lead"] == {"name": "Ada", "tags": ["vip"], "status": "new"}


def test_method_side_effects():
    assert side_effect_for_method("get") == READ
    assert side_effect_for_method("HEAD") == READ
    assert side_effect_for_method("post") == WRITE
    assert side_effect_for_method("PUT") == WRITE
    assert side_effect_for_method("delete") == IRREVERSIBLE


def test_read_needs_no_approval_writes_do():
    assert requires_approval_for_method("GET") is False
    assert requires_approval_for_method("POST") is True
    assert requires_approval_for_method("DELETE") is True


def test_policy_gating():
    g = policy_for_method("GET")
    assert g.side_effect == READ and g.cacheable and not g.requires_approval
    d = policy_for_method("DELETE")
    assert d.requires_approval is True
    assert policy_for_method("POST").requires_approval is True
    assert policy_for_method("POST", requires_approval=False).side_effect == WRITE
    assert policy_for_method("POST", requires_approval=False).requires_approval is False


def test_operation_policy_roundtrip():
    op = IntegrationOperation(
        operation_id="deleteLead", tool_name="acme__deleteLead",
        http_method="DELETE", path_template="/leads/{id}", requires_approval=True)
    assert op.policy().requires_approval is True


def test_tool_name_namespacing_and_truncation():
    assert tool_name_for("Acme CRM", "createLead") == "acme_crm__createlead"
    long = tool_name_for("x" * 100, "y" * 100)
    assert len(long) <= 160 and "__" in long


def test_dynamic_ops_not_default_kept():
    available = {"view_file", "acme__a", "acme__b", "acme__c"}
    assert {"acme__a", "acme__b"} <= select_tool_names("hello", available)
    sel = select_tool_names("hello", available, dynamic_tools={"acme__a", "acme__b", "acme__c"})
    assert "view_file" in sel
    assert not ({"acme__a", "acme__b", "acme__c"} & sel)


def test_dynamic_picker_readds_relevant():
    available = {"view_file", "acme__a", "acme__b"}
    sel = select_tool_names(
        "anything", available,
        dynamic_tools={"acme__a", "acme__b"},
        dynamic_picker=lambda text, avail: {"acme__b"})
    assert "acme__b" in sel and "acme__a" not in sel


def test_picker_exception_is_swallowed():
    available = {"acme__a"}
    sel = select_tool_names("x", available, dynamic_tools={"acme__a"},
                            dynamic_picker=lambda t, a: (_ for _ in ()).throw(RuntimeError()))
    assert isinstance(sel, set)
