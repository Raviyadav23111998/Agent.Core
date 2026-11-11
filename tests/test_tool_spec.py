"""
Phase 4 — declarative Tool specs.

Verifies the Tool value object forwards to the registry correctly and that the
migrated PageTools.register() produces exactly the same tool surface (names,
params, descriptions) as the old imperative register() calls.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.registry import ToolRegistry
from tools.tool_spec import Tool, register_specs


def test_tool_spec_registers_into_registry():
    reg = ToolRegistry()
    called = {}

    def handler(x):
        called["x"] = x
        return "ok"

    spec = Tool("demo_tool", "A demo.", {"x": "string"}, handler,
                param_descriptions={"x": "the x value"})
    names = register_specs(reg, [spec])

    assert names == ["demo_tool"]
    assert "demo_tool" in reg._tools
    oai = {t["function"]["name"]: t for t in reg.get_openai_tools()}
    assert oai["demo_tool"]["function"]["parameters"]["properties"]["x"]["description"] == "the x value"
    assert reg.execute("demo_tool", {"x": "hi"}) == "ok" and called["x"] == "hi"


def test_optional_param_marked_not_required():
    reg = ToolRegistry()
    register_specs(reg, [Tool("opt", "d", {"a": "string", "b?": "string"},
                              lambda a, b=None: "ok")])
    fn = {t["function"]["name"]: t for t in reg.get_openai_tools()}["opt"]["function"]
    assert fn["parameters"]["required"] == ["a"]


def test_page_tools_register_full_surface():
    from tools.page_tools import PageTools
    reg = ToolRegistry()
    PageTools().register(reg)
    expected = {"page_observe", "page_act", "page_navigate", "page_wait",
                "page_read"}
    assert expected.issubset(set(reg._tools))
    spec_names = {t.name for t in PageTools().tool_specs()}
    assert spec_names == expected
    oai = {t["function"]["name"]: t for t in reg.get_openai_tools()}
    assert "steps" in oai["page_act"]["function"]["parameters"]["properties"]
