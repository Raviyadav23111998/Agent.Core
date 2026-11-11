"""
Embed sandbox containment tests — the security core of Phase 1.
Proves an embedded session can NEVER reach admin/filesystem/server tools,
regardless of mode or persona content.
"""
from core.embed_sandbox import build_embed_registry, embed_safe_tools
from tools.registry import ToolRegistry


def _full_registry():
    r = ToolRegistry()
    for name in ("brain_ask", "brain_search",
                 "page_observe", "page_act", "page_navigate", "page_read", "page_wait",
                 "run_command", "write_to_file", "view_file",
                 "jira_create_ticket", "agent_create", "memory_store"):
        r.register(name, f"desc {name}", {}, lambda **k: "ok")
    return r


_PRIVILEGED = {"run_command", "write_to_file", "view_file",
               "jira_create_ticket", "agent_create", "memory_store"}


def test_chat_mode_is_knowledge_only():
    scoped = build_embed_registry(_full_registry(), "chat")
    names = set(scoped._tools.keys())
    assert names == {"brain_ask", "brain_search"}
    assert not (names & _PRIVILEGED)


def test_copilot_mode_adds_page_tools_only():
    scoped = build_embed_registry(_full_registry(), "copilot")
    names = set(scoped._tools.keys())
    assert "page_act" in names and "brain_ask" in names
    assert not (names & _PRIVILEGED), "privileged tools leaked into copilot embed"


def test_persona_cannot_widen_beyond_ceiling():
    persona = ["brain_ask", "run_command", "write_to_file", "agent_create", "page_act"]
    scoped = build_embed_registry(_full_registry(), "copilot", persona_allowed=persona)
    names = set(scoped._tools.keys())
    assert names == {"brain_ask", "page_act"}
    assert not (names & _PRIVILEGED)


def test_persona_narrows_within_ceiling():
    persona = ["brain_search", "page_observe"]
    scoped = build_embed_registry(_full_registry(), "copilot", persona_allowed=persona)
    assert set(scoped._tools.keys()) == {"brain_search", "page_observe"}


def test_ceiling_never_contains_privileged():
    for mode in ("chat", "copilot"):
        assert not (embed_safe_tools(mode) & _PRIVILEGED)
