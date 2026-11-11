"""Claude Code parity features — G1..G10 regression tests.

Covers: native Anthropic conversion (thinking replay, is_error, cache
breakpoints), read-parallel/write-serial batching, todo_write + staleness
reminder, cancel/steering, parallel sub-agents + depth guard, shell cwd
persistence + exit codes, glob/grep modes, edit replace_all, resume
export/restore.
"""

import asyncio
import json
import os
import time

import pytest

from core import config as _cfg


@pytest.fixture
def cfg(monkeypatch):
    overrides = {
        "agent.context_strategy": "cache_first",
        "agent.history_token_budget": "200000",
        "agent.dynamic_tools": "true",
        "agent.planning_enabled": "false",
        "agent.compaction_llm": "false",
        "agent.behavior_nudges": "false",
        "agent.session_resume": "false",
    }
    ac = _cfg.app_config
    orig_get = ac.get

    def _get(key, default=None, *a, **k):
        return overrides.get(key, orig_get(key, default))

    def _get_bool(key, default=False):
        v = _get(key, None)
        if v is None:
            return default
        return str(v).strip().lower() in ("1", "true", "yes", "on")

    def _get_int(key, default=0):
        v = _get(key, None)
        try:
            return int(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    monkeypatch.setattr(ac, "get", _get)
    monkeypatch.setattr(ac, "get_bool", _get_bool)
    monkeypatch.setattr(ac, "get_int", _get_int)
    return overrides


# ──────────────────────────────────────────────────────────────────────
# G1 — native Anthropic conversion
# ──────────────────────────────────────────────────────────────────────

def _anthropic():
    from client.anthropic_client import AnthropicMessagesClient
    return AnthropicMessagesClient(api_key="k", model="claude-test", stream=False)


def test_anthropic_history_conversion():
    c = _anthropic()
    history = [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": None,
         "_thinking": [{"type": "thinking", "thinking": "hmm", "signature": "sig1"}],
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "echo", "arguments": '{"i": 1}'}}]},
        {"role": "tool", "tool_call_id": "t1", "name": "echo",
         "content": "Error executing echo: boom", "_is_error": True},
        {"role": "tool", "tool_call_id": "t2", "name": "echo", "content": "ok"},
        {"role": "user", "content": "and more"},
    ]
    msgs = c._convert_history(history)

    assert msgs[0]["role"] == "user"
    a = msgs[1]
    assert a["role"] == "assistant"
    # thinking block FIRST, verbatim with signature
    assert a["content"][0] == {"type": "thinking", "thinking": "hmm",
                               "signature": "sig1"}
    tool_use = [b for b in a["content"] if b["type"] == "tool_use"][0]
    assert tool_use["id"] == "t1" and tool_use["input"] == {"i": 1}
    # tool results + following user text merged into ONE user message
    u = msgs[2]
    assert u["role"] == "user"
    types = [b["type"] for b in u["content"]]
    assert types == ["tool_result", "tool_result", "text"]
    assert u["content"][0]["is_error"] is True
    assert "is_error" not in u["content"][1]
    assert len(msgs) == 3  # roles alternate — no consecutive same-role


def test_anthropic_cache_breakpoints_and_tools():
    c = _anthropic()
    tools = [{"type": "function", "function": {"name": "a", "description": "d",
                                               "parameters": {"type": "object"}}},
             {"type": "function", "function": {"name": "b", "description": "d",
                                               "parameters": {"type": "object"}}}]
    at = c._convert_tools(tools)
    assert "cache_control" not in at[0] and at[-1]["cache_control"]["type"] == "ephemeral"
    assert at[0]["input_schema"] == {"type": "object"}

    history = [
        {"role": "user", "content": "start"},
        {"role": "user", "content": "[COMPACTED CONTEXT — 5 messages]\nsummary"},
        {"role": "user", "content": "next"},
    ]
    msgs = c._convert_history(history)
    blocks = msgs[0]["content"]
    compacted = [b for b in blocks if b["text"].startswith("[COMPACTED")]
    assert compacted and compacted[0]["cache_control"] == {"type": "ephemeral"}

    payload = c._build_payload("sys", history, tools, "claude-test", None, None)
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_usage_mapping():
    c = _anthropic()
    u = c._map_usage({"input_tokens": 100, "output_tokens": 50,
                      "cache_read_input_tokens": 900,
                      "cache_creation_input_tokens": 10})
    assert u["prompt_tokens"] == 1010
    assert u["completion_tokens"] == 50
    assert u["cache_read_input_tokens"] == 900


# ──────────────────────────────────────────────────────────────────────
# G2 — read-parallel / write-serial batching
# ──────────────────────────────────────────────────────────────────────

def test_reads_parallel_writes_serial(cfg):
    from core.agent_loop import AgentLoop
    from tools.registry import ToolRegistry
    from async_bus.event_bus import EventBus

    events = []

    def make(name):
        def fn(**kw):
            events.append((name, "start", time.monotonic()))
            time.sleep(0.15)
            events.append((name, "end", time.monotonic()))
            return name
        return fn

    async def main():
        reg = ToolRegistry()
        reg.register("read_alpha", "r", {}, make("read_alpha"))
        reg.register("read_beta", "r", {}, make("read_beta"))
        reg.register("write_gamma", "w", {}, make("write_gamma"))
        reg.register("write_delta", "w", {}, make("write_delta"))

        class _L:  # no LLM needed
            model = "m"
        agent = AgentLoop("/tmp/ws", _L(), reg, EventBus())
        calls = [{"name": n, "args": {}} for n in
                 ("read_alpha", "write_gamma", "read_beta", "write_delta")]
        results = await agent._execute_tools_parallel(calls)
        return results

    results = asyncio.run(main())
    assert results == ["read_alpha", "write_gamma", "read_beta", "write_delta"]

    t = {(n, k): ts for n, k, ts in events}
    # the two reads overlap (parallel)
    assert t[("read_beta", "start")] < t[("read_alpha", "end")]
    # writes are serialized relative to each other
    assert t[("write_delta", "start")] >= t[("write_gamma", "end")]


# ──────────────────────────────────────────────────────────────────────
# G6 — todo_write + staleness reminder
# ──────────────────────────────────────────────────────────────────────

def test_todo_store_and_reminder(cfg):
    from state.todos import TodoStore, current_todo_store, todo_write
    from core.agent_loop import AgentLoop
    from tools.registry import ToolRegistry
    from async_bus.event_bus import EventBus

    store = TodoStore()
    token = current_todo_store.set(store)
    try:
        out = todo_write([{"content": "step one", "status": "in_progress"},
                          {"content": "step two", "status": "pending"}])
        assert "1 in progress" in out and "[~] step one" in out
        assert store.open_count == 2
    finally:
        current_todo_store.reset(token)

    async def main():
        reg = ToolRegistry()

        class _L:
            model = "m"
        agent = AgentLoop("/tmp/ws", _L(), reg, EventBus())
        agent.todos.update([{"content": "a", "status": "pending"}])
        for _ in range(agent.TODO_STALE_ITERATIONS + 1):
            agent._maybe_todo_reminder()
        return agent

    agent = asyncio.run(main())
    reminders = [m for m in agent.conversation_history
                 if "[SYSTEM REMINDER]" in str(m.get("content", ""))]
    assert len(reminders) == 1


# ──────────────────────────────────────────────────────────────────────
# G8 — cancel + steering
# ──────────────────────────────────────────────────────────────────────

class _ScriptLLM:
    def __init__(self, script):
        self.script = list(script)
        self.model = "m"
        self.max_tokens = 100

    async def generate(self, prompt, history, tools=None, token_callback=None,
                       model=None, response_format=None, **kw):
        if not self.script:
            return {"content": "Done.", "tool_calls": [],
                    "finish_reason": "stop", "usage": {}}
        return self.script.pop(0)


def _tc(i):
    return {"content": None,
            "tool_calls": [{"id": f"c{i}", "name": "echo", "args": {"i": i}}],
            "finish_reason": "tool_calls", "usage": {}}


def test_cancel_mid_turn(cfg):
    from core.agent_loop import AgentLoop
    from tools.registry import ToolRegistry
    from async_bus.event_bus import EventBus

    async def main():
        reg = ToolRegistry()
        agent_ref = {}

        def echo(i=0):
            agent_ref["a"].request_cancel()
            return "x"

        reg.register("echo", "e", {"i?": "int"}, echo)
        llm = _ScriptLLM([_tc(0), _tc(1), _tc(2)])
        bus = EventBus()
        responses = []
        bus.subscribe('agent_response', lambda p: responses.append(p))
        agent = AgentLoop("/tmp/ws", llm, reg, bus)
        agent_ref["a"] = agent
        await agent._process_input({"user_input": "please do the long thing now"})
        return agent, llm, responses

    agent, llm, responses = asyncio.run(main())
    assert any(p.get("cancelled") for p in responses)
    assert len(llm.script) == 2  # only one iteration ran
    # pairing intact after cancellation
    for m in agent.conversation_history:
        if m.get("role") == "assistant":
            assert not m.get("tool_calls") or any(
                t.get("role") == "tool" for t in agent.conversation_history)


def test_steering_message_injected(cfg):
    from core.agent_loop import AgentLoop
    from tools.registry import ToolRegistry
    from async_bus.event_bus import EventBus

    async def main():
        reg = ToolRegistry()
        reg.register("echo", "e", {"i?": "int"}, lambda i=0: "x")
        llm = _ScriptLLM([_tc(0), {"content": "Done.", "tool_calls": [],
                                   "finish_reason": "stop", "usage": {}}])
        agent = AgentLoop("/tmp/ws", llm, reg, EventBus())
        agent.queue_steering_message("actually use the staging server")
        await agent._process_input({"user_input": "please do the deploy now"})
        return agent

    agent = asyncio.run(main())
    steer = [m for m in agent.conversation_history
             if "arrived mid-task" in str(m.get("content", ""))]
    assert len(steer) == 1 and "staging server" in steer[0]["content"]


# ──────────────────────────────────────────────────────────────────────
# G3 — parallel sub-agents + depth guard
# ──────────────────────────────────────────────────────────────────────

def test_subagents_run_in_parallel(cfg):
    from agents.task_manager import SubAgentManager, delegation_depth

    from agents.factory import RunResult

    class _Factory:
        async def run_agent(self, definition, task_prompt=None, timeout=600):
            await asyncio.sleep(0.2)
            # The factory boundary returns a RunResult; SubAgentManager unwraps
            # .output. The fake must honour that contract.
            return RunResult(output=f"report:{definition.name}")

    class _Def:
        def __init__(self, name):
            self.name = name

    async def main():
        from async_bus.event_bus import EventBus
        mgr = SubAgentManager(EventBus())
        start = time.monotonic()
        r1, r2 = await asyncio.gather(
            mgr.run(_Factory(), _Def("a"), "t"),
            mgr.run(_Factory(), _Def("b"), "t"),
        )
        return r1, r2, time.monotonic() - start

    r1, r2, elapsed = asyncio.run(main())
    assert r1 == "report:a" and r2 == "report:b"
    assert elapsed < 0.38  # parallel, not 0.4 sequential

    token = delegation_depth.set(2)
    try:
        assert SubAgentManager.check_depth() is not None
    finally:
        delegation_depth.reset(token)


def test_spawn_delivers_to_conversation_bus(cfg):
    """Background sub-agent completion must reach the SPAWNING conversation's
    bus (API sessions: one bus per conversation), not the host/CLI bus."""
    from agents.task_manager import SubAgentManager, current_conversation_bus
    from async_bus.event_bus import EventBus

    from agents.factory import RunResult

    class _Factory:
        async def run_agent(self, definition, task_prompt=None, timeout=600):
            return RunResult(output="bg-report")

    class _Def:
        name = "explore"

    async def main():
        host_bus = EventBus()
        conv_bus = EventBus()
        host_got, conv_got = [], []
        host_bus.subscribe("subagent_completed", lambda p: host_got.append(p))
        conv_bus.subscribe("subagent_completed", lambda p: conv_got.append(p))
        mgr = SubAgentManager(host_bus)
        token = current_conversation_bus.set(conv_bus)
        try:
            tid = mgr.spawn(_Factory(), _Def(), "task")
        finally:
            current_conversation_bus.reset(token)
        for _ in range(50):
            if conv_got:
                break
            await asyncio.sleep(0.02)
        return tid, conv_got, host_got

    tid, conv_got, host_got = asyncio.run(main())
    assert conv_got and conv_got[0]["task_id"] == tid
    assert conv_got[0]["result"] == "bg-report"
    assert not host_got


def test_loop_consumes_subagent_completion(cfg):
    """subagent_completed on the loop's bus becomes a turn with the report."""
    from core.agent_loop import AgentLoop
    from tools.registry import ToolRegistry
    from async_bus.event_bus import EventBus

    async def main():
        llm = _ScriptLLM([])  # single stop response
        bus = EventBus()
        agent = AgentLoop("/tmp/ws", llm, ToolRegistry(), bus)
        await bus.publish("subagent_completed", {
            "task_id": "sub_1", "agent_name": "explore", "result": "findings X"})
        return agent

    agent = asyncio.run(main())
    injected = [m for m in agent.conversation_history
                if "[SUBAGENT COMPLETED]" in str(m.get("content", ""))]
    assert injected and "findings X" in injected[0]["content"]


def test_explore_builtin_resolves():
    from agents.factory import BUILTIN_AGENTS, resolve_agent_definition

    class _Reg:
        def load(self, name):
            return None

    d = resolve_agent_definition(_Reg(), "Explore")
    assert d is BUILTIN_AGENTS["explore"]
    assert "view_file" in d.tools and "write_to_file" not in d.tools


def test_async_tool_callbacks_run_on_loop():
    from tools.registry import ToolRegistry
    from core.tool_executor import LocalToolExecutor

    reg = ToolRegistry()

    async def atool(x="1"):
        await asyncio.sleep(0)
        return f"async:{x}"

    reg.register("atool", "a", {"x?": "string"}, atool)
    assert reg.is_async("atool") and not reg.is_async("missing")

    async def main():
        ex = LocalToolExecutor(reg)
        return await ex.execute("atool", {"x": "7"})

    assert asyncio.run(main()) == "async:7"


# ──────────────────────────────────────────────────────────────────────
# G4 — shell cwd persistence + glob/grep/replace_all
# ──────────────────────────────────────────────────────────────────────

def test_shell_cwd_persists_and_exit_codes(tmp_path):
    from tools.shell_execution import ShellExecutionTool
    sub = tmp_path / "subdir"
    sub.mkdir()
    sh = ShellExecutionTool(str(tmp_path))

    out = sh.run_command("cd subdir && echo moved")
    assert "moved" in out and "__AG_CWD__" not in out
    out2 = sh.run_command("pwd")
    assert out2.strip().endswith("subdir")

    out3 = sh.run_command("echo boom >&2; exit 3")
    assert "(exit code: 3)" in out3

    import inspect
    sig = inspect.signature(ShellExecutionTool.run_command)
    assert sig.parameters["wait_ms_before_async"].default == 120_000
    assert sig.parameters["cwd"].default is None


def test_glob_and_grep_modes(tmp_path):
    from tools.filesystem import FilesystemTools
    (tmp_path / "a.py").write_text("alpha = 1\nneedle here\nomega = 2\n")
    time.sleep(0.02)
    (tmp_path / "b.py").write_text("no match\n")
    (tmp_path / "c.txt").write_text("needle here too\n")
    fs = FilesystemTools(str(tmp_path))

    out = fs.glob_files("*.py")
    lines = [l for l in out.splitlines()[1:] if l]
    assert lines[0].endswith("b.py") and lines[1].endswith("a.py")  # newest first

    out = fs.grep_search(str(tmp_path), "needle", files_with_matches=True)
    assert "a.py" in out and "c.txt" in out and "needle here" not in out

    out = fs.grep_search(str(tmp_path), "needle", context_lines=1,
                         includes="*.py")
    assert "alpha = 1" in out and "omega = 2" in out

    out = fs.grep_search(str(tmp_path), "e", max_results=2)
    assert "capped at 2" in out


def test_replace_all_and_edit_snippet(tmp_path):
    from tools.filesystem import FilesystemTools
    f = tmp_path / "x.py"
    f.write_text("foo()\nbar()\nfoo()\n")
    fs = FilesystemTools(str(tmp_path))

    out = fs.replace_file_content(str(f), "foo()", "baz()")
    assert out.startswith("Error") and "replace_all" in out

    out = fs.replace_file_content(str(f), "foo()", "baz()", replace_all=True)
    assert "Replaced 2 occurrence(s)" in out
    assert f.read_text() == "baz()\nbar()\nbaz()\n"

    out = fs.replace_file_content(str(f), "bar()", "qux()")
    assert "Edited region now reads:" in out and "qux()" in out


# ──────────────────────────────────────────────────────────────────────
# G9b — doctrine layering: personas cannot delete engine doctrine
# ──────────────────────────────────────────────────────────────────────

def test_doctrine_survives_persona_override():
    from core.prompt_builder import PromptBuilder
    pb = PromptBuilder("/tmp/ws")
    pb.set_available_tools(["todo_write", "run_command", "view_file",
                            "delegate_task", "cron_create"])
    p = pb.build_system_prompt(
        custom_identity="You are BrandBot, the ACME assistant.",
        custom_constraints="You may ONLY access email.",
        custom_communication="Formal tone.")
    assert "<engine_doctrine>" in p
    assert "BrandBot" in p and "ONLY access email" in p and "Formal tone." in p
    # doctrine blocks for available tools present
    assert "todo_write" in p and "delegate_task" in p and "exit code" in p
    # default identity/constraints fully replaced
    assert "autonomous AI agent. You work with the user" not in p


def test_doctrine_gates_on_available_tools():
    from core.prompt_builder import PromptBuilder
    pb = PromptBuilder("/tmp/ws")
    pb.set_available_tools(["fetch_recent_emails", "outlook_reply"])
    p = pb.build_system_prompt()
    assert "<engine_doctrine>" in p
    # universal doctrine present
    assert "PICK THE RIGHT TOOL ONCE" in p
    # tool-gated blocks absent for this persona's toolset
    assert "todo_write" not in p
    assert "delegate_task" not in p
    assert "cron_create" not in p


# ── productivity fixes: sandbox-aware shell policy, Retry-After, effort ──

def test_sandboxed_shell_policy_is_kernel_jail_aware(tmp_path, monkeypatch):
    from tools.shell_execution import ShellExecutionTool

    # kernel-jailed (bwrap present) → write: no approval gate
    monkeypatch.setattr(ShellExecutionTool, "_BWRAP_PATH", "/usr/bin/bwrap")
    jailed = ShellExecutionTool(str(tmp_path), sandboxed=True)
    p = jailed.recommended_policy()
    assert p.side_effect == "write" and not p.requires_approval

    # sandboxed but NO bwrap (regex-only fallback) → approval stays
    monkeypatch.setattr(ShellExecutionTool, "_BWRAP_PATH", None)
    weak = ShellExecutionTool(str(tmp_path), sandboxed=True)
    assert weak.recommended_policy().requires_approval

    # unsandboxed → approval stays
    monkeypatch.setattr(ShellExecutionTool, "_BWRAP_PATH", "/usr/bin/bwrap")
    open_shell = ShellExecutionTool(str(tmp_path), sandboxed=False)
    assert open_shell.recommended_policy().requires_approval


def test_retry_after_parsed_and_carried():
    from client.cli import GenericLLMClient, _RetryableError

    class _Resp:
        headers = {"retry-after": "7"}
    assert GenericLLMClient._retry_after_s(_Resp()) == 7.0

    class _NoHeader:
        headers = {}
    assert GenericLLMClient._retry_after_s(_NoHeader()) == 0.0

    e = _RetryableError("HTTP 429", retry_after=7.0)
    assert e.retry_after == 7.0


def test_reasoning_effort_attached_for_openrouter(cfg, monkeypatch):
    from client.cli import GenericLLMClient

    captured = {}

    def fake_sync(self, headers, payload, attempt):
        captured.update(payload)
        return {"content": "ok", "tool_calls": [],
                "finish_reason": "stop", "usage": {}}

    monkeypatch.setattr(GenericLLMClient, "_generate_sync", fake_sync)
    c = GenericLLMClient(api_key="k", model="m", stream=False)
    c.provider = "openrouter"

    asyncio.run(c.generate("sys", [{"role": "user", "content": "hi"}],
                           reasoning_effort="low"))
    assert captured.get("reasoning") == {"effort": "low"}

    captured.clear()
    asyncio.run(c.generate("sys", [{"role": "user", "content": "hi"}],
                           reasoning_effort="off"))
    assert captured.get("reasoning") == {"enabled": False}

    # non-openrouter providers never get the field
    captured.clear()
    c.provider = "deepseek"
    asyncio.run(c.generate("sys", [{"role": "user", "content": "hi"}],
                           reasoning_effort="low"))
    assert "reasoning" not in captured


# ──────────────────────────────────────────────────────────────────────
# G7 — resume roundtrip
# ──────────────────────────────────────────────────────────────────────

def test_history_export_restore_roundtrip(cfg):
    from core.agent_loop import AgentLoop
    from tools.registry import ToolRegistry
    from async_bus.event_bus import EventBus

    async def main():
        reg = ToolRegistry()

        class _L:
            model = "m"
        a1 = AgentLoop("/tmp/ws", _L(), reg, EventBus())
        a1._append_message({"role": "user", "content": "hello"})
        a1._append_message({"role": "assistant", "content": "hi",
                            "_thinking": [{"type": "thinking", "thinking": "t",
                                           "signature": "s"}]})
        exported = a1.export_history()
        assert all("_tokens" not in m for m in exported)

        a2 = AgentLoop("/tmp/ws", _L(), reg, EventBus())
        a2.restore_history(exported)
        return a1, a2

    a1, a2 = asyncio.run(main())
    assert len(a2.conversation_history) == 2
    assert a2.conversation_history[1]["_thinking"][0]["signature"] == "s"
    assert a2._total_history_tokens > 0
    assert a2._total_history_tokens == a1._total_history_tokens
