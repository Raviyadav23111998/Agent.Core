"""Cache-first context strategy — the properties that make provider prompt
caching work (append-only history, byte-stable prefix, one-shot compaction,
insertion-time result caps, session-sticky tool subsets).

A 'prefix break' = a later LLM call whose message list does not start with
the previous call's full message list. Every break invalidates the provider's
prompt cache at that point and re-bills the remainder at full price.
"""

import asyncio
import copy
import json

import pytest

from core import config as _cfg
from core.agent_loop import AgentLoop
from tools.registry import ToolRegistry
from async_bus.event_bus import EventBus


@pytest.fixture
def cfg(monkeypatch):
    overrides = {
        "agent.context_strategy": "cache_first",
        "agent.history_token_budget": "200000",
        "agent.dynamic_tools": "true",
        "agent.planning_enabled": "false",
        "agent.compaction_llm": "false",  # heuristic → deterministic tests
        "agent.tool_result_max_chars": "30000",
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


def _norm(history):
    return [json.dumps({k: v for k, v in m.items() if k != "_tokens"},
                       sort_keys=True) for m in history]


class FakeLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.model = "claude-test"
        self.max_tokens = 1000

    async def generate(self, prompt, history, tools=None, token_callback=None,
                       model=None, response_format=None):
        self.calls.append({"prompt": prompt,
                           "tools": copy.deepcopy(tools),
                           "messages": _norm(history)})
        if not self.script:
            return {"content": "Done.", "tool_calls": [],
                    "finish_reason": "stop", "usage": {}}
        return self.script.pop(0)


def _tc(i):
    return {"content": None,
            "tool_calls": [{"id": f"c{i}", "name": "echo", "args": {"i": i}}],
            "finish_reason": "tool_calls", "usage": {}}


STOP = {"content": "Done.", "tool_calls": [], "finish_reason": "stop", "usage": {}}


def _breaks(calls):
    n = 0
    for prev, cur in zip(calls, calls[1:]):
        p, c = prev["messages"], cur["messages"]
        if len(c) < len(p) or c[:len(p)] != p:
            n += 1
    return n


def _drive(script, turns, big_result=1200):
    async def main():
        reg = ToolRegistry()
        reg.register("echo", "echo tool", {"i?": "int"},
                     lambda i=0: ("x" * big_result) + f"[{i}]")
        llm = FakeLLM(script)
        agent = AgentLoop("/tmp/ws", llm, reg, EventBus())
        for t in turns:
            await agent._process_input({"user_input": t})
        return agent, llm
    return asyncio.run(main())


def test_append_only_zero_prefix_breaks(cfg):
    agent, llm = _drive([_tc(i) for i in range(8)] + [STOP, STOP],
                        ["please handle the data task for me now",
                         "and now the second part of it too"])
    assert _breaks(llm.calls) == 0
    assert not any("[Compressed:" in str(m.get("content"))
                   for m in agent.conversation_history)
    assert len({c["prompt"] for c in llm.calls}) == 1
    assert len({json.dumps(c["tools"], sort_keys=True) for c in llm.calls}) == 1


def test_legacy_control_breaks_prefix(cfg):
    cfg["agent.context_strategy"] = "legacy"
    _agent, llm = _drive([_tc(i) for i in range(8)] + [STOP, STOP],
                         ["please handle the data task for me now",
                          "and now the second part of it too"])
    assert _breaks(llm.calls) > 0


def test_compaction_is_one_shot(cfg):
    cfg["agent.history_token_budget"] = "3000"
    agent, llm = _drive([_tc(i) for i in range(10)] + [STOP],
                        ["please handle the data task for me now"])
    assert agent._compaction_count >= 1
    assert any(str(m.get("content", "")).startswith("[COMPACTED")
               for m in agent.conversation_history)
    assert _breaks(llm.calls) == agent._compaction_count

    pending = set()
    for m in agent.conversation_history:
        if m.get("role") == "tool":
            assert m.get("tool_call_id") in pending
            pending.discard(m["tool_call_id"])
        else:
            assert not pending
            if m.get("role") == "assistant" and m.get("tool_calls"):
                pending = {tc["id"] for tc in m["tool_calls"]}
    assert not pending


def test_tool_result_capped_at_insertion(cfg):
    agent, _llm = _drive([_tc(0), STOP],
                         ["please handle the data task for me now"],
                         big_result=100_000)
    tool_msgs = [m for m in agent.conversation_history if m.get("role") == "tool"]
    assert tool_msgs
    assert len(tool_msgs[0]["content"]) < 32_000
    assert "chars omitted" in tool_msgs[0]["content"]


def test_session_sticky_tool_subset_grows_never_shrinks(cfg):
    cfg["agent.stable_tools_max"] = "3"

    async def main():
        reg = ToolRegistry()
        for n in ["view_file", "list_dir", "grep_search", "run_command",
                  "jira_create_ticket", "jira_list_tickets",
                  "outlook_send", "fetch_recent_emails"]:
            reg.register(n, n, {}, lambda: "ok")
        llm = FakeLLM([STOP, STOP, STOP])
        agent = AgentLoop("/tmp/ws", llm, reg, EventBus())
        await agent._process_input(
            {"user_input": "please look at the files in the folder now"})
        t1 = {t["function"]["name"] for t in llm.calls[-1]["tools"]}
        await agent._process_input(
            {"user_input": "create a jira ticket for the sprint backlog"})
        t2 = {t["function"]["name"] for t in llm.calls[-1]["tools"]}
        await agent._process_input(
            {"user_input": "please look at the files again for me"})
        t3 = {t["function"]["name"] for t in llm.calls[-1]["tools"]}
        return t1, t2, t3

    t1, t2, t3 = asyncio.run(main())
    assert "jira_create_ticket" in t2 and t1 <= t2
    assert t2 == t3


def test_tool_subset_growth_is_append_only(cfg):
    """When the session subset grows, the new tools array must START with the
    previous array byte-for-byte (new schemas appended at the END). Interleaving
    a new tool at its registry position would shift every schema after it and
    invalidate the provider's prefix cache for the whole request."""
    cfg["agent.stable_tools_max"] = "3"

    async def main():
        reg = ToolRegistry()
        # Registry order deliberately puts jira_* BEFORE the email tools so a
        # registry-ordered serialization would interleave, not append.
        for n in ["jira_create_ticket", "jira_list_tickets", "view_file",
                  "list_dir", "grep_search", "run_command",
                  "outlook_send", "fetch_recent_emails"]:
            reg.register(n, n, {}, lambda: "ok")
        llm = FakeLLM([STOP, STOP])
        agent = AgentLoop("/tmp/ws", llm, reg, EventBus())
        await agent._process_input(
            {"user_input": "please look at the files in the folder now"})
        first = llm.calls[-1]["tools"]
        await agent._process_input(
            {"user_input": "create a jira ticket for the sprint backlog"})
        second = llm.calls[-1]["tools"]
        return first, second

    first, second = asyncio.run(main())
    assert len(second) > len(first)
    assert second[:len(first)] == first  # strict prefix — cache-safe growth


def test_provider_cache_strategy_matrix():
    """cache_strategy: Anthropic needs explicit cache_control markers; DeepSeek/
    OpenAI/Gemini/GLM/Kimi/Qwen/… auto-cache a stable prefix and must get CLEAN
    requests. Model name wins over provider; unknown defaults to auto."""
    from client.cli import GenericLLMClient
    c = GenericLLMClient(api_key="x", model="m", url="http://x/v1/chat/completions")
    assert c.cache_strategy("claude-sonnet-4-5") == "explicit"
    assert c.cache_strategy("anthropic/claude-3.5-haiku") == "explicit"
    for m in ("deepseek-v4-flash", "gpt-4o", "gemini-2.5-pro", "glm-4-plus",
              "qwen-max", "moonshot-v1-8k", "kimi-k2", "grok-3",
              "mistral-large", "meta-llama/llama-3.3-70b"):
        assert c.cache_strategy(m) == "auto", m
    c.provider = "anthropic"
    assert c.cache_strategy("some-custom-model") == "explicit"
    c.provider = "deepseek"
    assert c.cache_strategy("some-custom-model") == "auto"
    c.provider = None
    assert c.cache_strategy("totally-unknown") == "auto"


def test_explicit_markers_only_for_anthropic_shape():
    """_apply_prompt_caching wraps system into content-blocks with
    cache_control (Anthropic shape); auto-caching vendors must never see it —
    generate() gates on cache_strategy() == 'explicit'."""
    from client.cli import GenericLLMClient
    c = GenericLLMClient(api_key="x", model="claude-sonnet-4-5",
                         url="http://x/v1/chat/completions")
    messages = [{"role": "system", "content": "You are the agent."},
                {"role": "user", "content": "hi"}]
    payload = {"tools": [{"type": "function", "function": {"name": "a"}},
                         {"type": "function", "function": {"name": "b"}}]}
    c._apply_prompt_caching(messages, payload)
    assert isinstance(messages[0]["content"], list)
    assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" in payload["tools"][-1]
    assert "cache_control" not in payload["tools"][0]


def test_compaction_survives_none_content_at_index_1(cfg):
    """Regression: history[1] being an assistant tool-call message with
    content=None used to crash _compact_conversation (.get('content','')
    returns the stored None)."""
    cfg["agent.history_token_budget"] = "2500"
    agent, llm = _drive([_tc(i) for i in range(9)] + [STOP],
                        ["please handle the data task for me now"])
    # turn completed with a real answer, not the generic error message
    assert agent.conversation_history[-1].get("role") == "assistant"
    assert agent._compaction_count >= 1
