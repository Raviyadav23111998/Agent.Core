"""Phase 1 core plumbing for per-app/per-user model profiles + budgets.

- create_llm_client(profile=): every connection value pinned on the instance
  (no drift with global llm.* config), config-sentinel api_key resolution,
  provider-routed class selection.
- get_profile_client: one client per profile generation, edit busts cache.
- AgentLoop budget_check: host-injected limit stops the turn gracefully.
"""

import asyncio

from client.cli import (GenericLLMClient, create_llm_client,
                        get_profile_client, invalidate_profile_clients,
                        _profile_clients)


PROFILE_GLM = {
    "profile_id": 7,
    "name": "coding-glm",
    "provider": "zhipu",
    "model": "glm-5.2",
    "base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "api_key": "glm-key-123",
    "fallback_model": "glm-4-plus",
    "temperature": 0.3,
    "max_tokens": 8000,
    "reasoning_effort": "low",
    "updated_at": "2026-07-04T10:00:00",
}


def test_profile_pins_every_connection_value():
    c = create_llm_client(profile=PROFILE_GLM)
    assert isinstance(c, GenericLLMClient)
    assert c.model == "glm-5.2"
    assert c.api_key == "glm-key-123"
    assert "bigmodel.cn" in c.url
    assert c.provider == "zhipu"
    assert float(c.temperature) == 0.3
    assert int(c.max_tokens) == 8000
    assert c.fallback_model == "glm-4-plus"
    assert c._reasoning_effort_override == "low"
    assert c.profile_id == 7
    # auto-caching vendor → clean requests
    assert c.cache_strategy() == "auto"


def test_profile_anthropic_routes_native_adapter():
    from client.anthropic_client import AnthropicMessagesClient
    prof = dict(PROFILE_GLM, provider="anthropic", model="claude-sonnet-4-6",
                base_url="https://api.anthropic.com/v1/messages")
    c = create_llm_client(profile=prof)
    assert isinstance(c, AnthropicMessagesClient)
    assert c.model == "claude-sonnet-4-6"


def test_profile_config_sentinel_resolves_api_key(monkeypatch):
    from core import config as _cfg
    orig = _cfg.app_config.get
    monkeypatch.setattr(
        _cfg.app_config, "get",
        lambda k, d=None, **kw: "secret-from-config" if k == "llm.api_key"
        else orig(k, d))
    prof = dict(PROFILE_GLM, api_key="__config:llm.api_key__")
    c = create_llm_client(profile=prof)
    assert c.api_key == "secret-from-config"


def test_profile_client_cache_and_invalidation():
    invalidate_profile_clients()
    c1 = get_profile_client(PROFILE_GLM)
    c2 = get_profile_client(PROFILE_GLM)
    assert c1 is c2  # same generation → shared instance (stable prefix)
    edited = dict(PROFILE_GLM, updated_at="2026-07-04T11:00:00",
                  model="glm-5.2-air")
    c3 = get_profile_client(edited)
    assert c3 is not c1 and c3.model == "glm-5.2-air"
    # stale generation evicted
    assert len([k for k in _profile_clients if k[1] == 7]) == 1  # key = (tenant, id, updated_at, stream)
    invalidate_profile_clients(7)
    assert not [k for k in _profile_clients if k[1] == 7]


def test_budget_check_stops_turn_gracefully():
    from core.agent_loop import AgentLoop
    from tools.registry import ToolRegistry
    from async_bus.event_bus import EventBus

    class LoopingLLM:
        model, max_tokens = "deepseek-v4-flash", 1000

        def __init__(self):
            self.calls = 0

        async def generate(self, prompt, history, tools=None, **kw):
            self.calls += 1
            return {"content": None,
                    "tool_calls": [{"id": f"c{self.calls}", "name": "echo",
                                    "args": {"i": self.calls}}],
                    "finish_reason": "tool_calls",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    events = []

    async def main():
        reg = ToolRegistry()
        reg.register("echo", "echo", {"i?": "int"}, lambda i=0: f"ok {i}")
        llm = LoopingLLM()
        bus = EventBus()
        bus.subscribe("agent_response", lambda p: events.append(p))
        state = {"n": 0}

        def over_budget():
            state["n"] += 1
            if state["n"] >= 2:
                return ("You've reached your daily usage limit. "
                        "Resets at 00:00 UTC.")
            return None

        agent = AgentLoop("/tmp/ws", llm, reg, bus, budget_check=over_budget)
        await agent._process_input({"user_input": "loop forever please"})
        return llm.calls

    calls = asyncio.run(main())
    assert calls <= 4  # stopped early, not at max_iterations
    assert any(e.get("budget_exceeded") for e in events)
    assert any("daily usage limit" in str(e.get("message", "")) for e in events)
