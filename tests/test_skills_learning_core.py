"""Skills + learning loop — core substrate (P1).

Fake-store level: tools contract, prompt index byte-stability, review-signal
observation, builtin reviewer/curator shape.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.skills_mcp import SkillsMCPTools, current_skill_tracker
from tools.native_mcp import NativeMCP


class FakeStore:
    def __init__(self):
        self.skills = {
            "email-triage": {
                "name": "email-triage",
                "description": "Classify and route inbound mail.",
                "tags": ["email"],
                "body": "# Email triage\nSteps...",
                "files": [{"kind": "reference", "path": "references/quirks.md"}],
            },
        }
        self.writes = []
        self.uses = []

    def index(self):
        return [{k: v for k, v in s.items() if k != "body"}
                for s in self.skills.values()]

    def get(self, name):
        return self.skills.get(name)

    def get_file(self, name, path):
        return "quirk content" if name in self.skills else None

    def write(self, action, name, **kw):
        self.writes.append((action, name, kw))
        return f"{action} ok"

    def record_use(self, name):
        self.uses.append(name)


def _registry_with(store):
    mcp = NativeMCP()
    SkillsMCPTools(store_resolver=lambda: store).register_tools(mcp)
    return mcp


def _call(mcp, tool, **args):
    return mcp._tools[tool](**args)


def test_skills_tools_without_store_degrade():
    mcp = _registry_with(None)
    out = _call(mcp, "skills_list")
    assert "not available" in out


def test_skill_view_tracks_in_play_and_use():
    store = FakeStore()
    mcp = _registry_with(store)
    tracker = set()
    token = current_skill_tracker.set(tracker)
    try:
        out = _call(mcp, "skill_view", name="email-triage")
    finally:
        current_skill_tracker.reset(token)
    assert "Email triage" in out
    assert "references/quirks.md" in out       # support files listed
    assert store.uses == ["email-triage"]      # effectiveness signal
    assert tracker == {"email-triage"}         # in-play marking


def test_skill_manage_actions_validated():
    store = FakeStore()
    mcp = _registry_with(store)
    assert "Unknown action" in _call(mcp, "skill_manage",
                                     action="explode", name="x")
    out = _call(mcp, "skill_manage", action="patch", name="email-triage",
                body="## Pitfall\nnew lesson")
    assert "ok" in out
    assert store.writes[0][0] == "patch"


def test_prompt_index_byte_stable_and_sorted():
    from core.prompt_builder import PromptBuilder
    from core.prompt_sections import (render_skills_index, SKILLS_KEY,
                                      SKILLS_ORDER)
    pb = PromptBuilder("/tmp")
    rows = [{"name": "zeta", "description": "Z."},
            {"name": "alpha", "description": "A."}]
    pb.set_section(SKILLS_KEY, render_skills_index(rows), SKILLS_ORDER)
    p1 = pb.build_system_prompt()
    # order must not matter
    pb.set_section(SKILLS_KEY, render_skills_index(list(reversed(rows))),
                   SKILLS_ORDER)
    p2 = pb.build_system_prompt()
    assert p1 == p2
    assert "<skills_index>" in p1
    assert p1.index("alpha") < p1.index("zeta")
    pb.set_section(SKILLS_KEY, render_skills_index([]), SKILLS_ORDER)
    assert "<skills_index>" not in pb.build_system_prompt()


def test_review_signals_observed_and_gated():
    from core.agent_loop import AgentLoop
    loop = AgentLoop.__new__(AgentLoop)
    loop._skills_in_play = {"email-triage"}
    loop._turn_review_signals = {}
    loop._last_turn_iterations = 3
    import types
    loop.user_context = types.SimpleNamespace(user_id="user_5")

    loop._observe_review_signals("actually, don't format it like that")
    assert loop._turn_review_signals["correction"] is True
    sig = loop.collect_review_signals()
    assert sig["should_review"] is True
    assert sig["skills_in_play"] == []          # cleared at turn start

    # Machine-originated turns never signal.
    loop._observe_review_signals("[AGENT:email-bot] never process twice")
    assert loop._turn_review_signals == {}


def test_builtin_reviewer_and_curator():
    from agents.factory import BUILTIN_AGENTS, _builtin_system_prompt
    r = BUILTIN_AGENTS["reviewer"]
    assert set(r.tools) == {"skills_list", "skill_view", "skill_manage",
                            "memory_store", "memory_recall"}
    prompt = _builtin_system_prompt("reviewer")
    assert "Nothing to save." in prompt
    assert "DO NOT CAPTURE" in prompt
    c = BUILTIN_AGENTS["curator"]
    assert "skill_manage" in c.tools and "memory_store" not in c.tools


def test_curator_prompt_loads():
    from agents.factory import _builtin_system_prompt
    p = _builtin_system_prompt("curator")
    assert "Library healthy" in p                 # the clean-run sentinel
    assert "NEVER archive or gut a pinned" in p    # protected-skill rule
    assert "at most FIVE changes" in p             # bounded churn


def test_factory_zero_tools_returns_runresult():
    """A definition whose tools don't resolve must return a RunResult with
    an error status — never a bare string (every caller does .output)."""
    from agents.factory import (AgentFactory, RunResult, AgentDefinition)

    class _EmptyReg:
        _tools = {}; _callbacks = {}; _policies = {}

    class _Cli:
        tool_registry = _EmptyReg(); workspace_path = "/tmp"
        llm = None; llm_scope = None

    class _Reg:
        def update_run_state(self, *a, **k): pass
        def save_run_log(self, *a, **k): pass

    f = AgentFactory(_Cli(), _Reg())
    d = AgentDefinition(name="ghost", description="x", tools=["nope"])
    res = asyncio.run(f.run_agent(d, task_prompt="hi"))
    assert isinstance(res, RunResult)
    assert res.status == "error"
    assert isinstance(res.output, str)
