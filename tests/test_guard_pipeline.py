"""
Phase 1 — guard pipeline mechanism + guardrails refactor.

Covers: Decision semantics, the operator registry, fail-closed on broken
operators, run_guards ordering/fail-closed, and that the refactored
GuardrailEngine.check_execution / check_embed_action / evaluate preserve the
documented behaviour (whitelist/blacklist/$deny/$in/$regex, danger→approval).
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.guard_pipeline import (
    Decision, GuardContext, EXECUTION_OPERATORS, evaluate_arg_rules,
    run_guards, operator,
)
from core.guardrails import GuardrailEngine


def test_decision_terminal_semantics():
    assert Decision.ok().allow and not Decision.ok().terminal
    assert Decision.block("no").terminal and not Decision.block("no").allow
    appr = Decision.approve("confirm?")
    assert appr.terminal and appr.needs_approval and not appr.allow


def test_operators_registered():
    for op in ("$in", "$not_in", "$regex", "$max_length"):
        assert op in EXECUTION_OPERATORS


def test_evaluate_arg_rules_in_and_regex():
    assert evaluate_arg_rules("t", {"$in": ["Open", "Closed"]}, "status", "Open").allow
    assert not evaluate_arg_rules("t", {"$in": ["Open"]}, "status", "Closed").allow
    assert evaluate_arg_rules("t", {"$regex": "Closed"}, "s", "Closed").allow
    assert not evaluate_arg_rules("t", {"$regex": "Closed"}, "s", "Closed-EVIL").allow


def test_evaluate_arg_rules_fail_closed_on_bad_regex():
    d = evaluate_arg_rules("t", {"$regex": "([unterminated"}, "s", "x")
    assert not d.allow and "misconfigured" in d.reason.lower()


def test_unknown_operator_is_skipped_not_blocked():
    assert evaluate_arg_rules("t", {"$future_op": 1}, "s", "x").allow


def test_operator_is_extensible_via_decorator():
    @operator("$gt_test", lambda a, s, v: f"{a} must be > {s}")
    def _gt(value, spec):
        return value > spec
    try:
        assert evaluate_arg_rules("t", {"$gt_test": 5}, "n", 7).allow
        assert not evaluate_arg_rules("t", {"$gt_test": 5}, "n", 3).allow
    finally:
        EXECUTION_OPERATORS.pop("$gt_test", None)


def test_run_guards_first_terminal_wins_and_fails_closed():
    class Allow:
        name = "allow"
        def check(self, ctx): return Decision.ok()

    class Block:
        name = "block"
        def check(self, ctx): return Decision.block("nope", guard="block")

    class Boom:
        name = "boom"
        def check(self, ctx): raise RuntimeError("kaboom")

    ctx = GuardContext(tool_name="x")
    assert run_guards([Allow(), Allow()], ctx).allow
    assert run_guards([Allow(), Block(), Boom()], ctx).reason == "nope"
    d = run_guards([Boom()], ctx)
    assert not d.allow and not d.needs_approval


def test_execution_whitelist_blocks_unlisted():
    eng = GuardrailEngine({"execution_policies": {"$mode": "whitelist",
                                                  "brain_ask": {}}})
    assert eng.check_execution("brain_ask", {})[0] is True
    assert eng.check_execution("run_command", {"cmd": "rm"})[0] is False


def test_execution_blacklist_deny_and_argrules():
    eng = GuardrailEngine({"execution_policies": {
        "danger_tool": {"$deny": True},
        "jira_update": {"status": {"$in": ["Open", "Closed"]}},
    }})
    assert eng.check_execution("anything_else", {})[0] is True
    assert eng.check_execution("danger_tool", {})[0] is False
    assert eng.check_execution("jira_update", {"status": "Open"})[0] is True
    assert eng.check_execution("jira_update", {"status": "Deleted"})[0] is False


def test_embed_action_danger_needs_approval():
    eng = GuardrailEngine({}, embed_execution={})
    ok, needs, _ = eng.check_embed_action(
        "page_act", {"steps": [{"do": "click", "target": "Delete account"}]})
    assert ok is False and needs is True
    ok2, _, _ = eng.check_embed_action(
        "page_act", {"steps": [{"do": "click", "target": "Save"}]})
    assert ok2 is True


def test_evaluate_unifies_into_decision():
    eng = GuardrailEngine({"execution_policies": {"$mode": "whitelist",
                                                  "brain_ask": {}}})
    assert eng.evaluate(GuardContext(tool_name="brain_ask")).allow
    blocked = eng.evaluate(GuardContext(tool_name="run_command", args={"c": "x"}))
    assert not blocked.allow and blocked.guard == "execution_policy"

    eng2 = GuardrailEngine({}, embed_execution={})
    appr = eng2.evaluate(GuardContext(tool_name="page_navigate",
                                      args={"route": "/delete-account"}))
    assert appr.needs_approval and appr.guard == "embed_action"
    assert eng2.evaluate(GuardContext(tool_name="page_navigate",
                                      args={"route": "/dashboard"})).allow


import asyncio


def test_executor_gate_blocks_on_execution_policy():
    from core.tool_executor import LocalToolExecutor
    from tools.tool_policy import default_policy_for

    class Reg:
        def execute(self, n, a): return "ran"
        def get_policy(self, n): return default_policy_for(n)

    eng = GuardrailEngine({"execution_policies": {"$mode": "whitelist",
                                                  "brain_ask": {}}})
    ex = LocalToolExecutor(Reg(), guardrail_engine=eng)

    async def run():
        assert "GUARDRAIL BLOCKED" in await ex.execute("run_command", {"c": "x"})
        assert await ex.execute("brain_ask", {}) == "ran"
    asyncio.run(run())


def test_executor_gate_embed_action_needs_approval():
    from core.tool_executor import RemoteToolExecutor
    from core.approvals import ApprovalManager
    from async_bus.event_bus import EventBus

    class Reg:
        def get_policy(self, n): return None

    eng = GuardrailEngine({}, embed_execution={})

    async def run():
        mgr = ApprovalManager(EventBus(), user_id=1, mode="irreversible", timeout_s=0.1)
        ex = RemoteToolExecutor(websocket=None, tool_registry=Reg(),
                                guardrail_engine=eng, approval_manager=mgr)
        res = await ex.execute("page_navigate", {"route": "/delete-account"})
        assert "APPROVAL DENIED" in res
    asyncio.run(run())


def test_gate_block_messages_count_as_loop_failures():
    from core.agent_loop import AgentLoop

    class Fake:
        _GATE_BLOCK_PREFIXES = AgentLoop._GATE_BLOCK_PREFIXES
        _PAGE_INTERACTION_TOOLS = AgentLoop._PAGE_INTERACTION_TOOLS
        _SOFT_FAILURE_MARKERS = AgentLoop._SOFT_FAILURE_MARKERS

    f = Fake()
    assert AgentLoop._is_failure_result(f, "x", "[GUARDRAIL BLOCKED] nope")
    assert AgentLoop._is_failure_result(f, "x", "[APPROVAL DENIED] nope")
    assert AgentLoop._is_failure_result(f, "x", "[DUPLICATE BLOCKED] nope")
    assert not AgentLoop._is_failure_result(f, "x", "OK done")
