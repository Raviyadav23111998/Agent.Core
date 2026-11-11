"""
Guard pipeline — the composable security-decision MECHANISM for tool execution.
═══════════════════════════════════════════════════════════════════════════════
This module is the *mechanism* (how guards are evaluated). The *policy config*
(what the rules are, read from a persona/integration) lives in core/guardrails.py
and builds concrete guards from this toolkit. Splitting mechanism from policy is
the whole point: adding a guard or an operator becomes a small, unit-testable
ADDITION — never an edit to a god-method full of if/elif.

Senior-engineer design notes (this is the LLM-containment boundary, so these
properties matter more than aesthetics):

  * STRUCTURED decisions. Every gate returns a `Decision`, not a bare bool, so
    the outcome is auditable: allow/block/approval, the reason, which guard
    fired, and a severity. The agent loop and audit log consume this uniformly.

  * RICH context. Guards receive a `GuardContext` (tool, args, mode, side-effect
    policy, persona/tenant). A future guard — cost cap, rate limit, PII, tenant
    scope — reads ctx and slots into the chain without touching any caller.

  * FAIL CLOSED. A guard that raises is treated as a BLOCK, never a crash that
    aborts the turn and never a silent allow. A misconfigured policy denies.

  * DETERMINISTIC. Guards run in a fixed order; the first *terminal* decision
    (block or approval gate) wins and short-circuits. Order is explicit.

  * NO LLM IN THE HOT PATH. Pre-execution gating is pure code. The semantic
    OUTPUT filter is a separate, explicit, async stage — not part of this chain.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger("agent.guardrails")


@dataclass(frozen=True)
class Decision:
    """The structured outcome of a guard check.

    allow=True              → permitted, continue the chain.
    allow=False             → blocked (terminal).
    needs_approval=True     → not allowed as-is, but a human MAY approve (terminal).
    """
    allow: bool
    needs_approval: bool = False
    reason: str = ""
    guard: str = ""
    severity: str = "info"

    @property
    def terminal(self) -> bool:
        """A decision that stops the pipeline (a block or an approval gate)."""
        return (not self.allow) or self.needs_approval

    @staticmethod
    def ok() -> "Decision":
        return Decision(allow=True)

    @staticmethod
    def block(reason: str, guard: str = "", severity: str = "block") -> "Decision":
        return Decision(allow=False, reason=reason, guard=guard, severity=severity)

    @staticmethod
    def approve(reason: str, guard: str = "") -> "Decision":
        return Decision(allow=False, needs_approval=True, reason=reason,
                        guard=guard, severity="warn")


ALLOW = Decision.ok()


@dataclass
class GuardContext:
    """Everything a guard might need to make a decision. Extend `extra` rather
    than the signature so adding a field never breaks existing guards."""
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    mode: Optional[str] = None
    policy: Any = None
    persona_id: Optional[int] = None
    tenant_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Guard(Protocol):
    """A single security check. Stateless w.r.t. the turn; config is captured at
    construction. `name` is used in logs/audit and on fail-closed."""
    name: str

    def check(self, ctx: GuardContext) -> Decision: ...


def run_guards(guards: List[Guard], ctx: GuardContext) -> Decision:
    """Run guards in order; the first terminal decision wins. Fail CLOSED.

    A guard raising is a bug in policy evaluation, not licence to proceed — we
    convert it to a block so the LLM can never slip through a broken check.
    """
    for g in guards:
        try:
            decision = g.check(ctx)
        except Exception as e:
            gname = getattr(g, "name", g.__class__.__name__)
            logger.error("Guard '%s' raised on tool '%s': %s — failing closed",
                         gname, ctx.tool_name, e)
            return Decision.block(
                f"A security check could not be evaluated for '{ctx.tool_name}'. "
                f"The action was blocked.", guard=gname, severity="block")
        if decision.terminal:
            if decision.severity == "block":
                logger.warning("GUARDRAIL BLOCKED [%s] %s: %s",
                               decision.guard, ctx.tool_name, decision.reason)
            return decision
    return ALLOW


@dataclass(frozen=True)
class Operator:
    """One IAM-style argument operator: a predicate + a human reason builder."""
    test: Callable[[Any, Any], bool]
    message: Callable[[str, Any, Any], str]


EXECUTION_OPERATORS: Dict[str, Operator] = {}


def operator(name: str, message: Callable[[str, Any, Any], str]):
    """Register an execution-policy operator. Adding `$gt`, `$contains`, etc. is
    a one-line decorator here — no edit to the policy evaluator."""
    def deco(test: Callable[[Any, Any], bool]):
        EXECUTION_OPERATORS[name] = Operator(test=test, message=message)
        return test
    return deco


@operator("$in", lambda a, s, v: f"Policy violation: '{a}' must be one of {s}, got '{v}'.")
def _op_in(value, spec) -> bool:
    return value in spec


@operator("$not_in", lambda a, s, v: f"Policy violation: '{v}' is a blocked value for '{a}'.")
def _op_not_in(value, spec) -> bool:
    return value not in spec


@operator("$regex", lambda a, s, v: f"Policy violation: '{a}' does not match the required format.")
def _op_regex(value, spec) -> bool:
    return re.fullmatch(spec, str(value)) is not None


@operator("$max_length", lambda a, s, v: f"Policy violation: '{a}' exceeds maximum length of {s}.")
def _op_max_length(value, spec) -> bool:
    return len(str(value)) <= spec


def evaluate_arg_rules(tool_name: str, rule: Dict[str, Any], arg_name: str,
                       value: Any) -> Decision:
    """Evaluate one argument's operator rules against a value. Fail CLOSED on a
    broken operator spec (e.g. invalid regex) — a misconfigured rule must deny,
    not grant. Unknown `$operators` are skipped (forward-compatible configs)."""
    for op_name, spec in rule.items():
        if not op_name.startswith("$"):
            continue
        op = EXECUTION_OPERATORS.get(op_name)
        if op is None:
            logger.debug("Unknown execution operator '%s' on %s.%s — skipped",
                         op_name, tool_name, arg_name)
            continue
        try:
            ok = op.test(value, spec)
        except Exception as e:
            logger.error("Operator %s misconfigured on %s.%s (%s) — blocking",
                         op_name, tool_name, arg_name, e)
            return Decision.block(
                f"Policy error: the security rule for '{arg_name}' is "
                f"misconfigured. Contact your administrator.",
                guard="execution_policy", severity="block")
        if not ok:
            return Decision.block(op.message(arg_name, spec, value),
                                  guard="execution_policy", severity="block")
    return ALLOW
