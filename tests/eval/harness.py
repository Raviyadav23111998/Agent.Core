"""
Eval Harness — regression evaluation for agent quality.
═══════════════════════════════════════════════════════════

A lightweight, dependency-free harness to run golden tasks against the agent
and score outcomes — so prompt/model/retrieval changes can be measured instead
of guessed. Designed to run in CI (offline cases) and locally (live cases).

Two scorer families:
  - deterministic: substring / regex / JSON-field assertions on the answer.
  - llm_judge: an LLM rates the answer against a rubric (0-1). Optional —
    only runs when a judge client is supplied.

Cases live in tests/eval/cases/*.json:
  {
    "id": "greet-trivial",
    "input": "hello",
    "expect": {"contains_any": ["hi", "hello", "hey"]},
    "tags": ["smoke"]
  }

Usage (offline scoring of a precomputed answer):
    from tests.eval.harness import score_case
    result = score_case(case, answer_text)

Usage (full run against a callable agent):
    runner = EvalRunner(run_agent=async_fn)        # run_agent(input)->answer
    report = await runner.run(cases)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    score: float
    detail: str = ""
    answer: str = ""


@dataclass
class EvalReport:
    results: List[CaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def summary(self) -> str:
        lines = [f"Eval: {len(self.results)} cases | "
                 f"pass_rate={self.pass_rate:.0%} | mean_score={self.mean_score:.2f}"]
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{mark}] {r.case_id} ({r.score:.2f}) {r.detail}")
        return "\n".join(lines)


def score_case(case: Dict[str, Any], answer: str) -> CaseResult:
    """Score an answer against a case's `expect` block (deterministic)."""
    exp = case.get("expect", {})
    cid = case.get("id", "?")
    a = answer or ""
    a_low = a.lower()

    checks: List[bool] = []
    details: List[str] = []

    if "contains_any" in exp:
        hit = any(s.lower() in a_low for s in exp["contains_any"])
        checks.append(hit)
        if not hit:
            details.append(f"none of {exp['contains_any']} present")

    if "contains_all" in exp:
        missing = [s for s in exp["contains_all"] if s.lower() not in a_low]
        checks.append(not missing)
        if missing:
            details.append(f"missing {missing}")

    if "not_contains" in exp:
        bad = [s for s in exp["not_contains"] if s.lower() in a_low]
        checks.append(not bad)
        if bad:
            details.append(f"should not contain {bad}")

    if "regex" in exp:
        ok = re.search(exp["regex"], a) is not None
        checks.append(ok)
        if not ok:
            details.append(f"regex /{exp['regex']}/ no match")

    if "min_length" in exp:
        ok = len(a) >= exp["min_length"]
        checks.append(ok)
        if not ok:
            details.append(f"len {len(a)} < {exp['min_length']}")

    if not checks:
        checks.append(bool(a.strip()))

    passed = all(checks)
    score = sum(1 for c in checks if c) / len(checks)
    return CaseResult(cid, passed, score, "; ".join(details), a)


def load_cases(cases_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    cases_dir = cases_dir or os.path.join(os.path.dirname(__file__), "cases")
    cases: List[Dict[str, Any]] = []
    if not os.path.isdir(cases_dir):
        return cases
    for fn in sorted(os.listdir(cases_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(cases_dir, fn), encoding="utf-8") as f:
                data = json.load(f)
                cases.extend(data if isinstance(data, list) else [data])
    return cases


class EvalRunner:
    """Runs cases against a callable agent and scores the outputs."""

    def __init__(self, run_agent: Callable[[str], Awaitable[str]],
                 judge_client: Any = None):
        self._run_agent = run_agent
        self._judge = judge_client

    async def run(self, cases: List[Dict[str, Any]]) -> EvalReport:
        report = EvalReport()
        for case in cases:
            try:
                answer = await self._run_agent(case.get("input", ""))
            except Exception as e:
                report.results.append(CaseResult(case.get("id", "?"), False, 0.0,
                                                 f"agent error: {e}"))
                continue
            result = score_case(case, answer)
            if self._judge and "rubric" in case:
                try:
                    jscore = await self._llm_judge(case["rubric"], case.get("input", ""), answer)
                    result.score = (result.score + jscore) / 2
                    result.passed = result.passed and jscore >= 0.5
                except Exception:
                    pass
            report.results.append(result)
        return report

    async def _llm_judge(self, rubric: str, question: str, answer: str) -> float:
        sys_prompt = (
            "You are an evaluation judge. Score the ANSWER against the RUBRIC "
            "from 0.0 to 1.0. Reply with ONLY the number."
        )
        user = f"RUBRIC:\n{rubric}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}"
        resp = await self._judge.generate(sys_prompt, [{"role": "user", "content": user}])
        text = (resp or {}).get("content", "0").strip()
        m = re.search(r"[01](?:\.\d+)?", text)
        return float(m.group(0)) if m else 0.0
