"""Tests for the eval harness scoring + runner (offline)."""

import asyncio

from tests.eval.harness import score_case, load_cases, EvalRunner


def test_contains_any_pass_and_fail():
    case = {"id": "c", "expect": {"contains_any": ["hello", "hi"]}}
    assert score_case(case, "well, hello there").passed
    assert not score_case(case, "goodbye").passed


def test_contains_all_and_not_contains():
    case = {"id": "c", "expect": {"contains_all": ["a", "b"], "not_contains": ["secret"]}}
    assert score_case(case, "a and b").passed
    assert not score_case(case, "only a").passed
    assert not score_case(case, "a b secret").passed


def test_regex_and_min_length():
    case = {"id": "c", "expect": {"regex": r"\d{3}", "min_length": 5}}
    assert score_case(case, "code 123 ok").passed
    assert not score_case(case, "no digits").passed


def test_empty_answer_fails_default():
    assert not score_case({"id": "c", "expect": {}}, "").passed
    assert score_case({"id": "c", "expect": {}}, "non-empty").passed


def test_load_smoke_cases():
    cases = load_cases()
    ids = {c["id"] for c in cases}
    assert "greet-trivial" in ids


def test_runner_scores_against_stub_agent():
    cases = load_cases()

    async def stub_agent(inp):
        return "Hi there! I can help you with files, email, and search."

    async def run():
        runner = EvalRunner(run_agent=stub_agent)
        report = await runner.run(cases)
        return report

    report = asyncio.run(run())
    assert len(report.results) == len(cases)
    assert report.pass_rate >= 0.66
    assert "pass_rate" in report.summary()


def test_runner_handles_agent_errors():
    async def boom(inp):
        raise RuntimeError("model down")

    async def run():
        return await EvalRunner(run_agent=boom).run([{"id": "x", "input": "hi"}])

    report = asyncio.run(run())
    assert not report.results[0].passed
    assert "agent error" in report.results[0].detail
