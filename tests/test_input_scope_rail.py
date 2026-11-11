"""
Input scope rail — the pre-generation topic gate on GuardrailEngine.

Covers: activation gating (mode=restrict + domain), OUT→block with the
persona's refusal, IN→allow, unparseable/error verdicts honoring fail_open vs
fail_closed, and that an unscoped persona never calls the classifier.
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.guardrails import GuardrailEngine


class _FakeLLM:
    """Minimal stand-in for the agent's LLM client: records calls and returns
    a scripted verdict (or raises)."""

    def __init__(self, verdict="IN", raise_exc=None):
        self.verdict = verdict
        self.raise_exc = raise_exc
        self.calls = 0

    async def generate(self, prompt, history, temperature=None, max_tokens=None):
        self.calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        return {"content": self.verdict}


def _run(coro):
    # A fresh, isolated loop per call — robust inside the full suite where an
    # earlier test may have closed the shared default event loop (get_event_loop
    # + run_until_complete would then raise). These test fns are sync, so no
    # loop is already running.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _persona(**scope):
    return {"scope_policy": {"mode": "restrict",
                             "domain": "Fleet Management",
                             "refusal_message": "Fleet only.",
                             **scope}}


def test_scope_inactive_without_restrict():
    eng = GuardrailEngine({"scope_policy": {"mode": "off", "domain": "X"}},
                          llm_client=_FakeLLM("OUT"))
    assert eng.has_input_scope is False
    # Inactive rail must never call the classifier and always allow.
    allowed, refusal = _run(eng.evaluate_input("write me a script"))
    assert allowed is True and refusal == ""
    assert eng._llm.calls == 0


def test_scope_inactive_without_domain():
    eng = GuardrailEngine({"scope_policy": {"mode": "restrict"}},
                          llm_client=_FakeLLM("OUT"))
    assert eng.has_input_scope is False


def test_out_of_scope_blocks_with_persona_refusal():
    llm = _FakeLLM("OUT")
    eng = GuardrailEngine(_persona(), llm_client=llm)
    assert eng.has_input_scope is True
    allowed, refusal = _run(eng.evaluate_input("write a python script for me"))
    assert allowed is False
    assert refusal == "Fleet only."
    assert llm.calls == 1


def test_in_scope_allows():
    llm = _FakeLLM("IN")
    eng = GuardrailEngine(_persona(), llm_client=llm)
    allowed, refusal = _run(eng.evaluate_input("add a new vehicle",
                                               recent_context="user: hi"))
    assert allowed is True and refusal == ""
    assert llm.calls == 1


def test_verdict_prefix_tolerated():
    # Model may answer "OUT OF SCOPE" / "IN SCOPE" — prefix match must hold.
    eng = GuardrailEngine(_persona(), llm_client=_FakeLLM("OUT OF SCOPE"))
    allowed, _ = _run(eng.evaluate_input("what's the weather"))
    assert allowed is False


def test_blank_input_short_circuits_allow():
    llm = _FakeLLM("OUT")
    eng = GuardrailEngine(_persona(), llm_client=llm)
    allowed, refusal = _run(eng.evaluate_input("   "))
    assert allowed is True and refusal == ""
    assert llm.calls == 0


def test_unparseable_verdict_fail_open_default():
    eng = GuardrailEngine(_persona(), llm_client=_FakeLLM("maybe?"))
    allowed, _ = _run(eng.evaluate_input("something"))
    assert allowed is True  # fail_open defaults true


def test_unparseable_verdict_fail_closed():
    eng = GuardrailEngine(_persona(fail_open=False),
                          llm_client=_FakeLLM("maybe?"))
    allowed, refusal = _run(eng.evaluate_input("something"))
    assert allowed is False and refusal == "Fleet only."


def test_classifier_error_fail_open():
    eng = GuardrailEngine(_persona(), llm_client=_FakeLLM(raise_exc=RuntimeError("boom")))
    allowed, _ = _run(eng.evaluate_input("something"))
    assert allowed is True


def test_classifier_error_fail_closed():
    eng = GuardrailEngine(_persona(fail_open=False),
                          llm_client=_FakeLLM(raise_exc=RuntimeError("boom")))
    allowed, refusal = _run(eng.evaluate_input("something"))
    assert allowed is False and refusal == "Fleet only."


def test_default_refusal_when_persona_has_none():
    persona = {"scope_policy": {"mode": "restrict", "domain": "Fleet Management"}}
    eng = GuardrailEngine(persona, llm_client=_FakeLLM("OUT"))
    allowed, refusal = _run(eng.evaluate_input("write code"))
    assert allowed is False
    assert refusal == GuardrailEngine._DEFAULT_SCOPE_REFUSAL


def test_scope_policy_accepts_json_string():
    # Defensive: a driver that returns JSONB as text must still parse.
    import json
    persona = {"scope_policy": json.dumps(
        {"mode": "restrict", "domain": "Fleet Management"})}
    eng = GuardrailEngine(persona, llm_client=_FakeLLM("OUT"))
    assert eng.has_input_scope is True
