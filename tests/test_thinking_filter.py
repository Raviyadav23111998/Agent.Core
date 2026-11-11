"""Tests for the streaming thinking-tag splitter (core.thinking_filter)."""

from core.thinking_filter import ThinkingSplitter


def _drain(chunks):
    """Feed chunks through a fresh splitter; return (answer, reasoning)."""
    f = ThinkingSplitter()
    ans, rea = [], []
    for c in chunks:
        a, r = f.feed(c)
        ans.append(a)
        rea.append(r)
    a, r = f.flush()
    ans.append(a)
    rea.append(r)
    return "".join(ans), "".join(rea)


def test_no_tags_passthrough():
    ans, rea = _drain(["Hello ", "world."])
    assert ans == "Hello world."
    assert rea == ""


def test_single_block_in_one_chunk():
    ans, rea = _drain(["<think>plan it</think>The answer."])
    assert ans == "The answer."
    assert rea == "plan it"


def test_tag_split_across_chunks():
    ans, rea = _drain(["Before <th", "ink>secret</thi", "nk>After"])
    assert ans == "Before After"
    assert rea == "secret"


def test_reasoning_first_then_answer():
    ans, rea = _drain(["<thinking>", "step 1", " step 2", "</thinking>", "Done"])
    assert ans == "Done"
    assert rea == "step 1 step 2"


def test_multiple_blocks():
    ans, rea = _drain(["a<think>x</think>b<think>y</think>c"])
    assert ans == "abc"
    assert rea == "xy"


def test_unclosed_thinking_is_reasoning_not_answer():
    ans, rea = _drain(["visible<think>dangling reasoning"])
    assert ans == "visible"
    assert rea == "dangling reasoning"


def test_case_insensitive():
    ans, rea = _drain(["<THINK>R</THINK>A"])
    assert ans == "A"
    assert rea == "R"


def test_char_by_char_streaming():
    src = "Hi <think>quietly</think>there"
    ans, rea = _drain(list(src))
    assert ans == "Hi there"
    assert rea == "quietly"


def test_partial_false_alarm_emitted():
    ans, rea = _drain(["a < b ", "and c"])
    assert ans == "a < b and c"
    assert rea == ""
