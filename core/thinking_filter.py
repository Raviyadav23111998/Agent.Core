"""
Streaming thinking-tag splitter
═══════════════════════════════
Separates a model's inline chain-of-thought from its actual answer while the
response streams.

Reasoning models emit thinking on a dedicated channel (OpenRouter
`delta.reasoning` / `delta.reasoning_content`) which the LLM client already
routes to the UI's "thinking" block. But MANY models (and some prompt styles)
emit their reasoning as ordinary answer content wrapped in tags like
``<think>…</think>``. Without separation that reasoning leaks straight into the
answer bubble — the #1 "this doesn't look like a real AI agent" complaint.

This filter consumes ``content`` deltas and returns, per chunk, the text that is
genuinely the answer vs the text that is reasoning, so the caller can emit each
on the right channel. It is a small state machine that buffers a partial tag
across chunk boundaries (a tag can be split between two SSE deltas).

Usage:
    f = ThinkingSplitter()
    for chunk in stream:
        answer, reasoning = f.feed(chunk)
        ...
    answer, reasoning = f.flush()   # at stream end

Supported tags (case-insensitive): <think>…</think>, <thinking>…</thinking>.
Unclosed thinking at stream end is treated as reasoning (never silently dropped
into the answer).
"""

from __future__ import annotations

from typing import List, Tuple

_OPEN = ("<think>", "<thinking>")
_CLOSE = ("</think>", "</thinking>")


def _find_first(hay_lower: str, markers: Tuple[str, ...]) -> Tuple[int, str]:
    """Earliest occurrence of any marker; (-1, '') if none present."""
    best_pos, best_marker = -1, ""
    for m in markers:
        p = hay_lower.find(m)
        if p != -1 and (best_pos == -1 or p < best_pos):
            best_pos, best_marker = p, m
    return best_pos, best_marker


def _partial_suffix_len(hay_lower: str, markers: Tuple[str, ...]) -> int:
    """Length of the longest suffix of `hay_lower` that is a *proper* prefix of
    some marker — i.e. text we must hold back because it might be the start of a
    tag completed by the next chunk."""
    max_k = 0
    n = len(hay_lower)
    for m in markers:
        limit = min(len(m) - 1, n)
        for k in range(limit, 0, -1):
            if hay_lower.endswith(m[:k]):
                if k > max_k:
                    max_k = k
                break
    return max_k


class ThinkingSplitter:
    """Stateful splitter — one instance per streamed response."""

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, text: str) -> Tuple[str, str]:
        """Process a content chunk → (answer_text, reasoning_text)."""
        if text:
            self._buf += text
        answer: List[str] = []
        reasoning: List[str] = []

        while self._buf:
            markers = _CLOSE if self._in_think else _OPEN
            low = self._buf.lower()
            pos, marker = _find_first(low, markers)

            if pos == -1:
                hold = _partial_suffix_len(low, markers)
                cut = len(self._buf) - hold
                emit = self._buf[:cut]
                self._buf = self._buf[cut:]
                if emit:
                    (reasoning if self._in_think else answer).append(emit)
                break

            before = self._buf[:pos]
            if before:
                (reasoning if self._in_think else answer).append(before)
            self._buf = self._buf[pos + len(marker):]
            self._in_think = not self._in_think

        return "".join(answer), "".join(reasoning)

    def flush(self) -> Tuple[str, str]:
        """Emit any buffered remainder at stream end. An unclosed <think> block
        is returned as reasoning so it never leaks into the answer."""
        rest, self._buf = self._buf, ""
        if not rest:
            return "", ""
        return ("", rest) if self._in_think else (rest, "")
