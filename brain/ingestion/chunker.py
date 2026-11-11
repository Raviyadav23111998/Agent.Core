"""
Document chunker — parent-child retrieval foundation.
═══════════════════════════════════════════════════════

The agent historically embedded the WHOLE document as a single vector
(truncated to ~8k chars), which dilutes meaning and loses anything past the
cutoff — poor recall on long PDFs / manuals / policies.

This splits a document into small, overlapping CHILD chunks. Each chunk is
embedded and indexed separately so vector search matches the precise passage;
retrieval then returns the PARENT document for full context (parent-child).

Strategy (dependency-free):
  - Split on paragraph boundaries first (preserve semantic units).
  - Pack paragraphs into windows of ~`chunk_size` chars.
  - A paragraph larger than the window is hard-split on sentence boundaries.
  - Consecutive windows overlap by ~`overlap` chars so a fact straddling a
    boundary still lands wholly inside at least one chunk.

Sizes are in CHARACTERS (≈ chars/4 tokens) to stay model-agnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_OVERLAP = 320
MIN_CHUNK_SIZE = 80
MAX_CHUNKS_PER_DOC = 400

_PARA_SPLIT = re.compile(r"\n\s*\n+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    index: int
    content: str


def _hard_split(text: str, size: int) -> List[str]:
    """Split an over-long block on sentence boundaries, packing to `size`."""
    out: List[str] = []
    buf = ""
    for sent in _SENT_SPLIT.split(text):
        if not sent:
            continue
        if len(buf) + len(sent) + 1 > size and buf:
            out.append(buf.strip())
            buf = sent
        elif len(sent) > size:
            if buf:
                out.append(buf.strip())
                buf = ""
            for i in range(0, len(sent), size):
                out.append(sent[i:i + size].strip())
        else:
            buf = f"{buf} {sent}".strip() if buf else sent
    if buf.strip():
        out.append(buf.strip())
    return out


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_OVERLAP) -> List[Chunk]:
    """Split `text` into overlapping child chunks. Empty input → []."""
    if not text or not text.strip():
        return []
    chunk_size = max(int(chunk_size), MIN_CHUNK_SIZE * 2)
    overlap = max(min(int(overlap), chunk_size // 2), 0)

    pieces: List[str] = []
    for para in _PARA_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            pieces.append(para)
        else:
            pieces.extend(_hard_split(para, chunk_size))

    windows: List[str] = []
    buf = ""
    for piece in pieces:
        if len(buf) + len(piece) + 2 > chunk_size and buf:
            windows.append(buf.strip())
            buf = (buf[-overlap:] + "\n\n" + piece) if overlap else piece
        else:
            buf = f"{buf}\n\n{piece}".strip() if buf else piece
    if buf.strip():
        windows.append(buf.strip())

    chunks = [Chunk(index=i, content=w)
              for i, w in enumerate(windows)
              if len(w.strip()) >= MIN_CHUNK_SIZE]
    if not chunks and text.strip():
        chunks = [Chunk(index=0, content=text.strip())]
    return chunks[:MAX_CHUNKS_PER_DOC]
