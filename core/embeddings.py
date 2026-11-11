"""
Embedding Client — dense vector embeddings for semantic retrieval.
═══════════════════════════════════════════════════════════════════

Wraps an OpenAI-compatible `/embeddings` endpoint, reusing the same config
pattern as the chat LLM. This is the foundation for the Phase-1 accuracy leap:
semantic (vector) retrieval to complement the existing BM25 + graph search.

Design:
- Config-driven, lazily initialized.
- Graceful degradation: if no embedding model is configured, `available` is
  False and callers fall back to keyword search. NOTHING breaks.
- Sync HTTP (requests) run off the event loop by callers via asyncio.to_thread,
  mirroring GenericLLMClient.

Config keys (ag_system.configs / env):
    llm.embedding_model      e.g. "text-embedding-3-small" / "openai/text-embedding-3-small"
    llm.embedding_base_url   (defaults to llm.base_url)
    llm.embedding_api_key    (defaults to llm.api_key)
    llm.embedding_dimensions (optional int, for models that support it)
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

logger = logging.getLogger("agent.embeddings")


class EmbeddingClient:
    """OpenAI-compatible embeddings client with graceful degradation."""

    def __init__(self, model: str = None, base_url: str = None,
                 api_key: str = None, dimensions: int = None):
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._dimensions = dimensions


    def _cfg(self, key: str, default=None):
        try:
            from core.config import app_config
            return app_config.get(key, default)
        except Exception:
            return default

    @property
    def model(self) -> Optional[str]:
        return self._model or self._cfg("llm.embedding_model")

    @property
    def base_url(self) -> Optional[str]:
        url = self._base_url or self._cfg("llm.embedding_base_url") or self._cfg("llm.base_url")
        if url and not url.rstrip("/").endswith("/embeddings"):
            url = url.rstrip("/")
            if url.endswith("/chat/completions"):
                url = url[: -len("/chat/completions")] + "/embeddings"
            else:
                url = url + "/embeddings"
        return url

    @property
    def api_key(self) -> Optional[str]:
        return self._api_key or self._cfg("llm.embedding_api_key") or self._cfg("llm.api_key")

    @property
    def dimensions(self) -> Optional[int]:
        d = self._dimensions or self._cfg("llm.embedding_dimensions")
        return int(d) if d else None

    @property
    def available(self) -> bool:
        """True when a model + endpoint are configured. The API key is OPTIONAL:
        local/self-hosted providers (e.g. Ollama at localhost:11434) are keyless,
        so requiring a key here wrongly disabled the entire vector arm for them —
        documents silently never got embedded. When a key IS configured it is
        still sent (see embed()); hosted endpoints that need one will 401 and we
        degrade to keyword search rather than failing ingestion."""
        return bool(self.model and self.base_url)


    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns [] if not configured or on error.

        Callers MUST handle the empty-list (degraded) case by falling back
        to keyword search.
        """
        if not texts or not self.available:
            return []
        import requests

        payload = {"model": self.model, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            return [it.get("embedding", []) for it in items]
        except Exception as e:
            logger.warning("embeddings.failed: %s", e)
            return []

    def embed_one(self, text: str) -> List[float]:
        """Embed a single text. Returns [] if unavailable/error."""
        out = self.embed([text])
        return out[0] if out else []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 for empty/degenerate input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
