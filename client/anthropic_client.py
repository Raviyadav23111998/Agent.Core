"""Native Anthropic Messages API client — the Claude Code engine contract.
═══════════════════════════════════════════════════════════════════════════

Why a native adapter instead of the OpenAI-compat shim:
  * THINKING PRESERVATION — extended-thinking blocks are returned, stored on
    the assistant message (`_thinking`), and replayed verbatim on subsequent
    iterations, so the model keeps its reasoning chain across tool calls
    instead of re-deriving it (fewer output tokens, better tool coherence).
  * STRUCTURED ERRORS — tool results carry `is_error` (from the loop's
    `_is_error` flag) instead of relying on "Error…" string sniffing.
  * RETRY DISCIPLINE — honors `Retry-After`, distinguishes overloaded_error,
    exponential backoff with jitter.
  * FIRST-CLASS CACHING — breakpoints on system, tools, and the compaction
    summary; cache read/write token accounting surfaced in usage.

Interface-compatible with GenericLLMClient.generate(): the AgentLoop and all
auxiliary callers (planner, compaction, salience, brain) work unchanged.
Selected via `llm.provider = anthropic`.
"""

import asyncio
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import requests

from core.config import app_config

logger = logging.getLogger("agent.llm.anthropic")

_ANTHROPIC_VERSION = "2023-06-01"


def _attachment_block(a: dict) -> Optional[Dict[str, Any]]:
    """Map a hydrated attachment ({kind, media_type, data_b64}) to an Anthropic
    content block. Images → `image` block, PDFs → `document` block.

    Attachments ride as a sibling `_attachments` field on the user message dict
    (NOT inside `content`, which the rest of core assumes is a str — mirrors how
    assistant messages carry `_thinking`). Kinds Claude can't accept (audio/video)
    return None; the API layer must have already warned the user per the selected
    model's capabilities (core.model_capabilities), so anything reaching here that
    doesn't map is simply skipped rather than corrupting the request.
    """
    if not a or not a.get("data_b64"):
        return None
    kind = a.get("kind")
    mt = a.get("media_type") or ""
    if kind == "image" or mt.startswith("image/"):
        return {"type": "image", "source": {"type": "base64",
                "media_type": mt or "image/png", "data": a["data_b64"]}}
    # Base64 document blocks are PDF-only; skip any other doc type rather than
    # sending a shape the API will 400 on.
    if mt == "application/pdf" or (kind in ("document", "pdf") and not mt):
        return {"type": "document", "source": {"type": "base64",
                "media_type": "application/pdf", "data": a["data_b64"]}}
    return None


class _RetryableError(Exception):
    def __init__(self, msg: str, retry_after: float = 0.0):
        super().__init__(msg)
        self.retry_after = retry_after


class AnthropicMessagesClient:
    """Production client for the native /v1/messages API."""

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}
    MAX_RETRIES = 4
    BASE_DELAY = 1.0
    MAX_RETRY_AFTER_S = 60.0

    def __init__(self, api_key: str = None, model: str = None,
                 url: str = None, stream: bool = True):
        self._api_key = api_key
        self._model = model
        self._url = url
        self._temperature = None
        self._max_tokens = None
        self.stream = stream
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    @property
    def api_key(self):
        return self._api_key or app_config.get("llm.api_key")

    @api_key.setter
    def api_key(self, value):
        self._api_key = value

    @property
    def model(self):
        return self._model or app_config.get("llm.model")

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def url(self):
        base = self._url or app_config.get("llm.base_url") or ""
        # A configured base_url belonging to an OpenAI-compatible setup
        # (…/chat/completions) must not leak into the native provider.
        if not base or "chat/completions" in base:
            base = "https://api.anthropic.com"
        base = base.rstrip('/')
        if not base.endswith('/v1/messages'):
            if base.endswith('/v1'):
                base += '/messages'
            else:
                base += '/v1/messages'
        return base

    @url.setter
    def url(self, value):
        self._url = value

    @property
    def temperature(self):
        if self._temperature is not None:
            return float(self._temperature)
        t = app_config.get("llm.temperature")
        return float(t) if t else 0.2

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    @property
    def max_tokens(self):
        if self._max_tokens is not None:
            return int(self._max_tokens)
        m = app_config.get("llm.max_tokens")
        return int(m) if m else 16000

    @max_tokens.setter
    def max_tokens(self, value):
        self._max_tokens = value

    @property
    def prompt_caching(self) -> bool:
        return True

    @property
    def thinking_budget(self) -> int:
        try:
            return app_config.get_int("llm.thinking_budget", 0) or 0
        except Exception:
            return 0

    def get_token_stats(self) -> str:
        total = self._total_input_tokens + self._total_output_tokens
        return (f"Input: {self._total_input_tokens} | "
                f"Output: {self._total_output_tokens} | Total: {total}")

    # ------------------------------------------------------------------
    # Request construction
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: Optional[list]) -> Optional[list]:
        """OpenAI function-tool format → Anthropic tool format."""
        if not tools:
            return None
        out = []
        for t in tools:
            fn = t.get("function", {}) if isinstance(t, dict) else {}
            out.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters")
                                or {"type": "object", "properties": {}},
            })
        if out:
            out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
        return out

    @staticmethod
    def _convert_history(history: list) -> list:
        """Loop history (OpenAI-shaped dicts) → Anthropic content blocks.

        - Consecutive user/tool messages merge into ONE user message
          (tool_result blocks first) — the API requires alternating roles.
        - Assistant messages replay stored `_thinking` blocks VERBATIM
          (signatures intact), then text, then tool_use blocks.
        - `_is_error` on a tool message becomes is_error on its tool_result.
        """
        messages: List[Dict[str, Any]] = []
        pending: List[Dict[str, Any]] = []

        def flush_user():
            nonlocal pending
            if pending:
                messages.append({"role": "user", "content": pending})
                pending = []

        for msg in history:
            role = msg.get("role", "user")
            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", "unknown"),
                    "content": str(msg.get("content") or ""),
                }
                if msg.get("_is_error"):
                    block["is_error"] = True
                pending.append(block)
            elif role == "assistant":
                flush_user()
                blocks: List[Dict[str, Any]] = []
                for tb in (msg.get("_thinking") or []):
                    if isinstance(tb, dict):
                        blocks.append(tb)
                content = msg.get("content")
                if isinstance(content, str) and content:
                    blocks.append({"type": "text", "text": content})
                for tc in (msg.get("tool_calls") or []):
                    fn = tc.get("function", {})
                    raw = fn.get("arguments")
                    try:
                        tool_input = json.loads(raw) if isinstance(raw, str) else (raw or {})
                    except json.JSONDecodeError:
                        tool_input = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": tool_input if isinstance(tool_input, dict) else {},
                    })
                if not blocks:
                    blocks = [{"type": "text", "text": "(empty)"}]
                messages.append({"role": "assistant", "content": blocks})
            else:  # user / anything else
                content = msg.get("content")
                text = content if isinstance(content, str) else json.dumps(content)
                # Multimodal attachments expand into image/document blocks BEFORE
                # the text block (Anthropic's recommended ordering). They live on
                # `_attachments`, hydrated with base64 by the API layer.
                for a in (msg.get("_attachments") or []):
                    block = _attachment_block(a)
                    if block:
                        pending.append(block)
                pending.append({"type": "text", "text": text or "(empty)"})

        flush_user()

        if messages and messages[0]["role"] != "user":
            messages.insert(0, {"role": "user",
                                "content": [{"type": "text",
                                             "text": "(conversation continues)"}]})

        # Third cache breakpoint: the compaction summary — the stable head
        # after a compact (system + tools take the other two).
        for m in reversed(messages):
            if m["role"] != "user":
                continue
            hit = False
            for b in m["content"]:
                if (b.get("type") == "text"
                        and str(b.get("text", "")).startswith("[COMPACTED CONTEXT")):
                    b["cache_control"] = {"type": "ephemeral"}
                    hit = True
                    break
            if hit:
                break
        return messages

    def _build_payload(self, prompt: str, history: list, tools, model,
                       temperature, max_tokens) -> dict:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": int(max_tokens if max_tokens is not None else self.max_tokens),
            "system": [{
                "type": "text",
                "text": prompt or "",
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": self._convert_history(history),
        }
        a_tools = self._convert_tools(tools)
        if a_tools:
            payload["tools"] = a_tools

        budget = self.thinking_budget
        if budget > 0:
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            payload["temperature"] = 1  # required with extended thinking
        else:
            payload["temperature"] = float(
                temperature if temperature is not None else self.temperature)
        return payload

    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    async def generate(self, prompt: str, history: list, tools: list = None,
                       response_format: dict = None, model: str = None,
                       token_callback=None, stream: bool = None,
                       temperature: float = None, max_tokens: int = None) -> dict:
        """Same contract as GenericLLMClient.generate, plus:
        result["thinking_blocks"] — raw thinking blocks for verbatim replay.
        """
        if not self.api_key:
            return {"content": "Missing `llm.api_key` configuration.",
                    "tool_calls": [], "finish_reason": "stop", "usage": {},
                    "error": True}

        use_stream = self.stream if stream is None else bool(stream)
        use_model = model or self.model
        payload = self._build_payload(prompt, history, tools, use_model,
                                      temperature, max_tokens)

        try:
            from core.telemetry import telemetry
        except Exception:
            telemetry = None

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                _start = time.perf_counter()
                if use_stream:
                    result = await asyncio.to_thread(
                        self._request_streaming, payload, token_callback)
                else:
                    result = await asyncio.to_thread(self._request_sync, payload)
                if telemetry:
                    telemetry.observe("llm_call_duration_seconds",
                                      time.perf_counter() - _start, model=use_model)
                    telemetry.incr("llm_calls_total", model=use_model, status="ok")
                    telemetry.record_llm_usage(use_model, result.get("usage") or {})
                self._total_input_tokens += (result.get("usage") or {}).get("prompt_tokens", 0)
                self._total_output_tokens += (result.get("usage") or {}).get("completion_tokens", 0)
                return result
            except _RetryableError as e:
                last_error = str(e)
                if telemetry:
                    telemetry.incr("llm_calls_total", model=use_model, status="retry")
                delay = e.retry_after or (self.BASE_DELAY * (2 ** attempt)
                                          + random.uniform(0, 0.5))
                delay = min(delay, self.MAX_RETRY_AFTER_S)
                logger.warning("anthropic retry in %.1fs (attempt %d/%d): %s",
                               delay, attempt + 1, self.MAX_RETRIES, e)
                await asyncio.sleep(delay)
            except Exception as e:
                last_error = str(e)
                break

        if telemetry:
            telemetry.incr("llm_calls_total", model=use_model, status="error")
        return {"content": f"API error after {self.MAX_RETRIES} attempts: {last_error}",
                "tool_calls": [], "finish_reason": "stop", "usage": {},
                "error": True}

    # ------------------------------------------------------------------
    # HTTP paths (run in a worker thread)
    # ------------------------------------------------------------------

    def _check_response(self, response):
        if response.status_code in self.RETRYABLE_STATUS_CODES:
            retry_after = 0.0
            try:
                retry_after = float(response.headers.get("retry-after") or 0)
            except (TypeError, ValueError):
                retry_after = 0.0
            raise _RetryableError(
                f"HTTP {response.status_code}: {response.text[:200]}",
                retry_after=retry_after)
        if response.status_code >= 400:
            raise Exception(f"HTTP {response.status_code} | {response.text[:500]}")

    @staticmethod
    def _map_usage(usage: dict) -> dict:
        """Anthropic usage → the loop's OpenAI-shaped accounting.

        prompt_tokens is the FULL context size (uncached + cache reads +
        cache writes) so budget/pruning math sees reality; the cache fields
        pass through for the cached-ratio telemetry.
        """
        u = usage or {}
        cache_r = u.get("cache_read_input_tokens", 0) or 0
        cache_w = u.get("cache_creation_input_tokens", 0) or 0
        return {
            "prompt_tokens": (u.get("input_tokens", 0) or 0) + cache_r + cache_w,
            "completion_tokens": u.get("output_tokens", 0) or 0,
            "cache_read_input_tokens": cache_r,
            "cache_creation_input_tokens": cache_w,
        }

    _STOP_MAP = {"end_turn": "stop", "stop_sequence": "stop",
                 "tool_use": "tool_calls", "max_tokens": "length"}

    def _finalize(self, content_parts, thinking_blocks, tool_calls, stop_reason,
                  usage) -> dict:
        parsed_calls = []
        for tc in tool_calls:
            raw = tc.get("_json", "")
            try:
                args = json.loads(raw) if raw else (tc.get("input") or {})
            except json.JSONDecodeError:
                args = {"_raw": raw}
            parsed_calls.append({"id": tc.get("id", ""),
                                 "name": tc.get("name", ""),
                                 "args": args})
        mapped_usage = self._map_usage(usage)
        try:
            from client.cli import _log_token_usage
            _log_token_usage(mapped_usage, self.model)
        except Exception:
            pass
        result = {
            "content": "".join(content_parts) if content_parts else None,
            "tool_calls": parsed_calls,
            "finish_reason": self._STOP_MAP.get(stop_reason, "stop"),
            "usage": mapped_usage,
        }
        if thinking_blocks:
            result["thinking_blocks"] = thinking_blocks
        return result

    def _request_sync(self, payload: dict) -> dict:
        response = requests.post(self.url, headers=self._headers(),
                                 json={**payload, "stream": False}, timeout=300)
        self._check_response(response)
        data = response.json()

        content_parts, thinking_blocks, tool_calls = [], [], []
        for block in data.get("content", []):
            btype = block.get("type")
            if btype == "text":
                content_parts.append(block.get("text", ""))
            elif btype in ("thinking", "redacted_thinking"):
                thinking_blocks.append(block)
            elif btype == "tool_use":
                tool_calls.append({"id": block.get("id"),
                                   "name": block.get("name"),
                                   "input": block.get("input") or {}})
        return self._finalize(content_parts, thinking_blocks, tool_calls,
                              data.get("stop_reason"), data.get("usage"))

    def _request_streaming(self, payload: dict, token_callback=None) -> dict:
        def _emit(kind, text):
            if token_callback and text:
                try:
                    token_callback(kind, text)
                except Exception:
                    pass

        response = requests.post(self.url, headers=self._headers(),
                                 json={**payload, "stream": True},
                                 timeout=300, stream=True)
        self._check_response(response)
        response.encoding = 'utf-8'

        content_parts: List[str] = []
        thinking_blocks: List[dict] = []
        tool_calls: List[dict] = []
        blocks: Dict[int, dict] = {}
        stop_reason = None
        usage: Dict[str, Any] = {}

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            etype = event.get("type")

            if etype == "error":
                err = event.get("error", {})
                if err.get("type") in ("overloaded_error", "rate_limit_error",
                                       "api_error"):
                    raise _RetryableError(f"{err.get('type')}: {err.get('message')}")
                raise Exception(f"{err.get('type')}: {err.get('message')}")

            if etype == "message_start":
                usage.update((event.get("message") or {}).get("usage") or {})
            elif etype == "content_block_start":
                idx = event.get("index", 0)
                cb = event.get("content_block") or {}
                blocks[idx] = {"type": cb.get("type"),
                               "id": cb.get("id"), "name": cb.get("name"),
                               "text": [], "_json": [], "thinking": [],
                               "signature": None}
            elif etype == "content_block_delta":
                idx = event.get("index", 0)
                blk = blocks.setdefault(idx, {"type": "text", "text": [],
                                              "_json": [], "thinking": [],
                                              "signature": None})
                delta = event.get("delta") or {}
                dtype = delta.get("type")
                if dtype == "text_delta":
                    txt = delta.get("text", "")
                    blk["text"].append(txt)
                    _emit("content", txt)
                elif dtype == "input_json_delta":
                    blk["_json"].append(delta.get("partial_json", ""))
                elif dtype == "thinking_delta":
                    th = delta.get("thinking", "")
                    blk["thinking"].append(th)
                    _emit("reasoning", th)
                elif dtype == "signature_delta":
                    blk["signature"] = delta.get("signature")
            elif etype == "message_delta":
                d = event.get("delta") or {}
                if d.get("stop_reason"):
                    stop_reason = d["stop_reason"]
                usage.update(event.get("usage") or {})
            elif etype == "message_stop":
                break

        for idx in sorted(blocks.keys()):
            blk = blocks[idx]
            btype = blk.get("type")
            if btype == "text":
                content_parts.append("".join(blk["text"]))
            elif btype == "thinking":
                tb = {"type": "thinking", "thinking": "".join(blk["thinking"])}
                if blk.get("signature"):
                    tb["signature"] = blk["signature"]
                thinking_blocks.append(tb)
            elif btype == "redacted_thinking":
                thinking_blocks.append({"type": "redacted_thinking",
                                        "data": blk.get("data", "")})
            elif btype == "tool_use":
                tool_calls.append({"id": blk.get("id"), "name": blk.get("name"),
                                   "_json": "".join(blk["_json"])})

        if not content_parts and not tool_calls and not thinking_blocks:
            raise _RetryableError("Empty response")

        return self._finalize(content_parts, thinking_blocks, tool_calls,
                              stop_reason, usage)
