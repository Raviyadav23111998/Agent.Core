import sys
import os
import asyncio
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import app_config
from core.tenant_db import TenantDB
from core.connections import Connections

from core.agent_loop import AgentLoop
from tools.registry import ToolRegistry
from tools.filesystem import FilesystemTools
from tools.shell_execution import ShellExecutionTool
from tools.background_task import BackgroundTaskManager
from state.transcript_logger import TranscriptLogger
from state.artifact_manager import ArtifactManager
from state.knowledge_items import KnowledgeItemManager
from state.memory_store import MemoryStore
from tools.memory_mcp import MemoryMCPTools
from state.conversation_summary import ConversationSummaryManager
from core.user_context import CLI_USER
from async_bus.event_bus import EventBus
from async_bus.scheduler import AgentScheduler

import requests
import logging

_llm_logger = logging.getLogger("agent.llm.usage")


def _log_token_usage(usage: dict, model: str = "") -> None:
    """Emit per-call token usage incl. CACHE telemetry, so we can see whether
    the provider (DeepSeek/OpenRouter/Anthropic) is actually serving cached
    input. Caching savings only show up here — if cached stays 0 across a
    multi-iteration turn, the static prefix is being re-billed in full."""
    if not usage:
        return
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = (details.get("cached_tokens")
              or usage.get("cache_read_input_tokens")
              or usage.get("prompt_cache_hit_tokens")
              or usage.get("cached_tokens") or 0)
    cache_write = usage.get("cache_creation_input_tokens", 0)
    pct = (100 * cached // pt) if pt else 0
    _llm_logger.info(
        "llm.usage model=%s prompt=%d completion=%d cached=%d (%d%%) cache_write=%d",
        (model or "?"), pt, ct, cached, pct, cache_write)
    try:
        from core.telemetry import telemetry
        if pt:
            # Parity acceptance metric: cached share of prompt tokens.
            telemetry.observe("llm_prompt_cached_ratio", cached / pt,
                              model=(model or "?"))
    except Exception:
        pass


def _openai_content_parts(text: str, attachments: list) -> list:
    """Build OpenAI / OpenRouter-style multimodal content for a user turn so
    ANY vision/audio-capable model reachable through the OpenAI-compatible API
    (OpenAI, Gemini, and everything on OpenRouter) receives attachments — not
    just Anthropic-native.

      image     → {"type":"image_url","image_url":{"url":"data:<mime>;base64,…"}}
      pdf/doc   → {"type":"file","file":{"filename":…,"file_data":"data:…;base64,…"}}
      audio     → {"type":"input_audio","input_audio":{"data":…,"format":"mp3"}}

    Video is intentionally not emitted (no standard OpenAI-compat shape); it is
    filtered upstream by capability so it never reaches here — no silent drop.
    """
    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    for a in attachments or []:
        b64 = a.get("data_b64")
        if not b64:
            continue
        mt = a.get("media_type") or ""
        kind = a.get("kind")
        if kind == "image" or mt.startswith("image/"):
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{mt or 'image/png'};base64,{b64}"}})
        elif mt == "application/pdf" or (kind in ("document", "pdf") and not mt):
            parts.append({"type": "file", "file": {
                "filename": a.get("filename") or "document.pdf",
                "file_data": f"data:application/pdf;base64,{b64}"}})
        elif kind == "audio" or mt.startswith("audio/"):
            fmt = mt.split("/")[-1] if "/" in mt else "mp3"
            fmt = {"mpeg": "mp3", "x-wav": "wav", "vnd.wave": "wav"}.get(fmt, fmt)
            parts.append({"type": "input_audio",
                          "input_audio": {"data": b64, "format": fmt}})
    if not parts:
        parts.append({"type": "text", "text": text or "(empty)"})
    return parts


class GenericLLMClient:
    """Production LLM client with native function calling + streaming.
    
    Architecture (identical to Agent):
    
    1. NATIVE FUNCTION CALLING via OpenAI-compatible `tools` parameter
    2. SSE STREAMING: text appears character-by-character in real-time
    3. STRUCTURED RESPONSE: {content, tool_calls, finish_reason, usage}
    4. RELIABILITY: 3x exponential backoff, empty response rejection
    5. THINKING TOKENS: Gemini 2.5 Flash thinking enabled via budget param
    6. TOKEN TRACKING: Actual API-reported usage (not estimates)
    """
    
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    MAX_RETRIES = 3
    BASE_DELAY = 1.0

    def __init__(self, api_key: str = None, model: str = None,
                 url: str = None, stream: bool = True):
        self._api_key = api_key
        self._model = model
        self._url = url
        self._provider = None
        self._provider_order = None
        self._temperature = None
        self._max_tokens = None
        self._fallback_model_override = None
        self._reasoning_effort_override = None
        self.stream = stream
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_thinking_tokens = 0

    @property
    def api_key(self): return self._api_key or app_config.get("llm.api_key")
    @api_key.setter
    def api_key(self, value): self._api_key = value

    @property
    def model(self): return self._model or app_config.get("llm.model")
    @model.setter
    def model(self, value): self._model = value

    @property
    def fallback_model(self):
        """Secondary model to try when the primary fails after all retries.

        Set via `llm.fallback_model` (config) — e.g. a different provider or a
        smaller model — so a primary-model outage degrades instead of failing
        the turn. Empty/unset disables fallback. Never falls back to itself.
        """
        fb = (self._fallback_model_override
              or app_config.get("llm.fallback_model") or "").strip()
        return fb if fb and fb != self.model else None
    @fallback_model.setter
    def fallback_model(self, value): self._fallback_model_override = value

    @property
    def url(self):
        u = self._url or app_config.get("llm.base_url")
        if u and not u.endswith('/chat/completions') and 'messages' not in u:
            u = u.rstrip('/') + '/chat/completions'
        return u
    @url.setter
    def url(self, value): self._url = value

    @property
    def provider(self): return self._provider or app_config.get("llm.provider")
    @provider.setter
    def provider(self, value): self._provider = value

    @property
    def provider_order(self): return self._provider_order or app_config.get("llm.provider_order")
    @provider_order.setter
    def provider_order(self, value): self._provider_order = value

    @property
    def temperature(self):
        if self._temperature is not None: return float(self._temperature)
        t = app_config.get("llm.temperature")
        return float(t) if t else 0.2
    @temperature.setter
    def temperature(self, value): self._temperature = value

    @property
    def max_tokens(self):
        if self._max_tokens is not None: return int(self._max_tokens)
        m = app_config.get("llm.max_tokens")
        return int(m) if m else 16000
    @max_tokens.setter
    def max_tokens(self, value): self._max_tokens = value

    # ── Provider caching matrix ─────────────────────────────────────────
    # Every major LLM vendor caches the large static prompt prefix (system +
    # tool schemas) — but they take it two very different ways, and sending
    # the wrong shape either wastes the cache or makes the request non-
    # standard. We resolve a strategy per model/provider:
    #
    #   'explicit' — Anthropic-style: the caller MUST mark the prefix with
    #                `cache_control: {type: ephemeral}` breakpoints. Only then
    #                does the vendor cache it (billed ~10% on read). Applies to
    #                Claude, whether hit directly or via OpenRouter.
    #   'auto'     — the vendor caches automatically on any byte-stable prefix;
    #                markers are unnecessary (and non-standard). We send CLEAN
    #                requests and let the engine's byte-stable prefix do the
    #                work. Applies to DeepSeek, OpenAI/GPT, Gemini (implicit
    #                caching on 2.5+), GLM/Zhipu, Moonshot/Kimi, Qwen, Grok,
    #                Mistral, Llama, and most modern OpenAI-compatible vendors.
    #   'none'     — no prefix caching; clean requests, no expectation.
    #
    # Because 'auto' vendors all cache on a *stable* prefix, the token win for
    # them lives entirely in the engine (cache-first append-only history +
    # byte-stable system/tools prefix), not in this request shaping.
    _CACHE_EXPLICIT_MODELS = ("claude", "anthropic")
    _CACHE_AUTO_MODELS = (
        "deepseek", "gpt", "o1", "o3", "o4", "gemini", "bison", "gecko",
        "glm", "zhipu", "qwen", "moonshot", "kimi", "grok", "mistral",
        "mixtral", "llama", "command",  # cohere command-r
    )
    _CACHE_EXPLICIT_PROVIDERS = ("anthropic",)
    _CACHE_AUTO_PROVIDERS = (
        "deepseek", "openai", "azure", "google", "gemini", "vertex",
        "zhipu", "glm", "moonshot", "together", "fireworks", "groq",
        "mistral", "cohere", "xai",
    )

    def cache_strategy(self, model: str = None) -> str:
        """Resolve the caching strategy ('explicit' | 'auto' | 'none') for a
        model/provider — the single source of truth for how we shape a request
        for prefix caching. Model name wins (it's the most specific signal,
        and routers like OpenRouter carry the real model in the id); provider
        is the fallback. Defaults to 'auto' because nearly every modern vendor
        auto-caches a stable prefix and clean requests are always safe."""
        m = (model or self.model or "").lower()
        if any(k in m for k in self._CACHE_EXPLICIT_MODELS):
            return "explicit"
        if any(k in m for k in self._CACHE_AUTO_MODELS):
            return "auto"
        prov = (self.provider or "").lower()
        if any(k == prov for k in self._CACHE_EXPLICIT_PROVIDERS):
            return "explicit"
        if any(k == prov for k in self._CACHE_AUTO_PROVIDERS):
            return "auto"
        return "auto"

    @property
    def prompt_caching(self) -> bool:
        """Master on/off for prefix caching. When true (default), the request
        is shaped per `cache_strategy()`: explicit markers for Anthropic, clean
        requests for auto-caching vendors. `llm.prompt_caching=false` disables
        all caching behaviour (e.g. to A/B its effect). Note: this only gates
        whether we *emit Anthropic markers*; auto-caching vendors cache the
        stable prefix regardless — you cannot turn their cache off from here."""
        cfg = app_config.get("llm.prompt_caching")
        if cfg is not None:
            return str(cfg).strip().lower() in ("1", "true", "yes", "on")
        return True

    def _apply_prompt_caching(self, messages: list, payload: dict):
        """Attach ephemeral cache_control to the static prompt prefix.

        Caches two breakpoints (Anthropic allows up to 4):
          1. The system prompt (largest single static block).
          2. The tool-schema array (via cache_control on the last tool).
        Providers that don't support caching ignore these markers, so this
        is a no-op cost win where supported and harmless elsewhere — it is
        only enabled for Claude models in the first place.
        """
        if not messages:
            return
        sys_msg = messages[0]
        if sys_msg.get("role") == "system" and isinstance(sys_msg.get("content"), str):
            text = sys_msg["content"]
            if text.strip():
                sys_msg["content"] = [{
                    "type": "text",
                    "text": text,
                    "cache_control": {"type": "ephemeral"},
                }]
        tools = payload.get("tools")
        if tools:
            last = tools[-1]
            if isinstance(last, dict):
                # Copy instead of mutating: the tools list aliases the
                # AgentLoop's cached schema dicts, and each turn selects a
                # different subset — in-place markers would accumulate across
                # turns until the request exceeds the provider's cache-
                # breakpoint limit (Anthropic allows 4) and starts failing.
                payload["tools"] = tools[:-1] + [
                    {**last, "cache_control": {"type": "ephemeral"}}
                ]

        # Third breakpoint: the compaction summary is the new stable head of
        # the message list after a compact — caching it means post-compaction
        # calls only pay for the tail. The message dicts here are per-request
        # copies, so wrapping is mutation-safe.
        for msg in reversed(messages):
            content = msg.get("content")
            if (msg.get("role") == "user" and isinstance(content, str)
                    and content.startswith("[COMPACTED CONTEXT")):
                msg["content"] = [{
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }]
                break

    @staticmethod
    def _strip_cache_markers(payload: dict) -> dict:
        """Return a copy of the payload without prompt-caching markers.

        Used before falling back to a model whose provider may reject
        cache_control fields (strict OpenAI-compatible endpoints).
        """
        clean = dict(payload)
        msgs = clean.get("messages") or []
        new_msgs = []
        for m in msgs:
            content = m.get("content")
            if (isinstance(content, list) and content
                    and isinstance(content[0], dict)
                    and "cache_control" in content[0]):
                m = {**m, "content": "".join(
                    b.get("text", "") for b in content if isinstance(b, dict))}
            new_msgs.append(m)
        clean["messages"] = new_msgs
        tools = clean.get("tools")
        if tools:
            clean["tools"] = [
                {k: v for k, v in t.items() if k != "cache_control"}
                if isinstance(t, dict) else t
                for t in tools
            ]
        return clean

    async def generate(self, prompt: str, history: list,
                       tools: list = None, response_format: dict = None,
                       model: str = None, token_callback=None,
                       stream: bool = None, temperature: float = None,
                       max_tokens: int = None,
                       reasoning_effort: str = None) -> dict:
        """Generate LLM response with native FC + optional streaming.

        Returns structured dict:
        {
            "content": str or None,
            "tool_calls": [{"id": "...", "name": "...", "args": {...}}],
            "finish_reason": "stop" | "tool_calls" | "length",
            "usage": {"prompt_tokens": N, "completion_tokens": N}
        }

        token_callback: optional ``fn(kind, text)`` invoked per streamed delta
        while the response is generated — kind is "content" (answer) or
        "reasoning" (thinking). Called from the streaming worker thread, so it
        must be thread-safe and non-blocking (the agent loop bridges it onto
        the event loop). Ignored for the non-streaming path.

        Note: History pruning is handled by AgentLoop._prune_history().
        This method sends the full history as-is.
        """
        if not self.api_key:
            return {
                "content": "Missing `llm.api_key` configuration.",
                "tool_calls": [], "finish_reason": "stop", "usage": {},
                "error": True,
            }

        messages = [{"role": "system", "content": prompt}]
        
        for msg in history:
            role = msg.get("role", "user")
            if role not in ("user", "assistant", "system", "tool"):
                role = "user"
            
            entry = {"role": role}
            
            if role == "tool":
                entry["tool_call_id"] = msg.get("tool_call_id", "unknown")
                if "name" in msg:
                    entry["name"] = msg["name"]
                entry["content"] = msg.get("content", "")
            elif role == "assistant" and "tool_calls" in msg:
                entry["content"] = msg.get("content") or ""
                entry["tool_calls"] = msg["tool_calls"]
            elif role == "user" and msg.get("_attachments"):
                # Multimodal user turn — OpenAI/OpenRouter/Gemini image_url+file+audio.
                entry["content"] = _openai_content_parts(
                    msg.get("content", ""), msg.get("_attachments"))
            else:
                entry["content"] = msg.get("content", "")

            messages.append(entry)

        db_name = app_config.get("db.name", "agent_dev")
        env_name = os.environ.get("AG_ENV", db_name.split('_')[-1] if '_' in db_name else db_name)
        
        referer = f"https://{env_name}-ai.storeglance.com" if env_name in ("uat", "sit", "prod") else "http://localhost"
        # Attribution title (shows as "App" on provider dashboards, e.g.
        # OpenRouter; harmless elsewhere). Include the bound tenant so
        # multi-tenant traffic is distinguishable per request — read lazily
        # because the tenant ContextVar changes per request.
        tenant_label = ""
        try:
            from core.tenant_context import current_tenant
            tenant_label = getattr(current_tenant(), "tenant_key", "") or ""
        except Exception:
            pass
        title = (f"Agent {tenant_label} ({env_name.upper()})" if tenant_label
                 else f"Agent ({env_name.upper()})")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": referer,
            "X-Title": title,
            "Content-Type": "application/json"
        }

        use_model = model or self.model
        use_stream = self.stream if stream is None else bool(stream)

        payload = {
            "model": use_model,
            "messages": messages,
            "temperature": float(temperature) if temperature is not None
                           else self.temperature,
            "stream": use_stream,
            "max_tokens": int(max_tokens) if max_tokens is not None
                          else self.max_tokens,
        }

        if self.provider == "openrouter":
            payload["usage"] = {"include": True}
        elif use_stream:
            payload["stream_options"] = {"include_usage": True}

        if self.provider == "openrouter" and self.provider_order:
            payload["provider"] = {
                "order": [p.strip() for p in self.provider_order.split(",")],
                "allow_fallbacks": True,
            }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if response_format and not tools:
            payload["response_format"] = response_format

        # Reasoning effort — first-class request control (the effort knob
        # Claude Code exposes). Reasoning models were measured spending
        # 4-10k completion tokens PER CALL on thinking for triage-grade
        # work. Per-call param wins; else `llm.reasoning_effort` config.
        # PROVIDER-NATIVE attach: gating this on OpenRouter alone silently
        # dropped the knob after the DeepSeek-direct switch — v4-flash
        # defaults to thinking ON, and a live probe (2026-07-04) measured a
        # trivial answer at 28 completion tokens, 26 of them reasoning;
        # `thinking:{type:disabled}` cut it to 1.
        effort = reasoning_effort
        if effort is None:
            effort = (self._reasoning_effort_override
                      or app_config.get("llm.reasoning_effort")
                      or "").strip().lower() or None
        if effort:
            prov = (self.provider or "").lower()
            url_l = (self.url or "").lower()
            is_off = effort in ("off", "none", "disabled")
            if prov == "openrouter":
                payload["reasoning"] = ({"enabled": False} if is_off
                                        else {"effort": effort})
            elif prov == "deepseek" or "deepseek.com" in url_l:
                if is_off:
                    payload["thinking"] = {"type": "disabled"}
                else:
                    payload["reasoning_effort"] = effort  # accepted (live-verified)
            elif prov in ("zhipu", "glm") or "bigmodel.cn" in url_l:
                # GLM hybrid reasoners expose a binary switch only.
                payload["thinking"] = {"type": ("disabled" if is_off
                                                else "enabled")}
            elif prov == "openai" and not is_off:
                payload["reasoning_effort"] = effort  # o-series native knob

        # Shape the request for the resolved provider's caching model. Only
        # Anthropic needs explicit cache_control markers; auto-caching vendors
        # (DeepSeek/OpenAI/Gemini/GLM/…) cache the byte-stable prefix on their
        # own, so we send them clean requests (no non-standard markers).
        if self.prompt_caching and self.cache_strategy(use_model) == "explicit":
            self._apply_prompt_caching(messages, payload)

        try:
            from core.telemetry import telemetry
        except Exception:
            telemetry = None

        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                import time as _t
                _start = _t.perf_counter()
                if use_stream:
                    result = await asyncio.to_thread(
                        self._generate_streaming, headers, payload, attempt,
                        token_callback
                    )
                else:
                    result = await asyncio.to_thread(
                        self._generate_sync, headers, payload, attempt
                    )
                if telemetry:
                    telemetry.observe("llm_call_duration_seconds",
                                      _t.perf_counter() - _start, model=use_model)
                    telemetry.incr("llm_calls_total", model=use_model, status="ok")
                    telemetry.record_llm_usage(use_model, result.get("usage") or {},
                                               provider=self.provider, url=self.url)
                return result

            except _RetryableError as e:
                if telemetry:
                    telemetry.incr("llm_calls_total", model=use_model, status="retry")
                # Honor the server's Retry-After when given (a 429 storm with
                # fixed 1/2/4s backoff burns all attempts and kills the turn);
                # cap it so a hostile header can't stall the loop.
                delay = min(getattr(e, "retry_after", 0.0)
                            or self.BASE_DELAY * (2 ** attempt), 30.0)
                print(f"  [LLM] {e}, retrying in {delay}s "
                      f"(attempt {attempt+1}/{self.MAX_RETRIES})...")
                await asyncio.sleep(delay)
                last_error = str(e)
            except _FatalError as e:
                last_error = str(e)
                break
            except Exception as e:
                last_error = str(e)
                break

        if telemetry:
            telemetry.incr("llm_calls_total", model=use_model, status="error")

        fb = self.fallback_model
        if fb == use_model:
            fb = None
        if fb:
            fb_payload = {**payload, "model": fb}
            # Reshape for the FALLBACK model's caching strategy — it may differ
            # from the primary's (e.g. primary DeepSeek 'auto' → fallback Claude
            # 'explicit', or vice-versa). Operate on COPIES: fb_payload aliases
            # the primary's messages list, and marking through the alias would
            # corrupt the primary's request shape for any later inspection.
            if self.prompt_caching and self.cache_strategy(fb) == "explicit":
                fb_messages = [dict(m) for m in messages]
                fb_payload["messages"] = fb_messages
                self._apply_prompt_caching(fb_messages, fb_payload)
            else:
                fb_payload = self._strip_cache_markers(fb_payload)
            import logging
            logging.getLogger(__name__).warning(
                "llm.fallback: primary=%s failed (%s) → trying %s",
                use_model, last_error, fb)
            try:
                if use_stream:
                    result = await asyncio.to_thread(
                        self._generate_streaming, headers, fb_payload, 0,
                        token_callback)
                else:
                    result = await asyncio.to_thread(
                        self._generate_sync, headers, fb_payload, 0)
                if telemetry:
                    telemetry.incr("llm_calls_total", model=fb, status="fallback_ok")
                    telemetry.record_llm_usage(fb, result.get("usage") or {},
                                               provider=self.provider, url=self.url)
                return result
            except Exception as e:
                if telemetry:
                    telemetry.incr("llm_calls_total", model=fb, status="fallback_error")
                last_error = f"{last_error}; fallback {fb}: {e}"

        return {
            "content": f"API error after {self.MAX_RETRIES} attempts: {last_error}",
            "tool_calls": [], "finish_reason": "stop", "usage": {},
            "error": True,
        }

    def _generate_sync(self, headers, payload, attempt) -> dict:
        """Non-streaming response. Runs in a thread via asyncio.to_thread."""
        payload_copy = {**payload, "stream": False}
        
        response = requests.post(
            self.url, headers=headers, json=payload_copy, timeout=120
        )

        if response.status_code in self.RETRYABLE_STATUS_CODES:
            raise _RetryableError(f"HTTP {response.status_code}",
                                  retry_after=self._retry_after_s(response))

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"{e} | Response Body: {response.text}")

        response.encoding = 'utf-8'
        data = response.json()
        return self._parse_sync_response(data, attempt)

    @staticmethod
    def _retry_after_s(response) -> float:
        """Server-requested backoff (Retry-After header), 0 when absent."""
        try:
            return float(response.headers.get("retry-after") or 0)
        except (TypeError, ValueError):
            return 0.0

    def _generate_streaming(self, headers, payload, attempt, token_callback=None) -> dict:
        """SSE streaming response — text appears in real-time.
        Runs in a thread via asyncio.to_thread.

        Handles the complexity of assembling tool_calls from stream deltas:
        - tool_calls arrive as incremental chunks
        - arguments field arrives as partial JSON strings that must be concatenated
        - Content deltas are printed immediately for real-time UX
        - reasoning deltas (reasoning models) are surfaced separately as
          "thinking" so the UI can render them apart from the answer
        - token_callback(kind, text), if given, receives each delta live
          (kind = "content" | "reasoning")
        """
        def _emit(kind, text):
            if token_callback and text:
                try:
                    token_callback(kind, text)
                except Exception:
                    pass
        payload_copy = {**payload, "stream": True}

        from core.thinking_filter import ThinkingSplitter
        split_thinking = app_config.get_bool("llm.split_inline_thinking", True)
        thinking_splitter = ThinkingSplitter() if split_thinking else None

        if app_config.get_bool("llm.debug_dump_payload", False):
            try:
                os.makedirs("temp", exist_ok=True)
                with open("temp/payload_dump.json", "w") as f:
                    json.dump(payload_copy, f, indent=2)
            except Exception:
                pass

        response = requests.post(
            self.url, headers=headers, json=payload_copy, 
            timeout=120, stream=True
        )
        
        if response.status_code in self.RETRYABLE_STATUS_CODES:
            raise _RetryableError(f"HTTP {response.status_code}",
                                  retry_after=self._retry_after_s(response))

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"{e} | Response Body: {response.text}")

        response.encoding = 'utf-8'

        content_parts = []
        tool_calls_accum = {}
        finish_reason = "stop"
        usage = {}
        printed_header = False
        
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            
            if not line.startswith("data: "):
                try:
                    err_data = json.loads(line)
                    if "error" in err_data:
                        print(f"  [LLM Error payload] {err_data['error']}")
                        raise _FatalError(f"OpenRouter Error: {err_data['error']}")
                except Exception:
                    pass
                continue

            data_str = line[6:]
            
            if data_str.strip() == "[DONE]":
                break
            
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            
            if "error" in data:
                print(f"  [LLM Error] {data['error']}")
                raise _FatalError(f"OpenRouter Error: {data['error']}")

            choice = data.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            fr = choice.get("finish_reason")
            
            if fr:
                finish_reason = fr
            
            if "usage" in data:
                usage = data["usage"]
            
            reasoning_chunk = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning_chunk:
                _emit("reasoning", reasoning_chunk)

            if "content" in delta and delta["content"]:
                chunk = delta["content"]
                if thinking_splitter is not None:
                    answer_chunk, reasoning_chunk2 = thinking_splitter.feed(chunk)
                    if reasoning_chunk2:
                        _emit("reasoning", reasoning_chunk2)
                    chunk = answer_chunk
                    if not chunk:
                        continue
                content_parts.append(chunk)
                _emit("content", chunk)

                if not printed_header:
                    print("\nAgent> ", end="", flush=True)
                    printed_header = True
                print(chunk, end="", flush=True)
            
            if "tool_calls" in delta:
                for tc_delta in delta["tool_calls"]:
                    idx = tc_delta.get("index", 0)
                    
                    if idx not in tool_calls_accum:
                        tool_calls_accum[idx] = {
                            "id": tc_delta.get("id", f"call_{int(time.time())}_{idx}"),
                            "name": "",
                            "arguments_parts": []
                        }
                    
                    acc = tool_calls_accum[idx]
                    
                    if "id" in tc_delta and tc_delta["id"]:
                        acc["id"] = tc_delta["id"]
                    
                    func = tc_delta.get("function", {})
                    if "name" in func and func["name"]:
                        acc["name"] = func["name"]
                    if "arguments" in func and func["arguments"]:
                        acc["arguments_parts"].append(func["arguments"])
        
        if thinking_splitter is not None:
            tail_answer, tail_reasoning = thinking_splitter.flush()
            if tail_reasoning:
                _emit("reasoning", tail_reasoning)
            if tail_answer:
                content_parts.append(tail_answer)
                _emit("content", tail_answer)
                if not printed_header:
                    print("\nAgent> ", end="", flush=True)
                    printed_header = True
                print(tail_answer, end="", flush=True)

        if printed_header:
            print()

        content = "".join(content_parts) if content_parts else None
        
        parsed_tool_calls = []
        for idx in sorted(tool_calls_accum.keys()):
            acc = tool_calls_accum[idx]
            raw_args = "".join(acc["arguments_parts"])
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            
            parsed_tool_calls.append({
                "id": acc["id"],
                "name": acc["name"],
                "args": args
            })
        
        if parsed_tool_calls and finish_reason != "length":
            finish_reason = "tool_calls"
        
        self._total_input_tokens += usage.get("prompt_tokens", 0)
        self._total_output_tokens += usage.get("completion_tokens", 0)
        # Attribute usage to the model actually in the request payload — on a
        # fallback call that is the FALLBACK model, and billing/telemetry must
        # follow the served model, not the configured primary.
        served_model = payload.get("model") or self.model
        _log_token_usage(usage, served_model)

        if not content and not parsed_tool_calls:
            print(f"  [LLM Debug] Status Code: {response.status_code}")
            print(f"  [LLM Debug] Headers: {response.headers}")
            print(f"  [LLM Debug] Raw streamed lines: {content_parts}")
            raise _RetryableError("Empty response")

        return {
            "content": content,
            "tool_calls": parsed_tool_calls,
            "finish_reason": finish_reason,
            "usage": usage,
            "served_model": served_model
        }

    def _parse_sync_response(self, data: dict, attempt: int) -> dict:
        """Parse a non-streaming API response."""
        choice = data["choices"][0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")
        usage = data.get("usage", {})

        self._total_input_tokens += usage.get("prompt_tokens", 0)
        self._total_output_tokens += usage.get("completion_tokens", 0)
        # The provider echoes the model that actually served the request —
        # trust it over self.model so fallback calls are billed correctly.
        served_model = data.get("model") or self.model
        _log_token_usage(usage, served_model)

        content = message.get("content")
        raw_tool_calls = message.get("tool_calls", [])
        
        parsed_tool_calls = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            call_id = tc.get("id", f"call_{int(time.time())}")
            raw_args = func.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            
            parsed_tool_calls.append({
                "id": call_id,
                "name": func.get("name", ""),
                "args": args
            })
        
        if parsed_tool_calls and finish_reason != "length":
            finish_reason = "tool_calls"
        
        if not content and not parsed_tool_calls:
            raise _RetryableError("Empty response")
        
        return {
            "content": content,
            "tool_calls": parsed_tool_calls,
            "finish_reason": finish_reason,
            "usage": usage,
            "served_model": served_model
        }

    def get_token_stats(self) -> str:
        """Return cumulative token usage (actual from API)."""
        total = self._total_input_tokens + self._total_output_tokens
        return (f"Input: {self._total_input_tokens} | "
                f"Output: {self._total_output_tokens} | "
                f"Total: {total}")


def create_llm_client(stream: bool = True, api_key: str = None,
                      model: str = None, url: str = None,
                      profile: dict = None):
    """Provider-selected LLM client.

    provider `anthropic` → native Messages adapter (thinking preservation,
    is_error tool results, Retry-After, native caching). Anything else →
    the OpenAI-compatible client (OpenRouter, DeepSeek, OpenAI, GLM, …).
    Both expose the same generate() contract.

    profile: a complete model identity (row from ag_system.llm_profiles —
    provider, model, base_url, api_key, temperature, max_tokens,
    fallback_model, reasoning_effort). When given, EVERY connection value is
    pinned on the instance so the client never drifts with global llm.*
    config — this is what makes App A run DeepSeek while App B runs GLM.
    An api_key of '__config:<key>__' resolves through app_config (the V035
    default-profile sentinel: secret stays in configs this release).
    """
    prof = profile or {}
    provider = ((prof.get("provider") or app_config.get("llm.provider"))
                or "").strip().lower()
    p_key = api_key or prof.get("api_key") or None
    if isinstance(p_key, str) and p_key.startswith("__config:") and p_key.endswith("__"):
        p_key = app_config.get(p_key[len("__config:"):-2]) or None
    p_model = model or prof.get("model") or None
    p_url = url or prof.get("base_url") or None

    if provider == "anthropic":
        from client.anthropic_client import AnthropicMessagesClient
        client = AnthropicMessagesClient(api_key=p_key, model=p_model,
                                         url=p_url, stream=stream)
    else:
        client = GenericLLMClient(api_key=p_key, model=p_model, url=p_url,
                                  stream=stream)

    if prof:
        client.provider = provider or None
        # Pin remaining knobs; setattr-guarded so both client classes work.
        for attr, key in (("temperature", "temperature"),
                          ("max_tokens", "max_tokens"),
                          ("fallback_model", "fallback_model")):
            val = prof.get(key)
            if val is not None and val != "":
                try:
                    setattr(client, attr, val)
                except Exception:
                    pass
        effort = (prof.get("reasoning_effort") or "").strip().lower()
        if effort:
            try:
                client._reasoning_effort_override = effort
            except Exception:
                pass
        client.profile_id = prof.get("profile_id")
        client.profile_small_model = prof.get("small_model") or None
    return client


# ── Per-profile client cache ────────────────────────────────────────────
# One client per profile per process: sessions of different apps hold
# different instances (different key/url/model), while sessions of the SAME
# profile share one — preserving the byte-stable prefix per provider. The
# API layer busts this on profile edits via invalidate_profile_clients()
# (same cross-worker pattern as the credential provider cache).
_profile_clients: dict = {}
_profile_clients_lock = __import__("threading").Lock()


def _cache_tenant_key() -> str:
    """profile_id is a PER-TENANT serial: tenant A's profile 1 and tenant
    B's profile 1 are different identities (different key/url/model). The
    cache key must carry the tenant or a multi-tenant worker could hand one
    tenant's client (and API key) to another."""
    try:
        from core.tenant_context import current_tenant_id
        return current_tenant_id() or ""
    except Exception:
        return ""


def get_profile_client(profile: dict, stream: bool = True):
    """Return the cached client for this profile row (keyed by tenant +
    id + credential + updated_at so an edited profile naturally misses).

    credential_ref is set by llm_scope.with_credential when a service key
    brings its own provider key (V058): the same profile under two
    credentials is two clients, and they must not evict each other — a
    production key and a UAT key alternating on one application would
    otherwise rebuild the client, and lose its warm provider prefix, on
    every other turn. Profiles without an override carry no ref ("") and
    behave exactly as before."""
    if not profile or not profile.get("profile_id"):
        return create_llm_client(stream=stream, profile=profile)
    key = (_cache_tenant_key(), profile["profile_id"],
           str(profile.get("credential_ref") or ""),
           str(profile.get("updated_at") or ""), stream)
    with _profile_clients_lock:
        client = _profile_clients.get(key)
        if client is None:
            client = create_llm_client(stream=stream, profile=profile)
            # Drop stale generations of the same tenant's profile UNDER THE
            # SAME CREDENTIAL; other credentials' clients stay.
            for k in [k for k in _profile_clients
                      if k[:3] == key[:3] and k != key]:
                _profile_clients.pop(k, None)
            _profile_clients[key] = client
    return client


def invalidate_profile_clients(profile_id=None):
    """Bust cached clients (all, or one profile id — across tenants; the
    over-invalidation is harmless) after a profile edit."""
    with _profile_clients_lock:
        if profile_id is None:
            _profile_clients.clear()
        else:
            for k in [k for k in _profile_clients if k[1] == profile_id]:
                _profile_clients.pop(k, None)


class _RetryableError(Exception):
    """Error that should trigger a retry.

    retry_after: server-requested delay (Retry-After header) in seconds; 0
    means use exponential backoff. Honoring it stops mid-turn 429 storms
    from exhausting the 3 fixed-backoff attempts and killing the turn.
    """
    def __init__(self, msg: str, retry_after: float = 0.0):
        super().__init__(msg)
        self.retry_after = retry_after

class _FatalError(Exception):
    """Error that should NOT be retried."""
    pass


class AgentCLI:
    """Main client — wires all subsystems and runs the REPL.
    
    Initialization order (dependency-aware):
    1. State managers (transcript, artifacts, KI, memory, summaries)
    2. Event bus + scheduler
    3. Tool registry
    4. LLM client (with streaming)
    5. Agent loop
    6. Knowledge + conversation summary injection
    """

    def _parse_company_env(self) -> dict:
        env_vars = {}
        
        env_paths = [
            os.path.abspath(os.path.join(self.workspace_path, '..', 'CompanyAgent', 'hermes', '.env')),
            os.path.abspath(os.path.join(self.workspace_path, '..', 'CompanyAgent', '.env')),
            os.path.abspath(os.path.join(self.workspace_path, '.env')),
        ]
        
        for env_path in env_paths:
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        if '=' in line and not line.strip().startswith('#'):
                            key, val = line.split("=", 1)
                            env_vars[key.strip()] = val.strip().strip('"').strip("' ")
        
        for key, val in env_vars.items():
            if key not in os.environ:
                os.environ[key] = val
        
        return env_vars

    def __init__(self, workspace_path: str, db=None):
        self.workspace_path = os.path.abspath(workspace_path)
        print(f"Initializing Agent in: {self.workspace_path}")

        self.env_vars = self._parse_company_env()

        app_config.load()
        self.config = app_config

        if db is not None:
            self.tenant_db = db
        else:
            self.tenant_db = TenantDB()
            self.tenant_db.connect()

        self.connections = Connections(self.tenant_db)

        self.logger = TranscriptLogger(
            self.tenant_db,
            user_id=CLI_USER.data_dir_name,
        )
        self.artifacts = ArtifactManager(self.tenant_db)
        self.knowledge = KnowledgeItemManager(self.workspace_path)
        self.memory = MemoryStore(db=self.tenant_db)
        self.summaries = ConversationSummaryManager(
            db=self.tenant_db,
        )

        self.event_bus = EventBus()
        self.scheduler = AgentScheduler(self.event_bus)
        self.bg_manager = BackgroundTaskManager(self.event_bus, self.workspace_path)
        from agents.task_manager import SubAgentManager
        self._subagents = SubAgentManager(self.event_bus)

        self.tool_registry = ToolRegistry()
        self._register_core_tools()

        self.llm = create_llm_client(stream=True)
        if not self.llm.api_key:
            print("  ⚠ WARNING: LLM API key not found in config. LLM calls will fail.")

        self._register_external_tools()

        self._integrations = None
        try:
            from core.embeddings import EmbeddingClient
            from core.integrations.runtime import (
                DynamicHttpExecutor, IntegrationToolProvider)
            self._embeddings = EmbeddingClient()
            self._integrations = IntegrationToolProvider(
                self.connections, self._embeddings,
                DynamicHttpExecutor(self.tenant_db, self.connections, self._embeddings))
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("dynamic integrations init skipped: %s", e)

        self.agent = AgentLoop(
            self.workspace_path,
            self.llm,
            self.tool_registry,
            self.event_bus,
            transcript_logger=self.logger,
            memory_manager=self.memory,
            user_context=CLI_USER,
        )

        _kis = list(self.knowledge.load_active_kis())
        recent_summaries = self.summaries.load_recent_summaries(count=3)
        if recent_summaries:
            _kis.append(f"Recent conversations:\n{recent_summaries}")
        from core.prompt_sections import (
            render_knowledge, KNOWLEDGE_KEY, KNOWLEDGE_ORDER)
        self.agent.prompt_builder.set_section(
            KNOWLEDGE_KEY, render_knowledge(_kis), KNOWLEDGE_ORDER)
            
        self.event_bus.subscribe('agent_response', self._handle_response)
        
        self.event_bus.subscribe('timer_wakeup', self._handle_timer_wakeup)
        
        self.event_bus.subscribe('task_completed', self._handle_task_completed)

        self.event_bus.subscribe('subagent_completed', self._handle_subagent_completed)

    async def _handle_response(self, payload: dict):
        """Handle agent response — only print if NOT already streamed.

        Error/budget/interaction responses are event-published, never token-
        streamed, so they must always be printed or the user sees nothing.
        """
        message = payload.get("message", "")
        if (not self.llm.stream
                or message.startswith(("[", "⚠", "⏰", "⏹"))
                or payload.get("error")
                or payload.get("cancelled")
                or payload.get("budget_exceeded")
                or payload.get("interaction_failed")):
            print(f"\nAgent> {message}")

    async def _handle_timer_wakeup(self, payload: dict):
        """Route timer wakeup to agent as a user input.
        
        Without this, scheduled timers fire but nothing happens.
        This matches Agent's reactive wakeup system.
        """
        prompt = payload.get("prompt", "Timer fired")
        timer_id = payload.get("timer_id", "?")
        print(f"\n  [Timer {timer_id}] {prompt}")
        await self.event_bus.publish('user_input', {
            "user_input": f"[TIMER WAKEUP] {prompt}"
        })

    async def _handle_task_completed(self, payload: dict):
        """Route background task completion to agent.
        
        When a background task finishes, the agent is notified so it can 
        react to the output (e.g., a build finished, a test completed).
        """
        task_id = payload.get("task_id", "?")
        command = payload.get("command", "?")[:60]
        return_code = payload.get("return_code", -1)
        stdout = payload.get("stdout", "")[:500]
        
        status = "succeeded" if return_code == 0 else f"failed (exit {return_code})"
        print(f"\n  [Task {task_id}] {command} — {status}")
        
        await self.event_bus.publish('user_input', {
            "user_input": (
                f"[BACKGROUND TASK COMPLETED] Task '{task_id}' ({command}) "
                f"{status}.\nOutput:\n{stdout}"
            )
        })


    _LOCAL_TOOL_NAMES = frozenset({
        "view_file", "write_to_file", "list_dir", "grep_search", "glob_files",
        "replace_file_content", "multi_replace_file_content", "run_command",
    })

    _BRAIN_TOOL_NAMES = frozenset({
        "brain_ingest_file", "brain_ingest_directory", "brain_ingest_text",
        "brain_ask", "brain_search", "brain_stats",
        "brain_sync", "brain_sync_status", "brain_fetch",
    })

    def create_user_tools(self, user_workspace: str, sandboxed: bool = True,
                          max_size_bytes: int = 0, user_context=None,
                          allowed_tools: list = None):
        """Create a ToolRegistry with sandboxed filesystem + shared server tools.
        
        For web/cloud users: filesystem tools are jailed to user_workspace.
        Server tools (jira, memory, etc.) are shared from the main registry.
        Brain tools are RE-CREATED per-user with a scoped resolver when
        user_context is provided (ensures app_id isolation for bots).
        
        Args:
            user_workspace: Absolute path to the user's workspace directory.
            sandboxed: If True, filesystem/shell are jailed (default for web users).
            max_size_bytes: Disk quota in bytes. 0 = no quota.
            user_context: Optional UserContext for scoping brain tools.
                         When provided, brain tools use this context's app_id
                         instead of the shared CLI context.
            allowed_tools: Optional list of allowed tool names. If provided,
                          ONLY these tools remain in the registry (deterministic
                          security — tools not on this list are physically removed).
        
        Returns:
            A ToolRegistry instance with sandboxed local tools + shared server tools.
        """
        from tools.registry import ToolRegistry as TR
        
        user_fs = FilesystemTools(user_workspace, sandboxed=sandboxed, 
                                  max_size_bytes=max_size_bytes)
        user_shell = ShellExecutionTool(user_workspace, 
                                         background_manager=self.bg_manager,
                                         sandboxed=sandboxed)
        
        user_registry = TR()
        
        user_registry.register(
            "view_file",
            f"Read a file's contents with line numbers. Your workspace is: {user_workspace}. You can ONLY access files within this directory.",
            {"absolute_path": "string", "start_line?": "int", "end_line?": "int"},
            user_fs.view_file,
            param_descriptions={
                "absolute_path": f"Path to the file. Must be within {user_workspace}",
                "start_line": "First line to show (1-indexed).",
                "end_line": "Last line to show (1-indexed, inclusive).",
            }
        )
        user_registry.register(
            "write_to_file",
            f"Create or overwrite a file within your workspace: {user_workspace}",
            {"target_file": "string", "code_content": "string", "overwrite?": "bool"},
            user_fs.write_to_file,
            param_descriptions={
                "target_file": f"Path for the file. Must be within {user_workspace}",
                "code_content": "The full content to write.",
                "overwrite": "Set to true to overwrite. Defaults to false.",
            }
        )
        user_registry.register(
            "list_dir",
            f"List directory contents. Restricted to your workspace: {user_workspace}",
            {"directory_path": "string"},
            user_fs.list_dir,
            param_descriptions={
                "directory_path": f"Path to directory. Must be within {user_workspace}",
            }
        )
        user_registry.register(
            "grep_search",
            f"Search files for text/regex. Restricted to your workspace: {user_workspace}",
            {"search_path": "string", "query": "string", "is_regex?": "bool",
             "case_insensitive?": "bool", "includes?": "string",
             "context_lines?": "int", "files_with_matches?": "bool",
             "max_results?": "int"},
            user_fs.grep_search,
            param_descriptions={
                "search_path": f"Path to search. Must be within {user_workspace}",
                "query": "Text or regex pattern to search for.",
                "is_regex": "If true, treat query as regex.",
                "case_insensitive": "If true, ignore case.",
                "includes": "Comma-separated glob patterns, e.g. '*.py,*.js'.",
                "context_lines": "Lines of context around each match (max 10).",
                "files_with_matches": "If true, return only matching file paths.",
                "max_results": "Raise the match cap (up to 200). Default 50.",
            }
        )
        user_registry.register(
            "glob_files",
            f"Find files matching a glob pattern, newest first. Restricted to your workspace: {user_workspace}",
            {"pattern": "string", "path?": "string"},
            user_fs.glob_files,
            param_descriptions={
                "pattern": "Glob pattern matched against file names and relative paths.",
                "path": f"Directory to search under. Defaults to {user_workspace}",
            }
        )
        user_registry.register(
            "replace_file_content",
            f"Replace text in a file. Restricted to your workspace: {user_workspace}",
            {"target_file": "string", "target_content": "string", "replacement_content": "string",
             "start_line?": "int", "end_line?": "int", "replace_all?": "bool"},
            user_fs.replace_file_content,
            param_descriptions={
                "target_file": f"Path to file. Must be within {user_workspace}",
                "target_content": "Exact text to find and replace.",
                "replacement_content": "New text.",
                "start_line": "Narrow search start (1-indexed).",
                "end_line": "Narrow search end (1-indexed).",
                "replace_all": "If true, replace every occurrence.",
            }
        )
        user_registry.register(
            "multi_replace_file_content",
            f"Apply multiple replacements to a file. Restricted to your workspace: {user_workspace}",
            {"target_file": "string", "replacements": "string"},
            user_fs.multi_replace_file_content,
            param_descriptions={
                "target_file": f"Path to file. Must be within {user_workspace}",
                "replacements": 'JSON array: [{"target": "old", "replacement": "new"}, ...]',
            }
        )
        # The shell instance itself declares its policy: kernel-jailed →
        # write (no approval stall); unsandboxed/regex-only → irreversible
        # (approval gate stays). See ShellExecutionTool.recommended_policy.
        _shell_policy = user_shell.recommended_policy()
        user_registry.register(
            "run_command",
            f"Execute a shell command (exit code reported; cwd persists across calls). Working directory must be within your workspace: {user_workspace}",
            {"command_line": "string", "cwd?": "string", "wait_ms_before_async?": "int"},
            user_shell.run_command,
            param_descriptions={
                "command_line": "The shell command to execute.",
                "cwd": f"Working directory (defaults to persistent session dir). Must be within {user_workspace}",
                "wait_ms_before_async": "Ms before promotion to background. Default 120000.",
            },
            policy=_shell_policy,
        )

        # Part 2 — media generation. Only exposed when a provider is configured;
        # otherwise the model has no way to generate and the UI reports output
        # support = false (so the user is told, never silently ignored).
        try:
            from core.media_providers import get_image_provider
            _img_provider = get_image_provider()
            if _img_provider is not None:
                from tools.media_tools import MediaGenerationTools
                _media = MediaGenerationTools(self.tenant_db, _img_provider, user_context)
                user_registry.register(
                    "generate_image",
                    "Generate an image from a text description and show it to the "
                    "user. Use when the user asks to create, draw, design, or render "
                    "an image. The image is displayed automatically; you only need to "
                    "add a short caption.",
                    {"prompt": "string", "size?": "string"},
                    _media.generate_image,
                    side_effect="write",
                    param_descriptions={
                        "prompt": "Detailed description of the image to generate.",
                        "size": "Optional dimensions, e.g. '1024x1024'.",
                    },
                )
        except Exception:
            logging.getLogger(__name__).debug(
                "media generation tool not registered", exc_info=True)

        skip_brain = user_context is not None
        for tool_name, tool in list(self.tool_registry._tools.items()):
            if tool_name not in self._LOCAL_TOOL_NAMES:
                if skip_brain and tool_name in self._BRAIN_TOOL_NAMES:
                    continue
                user_registry._tools[tool_name] = tool
                if tool_name in self.tool_registry._callbacks:
                    user_registry._callbacks[tool_name] = self.tool_registry._callbacks[tool_name]
                if tool_name in self.tool_registry._policies:
                    user_registry._policies[tool_name] = self.tool_registry._policies[tool_name]
        
        if user_context and hasattr(self, '_brain_subsystem'):
            self._register_scoped_brain_tools(user_registry, user_context)

        if user_context and getattr(self, "_integrations", None):
            try:
                n = self._integrations.register_session_direct(user_registry, user_context)
                if n:
                    import logging
                    logging.getLogger(__name__).info(
                        "create_user_tools: +%d dynamic API tools (app_id=%s)",
                        n, getattr(user_context, "app_id", None))
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug("dynamic API tools skipped: %s", e)

        if allowed_tools is not None:
            allowed_set = set(allowed_tools)
            # Procedural-memory READ tools are engine infrastructure over the
            # tenant's own skills — exempt from persona whitelists (which
            # predate the skills system) so no session is silently blinded to
            # learned skills. Writes (skill_manage) stay whitelist-governed.
            from tools.skills_mcp import SKILL_READ_TOOLS
            allowed_set |= set(SKILL_READ_TOOLS)
            removed = [k for k in user_registry._tools if k not in allowed_set]
            for tool_name in removed:
                user_registry._tools.pop(tool_name, None)
                user_registry._callbacks.pop(tool_name, None)
            if removed:
                import logging
                logging.getLogger(__name__).info(
                    "create_user_tools: filtered %d tools, kept %d (whitelist: %s)",
                    len(removed), len(user_registry._tools), allowed_tools
                )
        
        return user_registry

    def _register_scoped_brain_tools(self, registry, user_context):
        """Re-create brain tools with a resolver bound to a specific UserContext.
        
        This ensures each bot/user gets brain tool closures where:
          - _get_app_id() always returns the correct app_id
          - _get_visible_scopes() returns the correct scopes
          - No dependence on ContextVar propagation through threads
        
        Architecture:
          CLI creates brain tools once → shared resolver → CLI context (no app_id)
          create_user_tools(user_context) → creates FRESH brain tools → fixed resolver
          
        The resolver is a lambda that returns the SAME user_context forever.
        This is correct because bot sessions are tied to one app_id for their lifetime.
        """
        from tools.brain_mcp import BrainMCPTools
        from tools.native_mcp import NativeMCP
        
        bs = self._brain_subsystem
        
        scoped_brain = BrainMCPTools(
            pipeline=bs['pipeline'],
            brain=bs['brain'],
            storage=bs['storage'],
            sync_worker=bs['sync_worker'],
            user_context_resolver=lambda ctx=user_context: ctx,
            chat_history_resolver=lambda: self.agent.conversation_history if hasattr(self, 'agent') else [],
        )
        
        temp_mcp = NativeMCP()
        scoped_brain.register_tools(temp_mcp)
        
        _ext_param_descriptions = {
            "brain_ask": {
                "question": "Natural language question to ask the knowledge base.",
            },
            "brain_search": {
                "query": "Search keywords.",
                "limit": "Max results to return.",
            },
        }
        
        for tool_name, func in temp_mcp._tools.items():
            dense_schema = temp_mcp._dense_schemas[tool_name]
            pdesc = {**dense_schema.get("arg_descriptions", {}),
                     **_ext_param_descriptions.get(tool_name, {})}
            registry.register(
                tool_name, dense_schema.get("desc", ""), dense_schema, func,
                param_descriptions=pdesc
            )
        
        import logging
        logging.getLogger(__name__).info(
            "Scoped brain tools registered for app_id=%s (%d tools)",
            user_context.app_id, len(temp_mcp._tools)
        )
        registry._brain_mcp_instance = scoped_brain

    def _register_core_tools(self):
        """Register all tools."""
        fs = FilesystemTools(self.workspace_path)
        shell = ShellExecutionTool(self.workspace_path, background_manager=self.bg_manager)

        from tools.page_tools import PageTools
        PageTools().register(self.tool_registry)

        self.tool_registry.register(
            "view_file",
            "Read a file's contents with line numbers. Supports any absolute path. Output capped at 800 lines. Use start_line/end_line for large files.",
            {"absolute_path": "string", "start_line?": "int", "end_line?": "int"},
            fs.view_file,
            param_descriptions={
                "absolute_path": "Absolute path to the file to read.",
                "start_line": "First line to show (1-indexed). Omit to start from line 1.",
                "end_line": "Last line to show (1-indexed, inclusive). Omit to show to end of file.",
            }
        )
        self.tool_registry.register(
            "write_to_file",
            "Create a new file or overwrite an existing one. Creates parent directories automatically.",
            {"target_file": "string", "code_content": "string", "overwrite?": "bool"},
            fs.write_to_file,
            param_descriptions={
                "target_file": "Absolute path for the file to create or overwrite.",
                "code_content": "The full content to write to the file.",
                "overwrite": "Set to true to overwrite an existing file. Defaults to false.",
            }
        )
        self.tool_registry.register(
            "list_dir",
            "List contents of a directory showing files with sizes and subdirectories with child counts.",
            {"directory_path": "string"},
            fs.list_dir,
            param_descriptions={
                "directory_path": "Absolute path to the directory to list.",
            }
        )
        self.tool_registry.register(
            "grep_search",
            "Search files for text or regex pattern matches. Returns matching lines with file paths and line numbers (default cap 50, raisable via max_results). Use files_with_matches to cheaply scope which files mention a symbol before reading.",
            {"search_path": "string", "query": "string", "is_regex?": "bool",
             "case_insensitive?": "bool", "includes?": "string",
             "context_lines?": "int", "files_with_matches?": "bool",
             "max_results?": "int"},
            fs.grep_search,
            param_descriptions={
                "search_path": "Absolute path to a file or directory to search.",
                "query": "The text or regex pattern to search for.",
                "is_regex": "If true, treat query as a regex pattern. Defaults to false (literal match).",
                "case_insensitive": "If true, ignore case when matching. Defaults to false.",
                "includes": "Comma-separated glob patterns to filter files, e.g. '*.py,*.js'.",
                "context_lines": "Show this many lines of context around each match (max 10).",
                "files_with_matches": "If true, return only the file paths that contain matches.",
                "max_results": "Raise the match cap (up to 200). Default 50.",
            }
        )
        self.tool_registry.register(
            "glob_files",
            "Find files whose name or relative path matches a glob pattern (e.g. '*.py', 'src/*service*'), sorted newest-modified first. The fast way to LOCATE files before reading them.",
            {"pattern": "string", "path?": "string"},
            fs.glob_files,
            param_descriptions={
                "pattern": "Glob pattern matched against file names and relative paths.",
                "path": "Directory to search under. Defaults to the workspace root.",
            }
        )
        self.tool_registry.register(
            "replace_file_content",
            "Replace text in a file. target_content must match exactly once (use start_line/end_line to disambiguate) unless replace_all is set. The result echoes the edited region — no need to re-read the file to confirm.",
            {"target_file": "string", "target_content": "string", "replacement_content": "string",
             "start_line?": "int", "end_line?": "int", "replace_all?": "bool"},
            fs.replace_file_content,
            param_descriptions={
                "target_file": "Absolute path to the file to modify.",
                "target_content": "The exact text to find and replace. Must match character-for-character.",
                "replacement_content": "The new text to replace target_content with.",
                "start_line": "Narrow search to lines starting from this line number (1-indexed).",
                "end_line": "Narrow search to lines ending at this line number (1-indexed).",
                "replace_all": "If true, replace every occurrence instead of requiring uniqueness.",
            }
        )
        self.tool_registry.register(
            "multi_replace_file_content",
            "Apply multiple non-contiguous text replacements to a file in one operation.",
            {"target_file": "string", "replacements": "string"},
            fs.multi_replace_file_content,
            param_descriptions={
                "target_file": "Absolute path to the file to modify.",
                "replacements": "JSON array of objects: [{\"target\": \"old text\", \"replacement\": \"new text\"}, ...]",
            }
        )

        self.tool_registry.register(
            "run_command",
            "Execute a shell command and return its output with exit code. The working directory PERSISTS across calls (a `cd` carries to the next command). Commands exceeding the timeout are promoted to background tasks (manage them with manage_task; you are notified on completion).",
            {"command_line": "string", "cwd?": "string", "wait_ms_before_async?": "int"},
            shell.run_command,
            param_descriptions={
                "command_line": "The shell command to execute (e.g. 'ls -la', 'git status', 'python3 script.py').",
                "cwd": "Working directory. Omit to use the persistent session directory.",
                "wait_ms_before_async": "Milliseconds to wait before promoting to a background task. Default 120000 (2 min).",
            }
        )

        self.tool_registry.register(
            "schedule",
            "Set a one-shot timer that sends a notification after the specified duration.",
            {"duration_seconds": "int", "prompt": "string"},
            self.scheduler.set_timer,
            param_descriptions={
                "duration_seconds": "Number of seconds to wait before firing the timer.",
                "prompt": "The message to include in the notification when the timer fires.",
            }
        )

        from state.todos import todo_write, TODO_WRITE_JSON_SCHEMA
        self.tool_registry.register(
            "todo_write",
            "Create or update your task list for the current work. Pass the "
            "COMPLETE list every time — it replaces the previous one. Use for "
            "any task with 3 or more steps: mark an item in_progress when you "
            "start it and completed IMMEDIATELY when done. Keep exactly one "
            "item in_progress at a time; add newly discovered work as new items.",
            {},
            todo_write,
            json_schema=TODO_WRITE_JSON_SCHEMA,
        )

        self.tool_registry.register(
            "manage_task",
            "Manage background tasks. Actions: 'list' (show all tasks), 'status' (check a task), 'kill' (terminate a task), 'send_input' (send stdin to a task).",
            {"action": "string", "task_id?": "string", "input?": "string"},
            self._manage_task,
            param_descriptions={
                "action": "One of: 'list', 'status', 'kill', 'send_input'.",
                "task_id": "The task ID to manage. Required for status, kill, and send_input.",
                "input": "The text to send as stdin. Required for send_input.",
            }
        )

        self.tool_registry.register(
            "cron_list",
            "List all scheduled cron jobs with status, next run, and schedule.",
            {},
            self._cron_list_cli,
        )
        self.tool_registry.register(
            "cron_remove",
            "Remove/cancel a cron job by its ID.",
            {"job_id": "string"},
            self._cron_remove_cli,
            param_descriptions={
                "job_id": "The 12-character job ID to remove.",
            }
        )

        self.tool_registry.register(
            "add_memory",
            "Store a fact, observation, or user preference for future reference across conversations.",
            {"memory_item": "string"},
            self.memory.add_memory,
            param_descriptions={
                "memory_item": "The text to remember. Should be a clear, self-contained statement.",
            }
        )
        self.tool_registry.register(
            "get_memories",
            "Retrieve stored memories. Optionally filter by a relevance query to find specific memories.",
            {"query?": "string"},
            self.memory.get_memories,
            param_descriptions={
                "query": "Optional search query to filter memories by relevance. Omit to get most recent.",
            }
        )

        self.tool_registry.register(
            "search_web",
            "Search the web using DuckDuckGo. Returns summarized results with titles, URLs, and snippets.",
            {"query": "string"},
            self._search_web,
            param_descriptions={
                "query": "The search query string.",
            }
        )

        self.tool_registry.register(
            "ask_question",
            "Ask the user a clarifying question when the request is ambiguous. Returns the user's answer as a string.",
            {"question": "string"},
            self._ask_question,
            param_descriptions={
                "question": "The question to ask the user. Be specific about what information you need.",
            }
        )

        from tools.tool_policy import ToolPolicy
        _subagent_policy = ToolPolicy(side_effect="read", idempotent=False,
                                      cacheable=False, timeout_s=900.0, retries=0)
        self.tool_registry.register(
            "delegate_task",
            "Run a sub-agent with its OWN context window and return its report. "
            "Use for BROAD exploration (find every caller of X, survey a "
            "directory, research across documents) so raw reads/searches do not "
            "flood this conversation — agent_name 'explore' is a read-only "
            "investigator built for this. Emit several delegate_task calls in "
            "ONE response to run them in parallel. The task brief must be fully "
            "self-contained (the sub-agent cannot see this conversation) and "
            "must state exactly what the report should contain.",
            {"agent_name": "string", "task": "string"},
            self._delegate_task,
            param_descriptions={
                "agent_name": "Agent to run: 'explore' (read-only investigator) or a registered agent name.",
                "task": "Self-contained brief with all context and the required report contents.",
            },
            policy=_subagent_policy,
        )
        self.tool_registry.register(
            "spawn_agent",
            "Start a sub-agent in the BACKGROUND and return a task id "
            "immediately. Its report is delivered into the conversation when "
            "it finishes. Use only for long-running work that should proceed "
            "while you continue; for normal exploration use delegate_task.",
            {"agent_name": "string", "task": "string"},
            self._spawn_agent,
            param_descriptions={
                "agent_name": "Agent to run: 'explore' or a registered agent name.",
                "task": "Self-contained brief with all context and the required report contents.",
            },
            policy=ToolPolicy(side_effect="read", idempotent=False,
                              cacheable=False, timeout_s=60.0, retries=0),
        )


    def _register_external_tools(self):
        from tools.database import DatabaseTools
        from tools.native_mcp import NativeMCP
        from tools.microsoft365 import Microsoft365Tools
        
        neo4j_uri = app_config.get("neo4j.uri", "bolt://localhost:7687")
        neo4j_user = app_config.get("neo4j.user", "neo4j")
        neo4j_pass = app_config.get("neo4j.password", "")
        es_url = app_config.get("elasticsearch.url", "http://localhost:9200")
        from core.tenant_context import data_namespace
        _ns = data_namespace()
        es_index = f"brain_{_ns}" if _ns else app_config.get("elasticsearch.index", "brain")

        db_tools = DatabaseTools(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_pass=neo4j_pass,
            es_url=es_url,
            es_index=es_index
        )
        
        from core.user_context import current_user_context
        _user_ctx_resolver = lambda: current_user_context.get() or (self.agent.user_context if hasattr(self, 'agent') else None)

        m365_tools = Microsoft365Tools(
            credential_registry=self.connections,
            user_context_resolver=_user_ctx_resolver,
            db_resolver=lambda: self.tenant_db,
        )
        
        native_mcp = NativeMCP()
        db_tools.register_tools(native_mcp)
        m365_tools.register_tools(native_mcp)

        memory_tools = MemoryMCPTools(
            self.memory,
            user_id_resolver=lambda: (current_user_context.get().user_id if current_user_context.get() else self.agent.user_context.user_id) if hasattr(self, 'agent') else "",
        )
        memory_tools.set_user_context_resolver(_user_ctx_resolver)
        memory_tools.register_tools(native_mcp)

        # Skills (procedural memory). Store attaches later via the api
        # layer's attach_skills seam; until then the tools answer
        # "not available" — never an error.
        from tools.skills_mcp import SkillsMCPTools
        SkillsMCPTools(
            store_resolver=lambda: getattr(self, "skill_store", None),
        ).register_tools(native_mcp)

        try:
            from brain.storage.manager import StorageManager
            from brain.ingestion.pipeline import IngestionPipeline
            from brain.llm_adapter import BrainLLMAdapter
            from tools.brain_mcp import BrainMCPTools

            from core.tenant_context import multi_tenant_enabled, data_namespace
            _ns = data_namespace()
            _default_graph_tid = _ns or ("default" if multi_tenant_enabled() else None)
            brain_storage = StorageManager(
                db=self.tenant_db,
                neo4j_uri=app_config.get("neo4j.uri"),
                neo4j_user=app_config.get("neo4j.user"),
                neo4j_password=app_config.get("neo4j.password"),
                elasticsearch_url=app_config.get("elasticsearch.url"),
                elasticsearch_index=(f"brain_{_ns}" if _ns
                                     else app_config.get("elasticsearch.index", "brain")),
                graph_tenant_id=_default_graph_tid,
            )
            
            brain_storage.connect()

            brain_llm = BrainLLMAdapter(self.llm)

            brain_pipeline = IngestionPipeline(
                storage=brain_storage,
                llm_client=brain_llm,
                enable_graph_extraction=True,
            )

            brain_instance = None
            try:
                from brain.retrieval.brain import Brain
                brain_instance = Brain(storage=brain_storage, llm_client=brain_llm)
            except Exception as e:
                import logging; logging.getLogger(__name__).debug(
                    "Brain retrieval init skipped (non-critical): %s", e)

            sync_worker_instance = None
            try:
                from brain.ingestion.sync_worker import SyncWorker
                from brain.ingestion.connectors.jira_connector import JiraConnector
                from brain.ingestion.connectors.microsoft.outlook_connector import OutlookConnector
                from brain.ingestion.connectors.slack_connector import SlackConnector

                connectors = [
                    JiraConnector(app_config=app_config),
                    OutlookConnector(app_config=app_config,
                                     credential_registry=self.connections),
                    SlackConnector(),
                ]
                sync_worker_instance = SyncWorker(
                    document_store=brain_storage.documents,
                    content_store=brain_storage.content_cache,
                    entity_index=brain_storage.entity_index,
                    connectors=connectors,
                    llm_client=brain_llm,
                    db=self.tenant_db,
                )
            except Exception as e:
                import logging; logging.getLogger(__name__).debug(
                    "SyncWorker init skipped (non-critical): %s", e)

            brain_tools = BrainMCPTools(
                pipeline=brain_pipeline,
                brain=brain_instance,
                storage=brain_storage,
                sync_worker=sync_worker_instance,
                user_context_resolver=lambda: current_user_context.get() or (self.agent.user_context if hasattr(self, 'agent') else None),
                chat_history_resolver=lambda: self.agent.conversation_history if hasattr(self, 'agent') else [],
            )
            brain_tools.register_tools(native_mcp)

            self._brain_subsystem = {
                'pipeline': brain_pipeline,
                'brain': brain_instance,
                'storage': brain_storage,
                'sync_worker': sync_worker_instance,
            }

            if sync_worker_instance and sync_worker_instance.has_configured_sources():
                try:
                    from cron.jobs import create_job, list_jobs
                    existing = [j for j in list_jobs() if j.get("name") == "brain_auto_sync"]
                    if not existing:
                        job = create_job(
                            # `[TOOL:<name>] <json args>` → the gateway executes
                            # this tool DIRECTLY (no LLM). brain_sync is a
                            # deterministic op, so an LLM agent to "decide" to
                            # call it is pure overhead.
                            prompt='[TOOL:brain_sync] {"source": "all"}',
                            schedule="every 30m",
                            name="brain_auto_sync",
                        )
                        import logging; logging.getLogger(__name__).info(
                            "Auto-sync cron created: every 30m (sources: %s)",
                            ", ".join(sync_worker_instance.get_configured_sources())
                        )
                except Exception as e:
                    import logging; logging.getLogger(__name__).debug(
                        "Auto-sync cron setup skipped: %s", e)

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Brain tools init failed: %s", e)

        # Skill-library curator (P4): a weekly system cron that prunes/merges/
        # fixes the skills the reviewer writes. Seeded ONLY when learning is on
        # for this tenant — the plan's #1 rot-defense, shipped with the loop,
        # not after. Idempotent: created once per tenant DB.
        try:
            if app_config.get_bool("agent.skill_learning", False):
                from cron.jobs import create_job, list_jobs
                if not any(j.get("name") == "skill_curator"
                           for j in list_jobs(include_disabled=True)):
                    create_job(
                        prompt="[CURATOR] Maintain the skill library.",
                        schedule="every 7d",
                        name="skill_curator",
                    )
                    import logging; logging.getLogger(__name__).info(
                        "Skill curator cron created: every 7d")
        except Exception as e:
            import logging; logging.getLogger(__name__).debug(
                "Curator cron setup skipped: %s", e)

        try:
            from agents.registry import AgentRegistry
            from agents.factory import AgentFactory
            from tools.agent_tools import AgentMCPTools

            agent_registry = AgentRegistry(self.tenant_db)
            agent_factory = AgentFactory(self, agent_registry)

            agent_tools = AgentMCPTools(
                registry=agent_registry,
                factory=agent_factory,
                user_context_resolver=lambda: current_user_context.get() or (self.agent.user_context if hasattr(self, 'agent') else None),
            )
            agent_tools.register_tools(native_mcp)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Agent tools init skipped: %s", e)

        try:
            from tools.outlook_actions import OutlookActionTools
            outlook_actions = OutlookActionTools(
                credential_registry=self.connections,
                user_context_resolver=_user_ctx_resolver,
            )
            outlook_actions.register_tools(native_mcp)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Outlook action tools skipped: %s", e)

        try:
            from tools.teams_messaging import TeamsMessagingTools
            teams_messaging = TeamsMessagingTools(
                credential_registry=self.connections,
                user_context_resolver=_user_ctx_resolver,
            )
            teams_messaging.register_tools(native_mcp)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Teams messaging tools skipped: %s", e)

        try:
            from tools.smtp_tools import SmtpActionTools
            smtp_tools = SmtpActionTools(
                credential_registry=self.connections,
                user_context_resolver=_user_ctx_resolver,
            )
            smtp_tools.register_tools(native_mcp)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("SMTP tools skipped: %s", e)

        try:
            from tools.imap_tools import ImapActionTools
            imap_tools = ImapActionTools(
                credential_registry=self.connections,
                user_context_resolver=_user_ctx_resolver,
            )
            imap_tools.register_tools(native_mcp)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("IMAP tools skipped: %s", e)

        try:
            from tools.jira_tools import JiraActionTools
            jira_tools = JiraActionTools(
                credential_registry=self.connections,
                user_context_resolver=_user_ctx_resolver,
            )
            jira_tools.register_tools(native_mcp)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Jira tools skipped: %s", e)

        try:
            from tools.git_tools import GitActionTools
            git_tools = GitActionTools()
            git_tools.register_tools(native_mcp)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Git tools skipped: %s", e)

        _ext_param_descriptions = {
            "query_neo4j": {"cypher_query": "A Cypher query to execute against the Neo4j knowledge graph."},
            "search_elasticsearch": {"query": "The text query to search for in Elasticsearch."},
            "fetch_recent_emails": {"limit": "Maximum number of recent emails to fetch. Default 10."},
            "search_emails": {
                "query": "Search query to filter emails by subject or body.",
                "limit": "Maximum number of results to return. Default 10.",
            },
            "brain_ingest_file": {"file_path": "Absolute path to the file to ingest into the knowledge base."},
            "brain_ingest_directory": {
                "directory": "Absolute path to the directory to ingest.",
                "recursive": "Whether to recurse into subdirectories (true/false).",
            },
            "brain_ingest_text": {
                "content": "Raw text content to ingest.",
                "title": "Optional human-readable title.",
                "document_type": "Type: email, policy, code, document, meeting_notes, ticket.",
            },
            "brain_ask": {"question": "The question to ask the brain."},
            "brain_search": {
                "query": "Search keywords.",
                "limit": "Max results to return.",
            },
        }
        
        for tool_name, func in native_mcp._tools.items():
            dense_schema = native_mcp._dense_schemas[tool_name]
            pdesc = {**dense_schema.get("arg_descriptions", {}),
                     **_ext_param_descriptions.get(tool_name, {})}
            self.tool_registry.register(
                tool_name, dense_schema.get("desc", ""), dense_schema, func,
                param_descriptions=pdesc
            )

    def refresh_integrations(self) -> int:
        """Re-run external-tool registration to pick up config/credential changes.

        Credential-gated tools (Jira, M365, SMTP/IMAP) only register when a
        credential exists at registration time, and infra tools (Neo4j/ES) bake
        their connection settings in at construction. Both were therefore frozen
        at process startup — connecting a brand-new provider, or changing an
        infra URL, needed a server restart. This re-runs registration in place so
        those become available without one.

        Safe to call on the live singleton: NativeMCP is purely in-process (no
        subprocesses), register() overwrites by tool name (no duplicates), and
        create_user_tools() iterates a snapshot so a concurrent refresh can't
        corrupt an in-flight session build. Idempotent and best-effort —
        exceptions are swallowed so a bad credential can't wedge the registry.

        Returns the resulting server-tool count (0 on failure).
        """
        import logging
        _log = logging.getLogger(__name__)
        try:
            self._register_external_tools()
            if getattr(self, "_integrations", None):
                try:
                    self._integrations._cache.clear()
                    self._integrations.executor._op_cache.clear()
                except Exception:
                    pass
            # The live AgentLoop caches the OpenAI tool schemas once per
            # session — newly registered tools would otherwise stay invisible
            # to it until restart.
            if getattr(self, "agent", None) is not None:
                self.agent._openai_tools_cache = None
            count = len(self.tool_registry._tools)
            _log.info("refresh_integrations: external tools re-registered (registry now %d tools)", count)
            return count
        except Exception as e:
            _log.error("refresh_integrations failed: %s", e, exc_info=True)
            return 0


    def _cron_list_cli(self) -> str:
        """List all cron jobs (CLI mode)."""
        from cron.jobs import list_jobs
        jobs = list_jobs(include_disabled=True)
        if not jobs:
            return "No cron jobs scheduled."
        lines = ["**Scheduled Jobs:**\n"]
        for job in jobs:
            status = "✅" if job.get("enabled") else "⏸️"
            lines.append(
                f"{status} **{job.get('name', job['id'])}** (ID: `{job['id']}`)\n"
                f"   Schedule: {job.get('schedule_display', '?')}\n"
                f"   State: {job.get('state', '?')} | Next: {job.get('next_run_at', 'N/A')}\n"
                f"   Last: {job.get('last_status', 'never')}"
            )
        return "\n".join(lines)

    def _cron_remove_cli(self, job_id: str) -> str:
        """Remove a cron job (CLI mode)."""
        from cron.jobs import remove_job
        if remove_job(job_id):
            return f"✅ Job `{job_id}` removed."
        return f"❌ Job `{job_id}` not found."


    def _manage_task(self, action: str, task_id: str = None, input: str = None) -> str:
        """Dispatch manage_task actions to BackgroundTaskManager."""
        action = action.lower().strip()
        
        if action == "list":
            return self.bg_manager.list_tasks()
        elif action == "status":
            if not task_id:
                return "Error: task_id required for 'status' action."
            return self.bg_manager.get_status(task_id)
        elif action == "kill":
            if not task_id:
                return "Error: task_id required for 'kill' action."
            return self.bg_manager.kill_task(task_id)
        elif action == "send_input":
            if not task_id:
                return "Error: task_id required for 'send_input' action."
            if not input:
                return "Error: input required for 'send_input' action."
            # Tool callbacks run in a worker thread (asyncio.to_thread) — there
            # is no running loop here; schedule onto the main loop instead.
            loop = getattr(self.scheduler, "_loop", None)
            try:
                if loop is not None and loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        self.bg_manager.send_input(task_id, input), loop
                    )
                    return future.result(timeout=5)
                return asyncio.run(self.bg_manager.send_input(task_id, input))
            except Exception as e:
                return f"Error sending input: {e}"
        else:
            return f"Error: Unknown action '{action}'. Use: list, status, kill, send_input."

    def _search_web(self, query: str) -> str:
        """Search the web via DuckDuckGo HTML. No API key needed."""
        try:
            url = "https://html.duckduckgo.com/html/"
            resp = requests.post(url, data={"q": query}, timeout=10,
                                 headers={"User-Agent": "Agent/1.0"})
            resp.raise_for_status()
            
            import re as _re
            results = []
            links = _re.findall(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', resp.text)
            snippets = _re.findall(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', resp.text)
            
            for i, (href, title) in enumerate(links[:5]):
                clean_title = _re.sub(r'<[^>]+>', '', title).strip()
                clean_snippet = _re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
                results.append(f"{i+1}. {clean_title}\n   {href}\n   {clean_snippet}")
            
            if not results:
                return f"No results found for '{query}'."
            
            return f"Search results for '{query}':\n\n" + "\n\n".join(results)
        except Exception as e:
            return f"Search error: {e}"

    def _ask_question(self, question: str) -> str:
        """Present a question to the user and return their answer."""
        print(f"\n  [Question] {question}")
        try:
            answer = input("  Your answer> ").strip()
            return answer if answer else "(No answer provided)"
        except (EOFError, KeyboardInterrupt):
            return "(User declined to answer)"

    def _resolve_subagent(self, agent_name: str, task: str):
        """Shared resolution for delegate_task/spawn_agent.

        Returns (factory, definition, None) or (None, None, error_string).
        """
        from agents.task_manager import SubAgentManager
        blocked = SubAgentManager.check_depth()
        if blocked:
            return None, None, blocked
        if not agent_name or not task:
            return None, None, "✗ Both 'agent_name' and 'task' are required."
        try:
            from agents.registry import AgentRegistry
            from agents.factory import (AgentFactory, BUILTIN_AGENTS,
                                        resolve_agent_definition)
        except Exception as e:
            return None, None, f"✗ Agent subsystem unavailable: {e}"
        registry = AgentRegistry(self.tenant_db)
        agent_def = resolve_agent_definition(registry, agent_name)
        if not agent_def:
            try:
                names = [d.name for d in registry.list_all()]
            except Exception:
                names = []
            names += sorted(BUILTIN_AGENTS.keys())
            return None, None, (f"✗ Agent '{agent_name}' not found. "
                                f"Available agents: {', '.join(names) or '(none)'}")
        return AgentFactory(self, registry), agent_def, None

    async def _delegate_task(self, agent_name: str, task: str) -> str:
        """Foreground sub-agent (async tool — runs ON the event loop).

        Several delegate_task calls emitted in one batch execute as
        genuinely PARALLEL sub-agents, each with its own context window;
        only their reports enter this conversation.
        """
        factory, agent_def, err = self._resolve_subagent(agent_name, task)
        if err:
            return err
        try:
            result = await self._subagents.run(factory, agent_def, task)
            return f"[delegated to {agent_name}]\n{result}"
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("delegate_task failed: %s", e,
                                              exc_info=True)
            return f"✗ Delegation to '{agent_name}' failed: {e}"

    async def _spawn_agent(self, agent_name: str, task: str) -> str:
        """Background sub-agent: returns immediately with a task id; the
        report is injected into the conversation when it completes."""
        factory, agent_def, err = self._resolve_subagent(agent_name, task)
        if err:
            return err
        try:
            task_id = self._subagents.spawn(factory, agent_def, task)
            return (f"Sub-agent task '{task_id}' started (agent: {agent_name}). "
                    f"You will be notified with its report when it completes — "
                    f"continue other work meanwhile.")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("spawn_agent failed: %s", e,
                                              exc_info=True)
            return f"✗ Spawning '{agent_name}' failed: {e}"

    async def _handle_subagent_completed(self, payload: dict):
        """Console notification only — the AgentLoop subscribes to
        `subagent_completed` on its own bus and injects the report itself
        (re-publishing user_input here would double-process it)."""
        print(f"\n  [Sub-agent {payload.get('task_id', '?')}] "
              f"({payload.get('agent_name', '?')}) completed")

    async def run(self):
        """Main REPL loop with streaming and session summary."""
        self.scheduler.bind_loop()
        tool_count = len(self.tool_registry._tools)
        prompt_tokens = self.agent.prompt_builder.get_prompt_token_estimate()
        
        print(f"\n  Model: {self.llm.model}")
        print(f"  Tools: {tool_count} (native function calling)")
        print(f"  Prompt: ~{prompt_tokens} tokens")
        print(f"  Budget: ~{self.agent.history_token_budget} tokens")
        print(f"  Stream: {'ON' if self.llm.stream else 'OFF'}")
        print(f"  Max iterations: {self.agent.MAX_TOOL_ITERATIONS}")
        print(f"\nType '/help' for commands. 'exit' to quit.\n")
        
        while True:
            try:
                user_in = await asyncio.to_thread(input, "User> ")
                stripped = user_in.strip()
                
                if stripped.lower() in ('exit', 'quit'):
                    self._save_session()
                    self.scheduler.cancel_all()
                    break
                
                if stripped.startswith('/') and ' ' not in stripped and len(stripped) < 20:
                    self._handle_command(stripped)
                    continue
                
                if not stripped:
                    continue
                
                if not self.llm.stream:
                    print("  Thinking...")
                
                await self.event_bus.publish('user_input', {"user_input": stripped})
                
            except KeyboardInterrupt:
                self._save_session()
                self.scheduler.cancel_all()
                break
            except EOFError:
                self._save_session()
                break
    
    def _handle_command(self, cmd: str):
        """Handle slash commands."""
        if cmd == '/help':
            self._print_help()
        elif cmd == '/stats':
            self._print_stats()
        elif cmd == '/clear':
            self.agent.conversation_history.clear()
            self.agent._total_history_tokens = 0
            print("  History cleared.")
        elif cmd == '/tools':
            for t in sorted(self.tool_registry._tools.keys()):
                desc = self.tool_registry._tools[t].get("description", "")
                print(f"  {t}: {desc}")
        elif cmd == '/stream':
            self.llm.stream = not self.llm.stream
            print(f"  Streaming: {'ON' if self.llm.stream else 'OFF'}")
        elif cmd == '/memory':
            self._print_memories()
        elif cmd == '/checkpoint':
            self._print_checkpoints()
        else:
            print(f"  Unknown command: {cmd}. Type /help for available commands.")

    def _save_session(self):
        """Save conversation summary on exit."""
        summary = self.agent.generate_session_summary()
        if summary:
            result = self.summaries.save_summary(summary)
            print(f"  {result}")
    
    def _print_stats(self):
        print(f"  LLM: {self.llm.get_token_stats()}")
        print(f"  History: {len(self.agent.conversation_history)} messages "
              f"(~{self.agent._estimate_history_tokens()} tokens)")
        mem_stats = self.memory.get_stats()
        print(f"  Memories: {mem_stats['total_memories']} ({mem_stats['db_size_human']})")
        print(f"  Checkpoints: {len(self.agent.checkpoint_mgr.checkpoints)}")
        print(f"  Compactions: {self.agent._compaction_count}")
        print(f"  Timers: {len(self.scheduler.active_timers)}")
    
    def _print_memories(self):
        """Show all stored memories with categories."""
        rows = self.memory._conn.execute(
            "SELECT id, category, layer, key, value, created_at FROM memories ORDER BY last_accessed DESC LIMIT 30"
        ).fetchall()
        if not rows:
            print("  No memories stored.")
            return
        for row in rows:
            ts = (row['created_at'] or '?')[:10]
            cat = row['category']
            layer = row['layer']
            val = (row['value'] or '')[:80]
            print(f"  [{row['id']}] ({cat}/{layer}) ({ts}) {val}")
    
    def _print_checkpoints(self):
        """Show checkpoint history."""
        cps = self.agent.checkpoint_mgr.checkpoints
        if not cps:
            print("  No checkpoints yet.")
            return
        for cp in cps:
            files = ', '.join(cp.get('files_modified', [])[:5])
            print(f"  [iter {cp['iteration']}] {cp['tool_calls_count']} tools | files: {files}")
    
    def _print_help(self):
        print("""
  Commands:
    /help       — Show this help
    /stats      — Token usage, memory count, checkpoints
    /tools      — List all registered tools
    /memory     — Show stored memories
    /checkpoint — Show checkpoint history
    /clear      — Clear conversation history
    /stream     — Toggle response streaming ON/OFF
    exit        — Save session summary and quit
""")
