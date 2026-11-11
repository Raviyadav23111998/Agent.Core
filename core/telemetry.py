"""
Telemetry — metrics, cost accounting, and lightweight spans.
═══════════════════════════════════════════════════════════════

Dependency-free observability (no OpenTelemetry/Prometheus client required —
those can be layered on later). Provides:

  - Counters & histograms (thread-safe) with label sets.
  - Per-model token + COST accounting (input/output/cache-read priced
    separately, since cached input is ~10% on Claude).
  - A `span()` context manager that records latency + success/failure.
  - Prometheus text exposition via render_prometheus() for /metrics.

Everything is in-process and best-effort: telemetry must NEVER break a
request. All record calls swallow their own errors.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

logger = logging.getLogger("agent.telemetry")


# (input, output, cache_read) USD per 1M tokens — list-price snapshots
# (2026-07). Override any model without a deploy via config:
#   llm.price.<model> = "in,out,cache_read"
# Matching is substring on the model id (router prefixes stripped), first
# hit wins — keep more-specific keys above generic family keys.
_DEFAULT_PRICING: Dict[str, Tuple[float, float, float]] = {
    "claude-fable-5":      (3.00, 15.00, 0.30),
    "claude-opus-4-8":     (15.00, 75.00, 1.50),
    "claude-sonnet-4-6":   (3.00, 15.00, 0.30),
    "claude-haiku-4-5":    (0.80, 4.00, 0.08),
    "gpt-4o-mini":         (0.15, 0.60, 0.075),
    "gpt-4o":              (2.50, 10.00, 1.25),
    "text-embedding-3-small": (0.02, 0.0, 0.0),
    "deepseek-v4-flash":   (0.14, 0.28, 0.0028),   # verified 2026-07-04 (live cache test)
    "deepseek-v4-pro":     (0.56, 1.68, 0.014),
    "deepseek-chat":       (0.20, 0.80, 0.05),
    "deepseek":            (0.20, 0.80, 0.05),
    "glm-5.2":             (1.40, 4.40, 0.26),     # verified 2026-07-04 (z.ai list)
    "glm":                 (1.40, 4.40, 0.26),
    "gemini-2.5-flash":    (0.30, 2.50, 0.075),
    "gemini-2.5-pro":      (1.25, 10.00, 0.31),
    "gemini":              (1.25, 10.00, 0.31),
    "qwen-max":            (1.60, 6.40, 0.16),
    "qwen":                (0.80, 2.40, 0.08),
    "kimi":                (0.60, 2.50, 0.15),
    "moonshot":            (0.60, 2.50, 0.15),
    "grok":                (3.00, 15.00, 0.75),
    "mistral-large":       (2.00, 6.00, 0.50),
    "mistral":             (0.40, 1.20, 0.10),
    "llama":               (0.60, 0.60, 0.06),
}


# ── DeepSeek, billed DIRECT ──────────────────────────────────────────────
# A profile whose provider is `deepseek` (or whose URL is api.deepseek.com)
# is billed by DeepSeek itself, on DeepSeek's list — not the router prices
# above, which are what OpenRouter charges for the same weights and are a
# different number. Since 2026-08-17 DeepSeek prices by time of day: these
# are the OFF-PEAK rates, and a call inside a peak window costs twice as
# much (api-docs.deepseek.com/quick_start/pricing, read 2026-10-10).
#   in, out, cache_read — USD per 1M tokens, off-peak
_DEEPSEEK_DIRECT_OFFPEAK: Dict[str, Tuple[float, float, float]] = {
    "deepseek-v4-pro":   (0.66, 1.98, 0.022),
    "deepseek-flash":    (0.15, 0.60, 0.003),
    # The legacy name is still accepted and now routes to, and bills as, Flash.
    "deepseek-v4-flash": (0.15, 0.60, 0.003),
}
_DEEPSEEK_PEAK_MULTIPLIER = 2.0

# Peak: Monday–Friday, 01:00–04:00 and 06:00–10:00 UTC (09:00–12:00 and
# 14:00–18:00 Beijing). Weekends and Chinese public holidays are off-peak all
# day. The holidays move every year and DeepSeek publishes no list, so they
# are config, as Beijing dates:
#   llm.deepseek.offpeak_dates = "2027-01-01,2027-02-06,…"
# A holiday left out is priced as peak — the ledger errs high, never low.
#   llm.deepseek.peak_utc = "01:00-04:00,06:00-10:00"   (override, if it moves)
_DEEPSEEK_PEAK_UTC_DEFAULT = "01:00-04:00,06:00-10:00"


def is_deepseek_direct(provider: str = None, url: str = None) -> bool:
    return ((provider or "").strip().lower() == "deepseek"
            or "api.deepseek.com" in (url or "").lower())


def _config(key: str) -> str:
    try:
        from core.config import app_config
        return (app_config.get(key) or "").strip()
    except Exception:
        return ""


def deepseek_is_peak(at=None) -> bool:
    """Whether DeepSeek bills `at` (a UTC datetime; now when None) at peak."""
    import datetime as _dt
    now = at or _dt.datetime.now(_dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_dt.timezone.utc)
    now = now.astimezone(_dt.timezone.utc)
    beijing = (now + _dt.timedelta(hours=8)).date()
    if beijing.weekday() >= 5:
        return False
    holidays = {d.strip() for d in _config("llm.deepseek.offpeak_dates").split(",") if d.strip()}
    if beijing.isoformat() in holidays:
        return False
    minute = now.hour * 60 + now.minute
    for window in (_config("llm.deepseek.peak_utc") or _DEEPSEEK_PEAK_UTC_DEFAULT).split(","):
        try:
            start, end = window.strip().split("-")
            sh, sm = (int(x) for x in start.split(":"))
            eh, em = (int(x) for x in end.split(":"))
        except ValueError:
            continue
        if sh * 60 + sm <= minute < eh * 60 + em:
            return True
    return False


def deepseek_direct_price(model: str, at=None) -> Optional[Tuple[float, float, float]]:
    """DeepSeek's own price for `model` at `at`, or None for a model it does
    not list (the caller then falls back to the general table)."""
    m = (model or "").lower().split("/")[-1]
    for key, price in _DEEPSEEK_DIRECT_OFFPEAK.items():
        if key in m:
            if deepseek_is_peak(at):
                return tuple(p * _DEEPSEEK_PEAK_MULTIPLIER for p in price)
            return price
    return None


def cached_prompt_tokens(usage: dict) -> int:
    """Prompt tokens served from the provider's cache, whatever it calls them:
    OpenAI/OpenRouter `prompt_tokens_details.cached_tokens`, Anthropic
    `cache_read_input_tokens`, DeepSeek `prompt_cache_hit_tokens`."""
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens")
               or usage.get("cache_read_input_tokens")
               or usage.get("prompt_cache_hit_tokens")
               or usage.get("cached_tokens") or 0)


_unpriced_warned: set = set()


def _price_for(model: str) -> Tuple[float, float, float]:
    """Resolve (input, output, cache_read) price per 1M tokens for a model."""
    if not model:
        return (0.0, 0.0, 0.0)
    m = model.lower()
    if "/" in m:
        m = m.split("/")[-1]
    for key, price in _DEFAULT_PRICING.items():
        if key in m:
            return price
    if m not in _unpriced_warned:
        # $0 pricing makes USD budgets silently blind for this model — say
        # so once so admins add an `llm.price.<model>` config override.
        _unpriced_warned.add(m)
        import logging
        logging.getLogger(__name__).warning(
            "telemetry: no price for model %r — cost recorded as $0 "
            "(set config 'llm.price.%s' = 'in,out,cache_read' per 1M)",
            model, model)
    return (0.0, 0.0, 0.0)


class _Metric:
    """A single counter or histogram series, label-keyed."""

    def __init__(self, kind: str):
        self.kind = kind
        self.series: Dict[Tuple, object] = {}


_LAT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120)


class Telemetry:
    """In-process metrics registry with cost accounting."""

    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: Dict[str, _Metric] = {}
        self._cost_usd = 0.0
        self._tokens = {"input": 0, "output": 0, "cache_read": 0}

    def incr(self, name: str, value: float = 1.0, **labels):
        try:
            with self._lock:
                m = self._metrics.setdefault(name, _Metric("counter"))
                key = self._label_key(labels)
                m.series[key] = (m.series.get(key, 0) or 0) + value
        except Exception:
            pass

    def observe(self, name: str, value: float, **labels):
        try:
            with self._lock:
                m = self._metrics.setdefault(name, _Metric("histogram"))
                key = self._label_key(labels)
                rec = m.series.get(key)
                if rec is None:
                    rec = [0, 0.0] + [0] * len(_LAT_BUCKETS)
                    m.series[key] = rec
                rec[0] += 1
                rec[1] += value
                for i, b in enumerate(_LAT_BUCKETS):
                    if value <= b:
                        rec[2 + i] += 1
        except Exception:
            pass

    def record_llm_usage(self, model: str, usage: dict,
                         provider: str = None, url: str = None):
        """Record token usage and compute cost for one LLM call."""
        try:
            if not usage:
                return
            pin = usage.get("prompt_tokens", 0) or 0
            pout = usage.get("completion_tokens", 0) or 0
            cache_read = min(cached_prompt_tokens(usage), pin)
            uncached_in = max(pin - cache_read, 0)

            in_p, out_p, cr_p = self._resolve_price(model, provider=provider, url=url)
            cost = (uncached_in * in_p + pout * out_p + cache_read * cr_p) / 1_000_000.0

            with self._lock:
                self._tokens["input"] += uncached_in
                self._tokens["output"] += pout
                self._tokens["cache_read"] += cache_read
                self._cost_usd += cost
            self.incr("llm_tokens_input_total", uncached_in, model=model)
            self.incr("llm_tokens_output_total", pout, model=model)
            if cache_read:
                self.incr("llm_tokens_cache_read_total", cache_read, model=model)
            self.incr("llm_cost_usd_total", cost, model=model)
        except Exception:
            pass

    def _resolve_price(self, model: str, provider: str = None,
                       url: str = None, at=None):
        """(in, out, cache_read) per 1M for one call. An `llm.price.<model>`
        override wins; then, for a DeepSeek-direct client, DeepSeek's own
        time-of-day price; then the general table."""
        try:
            from core.config import app_config
            override = app_config.get(f"llm.price.{model}")
            if override:
                parts = [float(x) for x in override.split(",")]
                if len(parts) == 3:
                    return tuple(parts)
        except Exception:
            pass
        if is_deepseek_direct(provider, url):
            direct = deepseek_direct_price(model, at)
            if direct is not None:
                return direct
        return _price_for(model)

    @property
    def total_cost_usd(self) -> float:
        return round(self._cost_usd, 6)

    @property
    def token_totals(self) -> dict:
        return dict(self._tokens)

    @contextmanager
    def span(self, name: str, **labels):
        """Time a block; record latency histogram + success/error counter."""
        start = time.perf_counter()
        ok = True
        try:
            yield
        except Exception:
            ok = False
            raise
        finally:
            dur = time.perf_counter() - start
            self.observe(f"{name}_duration_seconds", dur, **labels)
            self.incr(f"{name}_total", 1, status=("ok" if ok else "error"), **labels)

    def render_prometheus(self) -> str:
        lines = []
        with self._lock:
            lines.append(f"# HELP agent_llm_cost_usd_total Cumulative LLM cost in USD")
            lines.append(f"# TYPE agent_llm_cost_usd_total counter")
            lines.append(f"agent_llm_cost_usd_total {self._cost_usd}")
            for tk, tv in self._tokens.items():
                lines.append(f"agent_llm_tokens_total{{kind=\"{tk}\"}} {tv}")
            for name, metric in self._metrics.items():
                safe = f"agent_{name}"
                if metric.kind == "counter":
                    lines.append(f"# TYPE {safe} counter")
                    for key, val in metric.series.items():
                        lines.append(f"{safe}{self._fmt_labels(key)} {val}")
                else:
                    lines.append(f"# TYPE {safe} histogram")
                    for key, rec in metric.series.items():
                        count, total = rec[0], rec[1]
                        for i, b in enumerate(_LAT_BUCKETS):
                            lbl = self._fmt_labels(key, extra=("le", str(b)))
                            lines.append(f"{safe}_bucket{lbl} {rec[2 + i]}")
                        inf = self._fmt_labels(key, extra=("le", "+Inf"))
                        lines.append(f"{safe}_bucket{inf} {count}")
                        lines.append(f"{safe}_count{self._fmt_labels(key)} {count}")
                        lines.append(f"{safe}_sum{self._fmt_labels(key)} {total}")
        return "\n".join(lines) + "\n"

    def snapshot(self) -> dict:
        with self._lock:
            counters = {}
            for name, metric in self._metrics.items():
                if metric.kind == "counter":
                    counters[name] = {self._key_str(k): v for k, v in metric.series.items()}
            return {
                "cost_usd": round(self._cost_usd, 6),
                "tokens": dict(self._tokens),
                "counters": counters,
            }

    @staticmethod
    def _label_key(labels: dict) -> Tuple:
        return tuple(sorted((str(k), str(v)) for k, v in labels.items()))

    @staticmethod
    def _key_str(key: Tuple) -> str:
        return ",".join(f"{k}={v}" for k, v in key) or "_"

    @staticmethod
    def _fmt_labels(key: Tuple, extra: Optional[Tuple[str, str]] = None) -> str:
        pairs = list(key)
        if extra:
            pairs.append(extra)
        if not pairs:
            return ""
        inner = ",".join(f'{k}="{v}"' for k, v in pairs)
        return "{" + inner + "}"


telemetry = Telemetry()
