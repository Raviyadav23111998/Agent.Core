"""
Login rate limiting & account lockout.
═════════════════════════════════════════════════════

Defends /Login against brute force and credential stuffing:
  - Sliding-window attempt counting per key (username or IP).
  - Temporary lockout after too many failures.
  - Successful login clears the key.

Backend (roadmap K.2): when `redis.url` is configured, state lives in Redis
so the limit holds across ALL workers/processes; otherwise the original
in-process tracking applies (best-effort, per worker). Redis failures
degrade to in-process — never to "no limiting" and never to a crash.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Tuple


class LoginRateLimiter:
    """Sliding-window failure tracker with auto-expiring lockout.

    Keyed on username (lowercased). NOTE: pure username keying allows an
    "account-lockout DoS" — an attacker who knows a username can trigger the
    cooldown for the real user. We accept this standard tradeoff because the
    lockout AUTO-EXPIRES (it's a temporary cooldown, not a permanent lock) and
    the alternative (no throttle) leaves online password guessing wide open.
    Thresholds are config-driven (auth.login_*).
    """

    def __init__(self, max_failures: int = 5, window_s: int = 300,
                 lockout_s: int = 900):
        self.max_failures = max_failures
        self.window_s = window_s
        self.lockout_s = lockout_s
        self._lock = threading.Lock()
        self._failures: Dict[str, Deque[float]] = {}
        self._locked_until: Dict[str, float] = {}
        self._configured = False

    def _ensure_configured(self):
        """Lazily pull thresholds from app_config once (config may load late)."""
        if self._configured:
            return
        try:
            from core.config import app_config
            self.max_failures = app_config.get_int("auth.login_max_failures", self.max_failures)
            self.window_s = app_config.get_int("auth.login_window_seconds", self.window_s)
            self.lockout_s = app_config.get_int("auth.login_lockout_seconds", self.lockout_s)
        except Exception:
            pass
        self._configured = True


    @staticmethod
    def _redis():
        try:
            from core.redis_client import get_redis
            return get_redis()
        except Exception:
            return None

    @staticmethod
    def _scope(key: str) -> str:
        """Prefix the rate-limit key with the ACTIVE tenant so the same username
        across two tenants can't share one lockout counter (cross-tenant
        availability isolation). Falls back to the bare key off-request."""
        try:
            from core.tenant_context import current_tenant
            tid = getattr(current_tenant(), "tenant_id", "default") or "default"
            return f"{tid}|{key}"
        except Exception:
            return key

    def _check_redis(self, r, key: str) -> Tuple[bool, int]:
        ttl = r.ttl(f"ag:lockout:{key}")
        if ttl and ttl > 0:
            return False, int(ttl)
        return True, 0

    def _record_failure_redis(self, r, key: str) -> None:
        fk = f"ag:fails:{key}"
        pipe = r.pipeline()
        pipe.incr(fk)
        pipe.expire(fk, self.window_s)
        count = pipe.execute()[0]
        if int(count) >= self.max_failures:
            r.set(f"ag:lockout:{key}", "1", ex=self.lockout_s)
            r.delete(fk)

    def _record_success_redis(self, r, key: str) -> None:
        r.delete(f"ag:fails:{key}", f"ag:lockout:{key}")


    def check(self, key: str) -> Tuple[bool, int]:
        """Return (allowed, retry_after_s). Call BEFORE verifying credentials."""
        if not key:
            return True, 0
        key = self._scope(key)
        self._ensure_configured()
        r = self._redis()
        if r is not None:
            try:
                return self._check_redis(r, key)
            except Exception:
                pass
        now = time.time()
        with self._lock:
            locked = self._locked_until.get(key)
            if locked and now < locked:
                return False, int(locked - now)
            if locked and now >= locked:
                self._locked_until.pop(key, None)
                self._failures.pop(key, None)
            return True, 0

    def record_failure(self, key: str) -> None:
        """Record a failed attempt; lock the key if over threshold."""
        if not key:
            return
        key = self._scope(key)
        self._ensure_configured()
        r = self._redis()
        if r is not None:
            try:
                self._record_failure_redis(r, key)
                return
            except Exception:
                pass
        now = time.time()
        with self._lock:
            dq = self._failures.setdefault(key, deque())
            dq.append(now)
            cutoff = now - self.window_s
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.max_failures:
                self._locked_until[key] = now + self.lockout_s

    def record_success(self, key: str) -> None:
        """Clear all failure state for a key on successful login."""
        if not key:
            return
        key = self._scope(key)
        r = self._redis()
        if r is not None:
            try:
                self._record_success_redis(r, key)
                return
            except Exception:
                pass
        with self._lock:
            self._failures.pop(key, None)
            self._locked_until.pop(key, None)


login_limiter = LoginRateLimiter()


_gen_lock = threading.Lock()
_gen_windows: Dict[str, Deque[float]] = {}


def check_rate(bucket: str, limit: int, window_s: int = 60) -> Tuple[bool, int]:
    """Allow at most `limit` events per `window_s` for `bucket`. Tenant-scoped.

    `limit <= 0` disables the cap (always allowed). Prefers Redis so the cap
    holds across workers; degrades to a per-worker sliding window, then to
    allow-on-error.
    """
    if not bucket or limit is None or int(limit) <= 0:
        return True, 0
    limit = int(limit)
    key = LoginRateLimiter._scope(bucket)
    r = LoginRateLimiter._redis()
    if r is not None:
        try:
            rk = f"ag:rl:{key}"
            pipe = r.pipeline()
            pipe.incr(rk)
            pipe.expire(rk, window_s)
            count = int(pipe.execute()[0])
            if count > limit:
                ttl = r.ttl(rk)
                return False, int(ttl if ttl and ttl > 0 else window_s)
            return True, 0
        except Exception:
            pass
    now = time.time()
    with _gen_lock:
        dq = _gen_windows.setdefault(key, deque())
        while dq and now - dq[0] > window_s:
            dq.popleft()
        if len(dq) >= limit:
            return False, int(window_s - (now - dq[0])) + 1
        dq.append(now)
        return True, 0
