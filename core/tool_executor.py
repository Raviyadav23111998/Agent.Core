"""
Tool Executor — Abstraction for local vs remote tool execution.
═══════════════════════════════════════════════════════════════════

Architecture:
    ToolExecutor (ABC)
    ├── LocalToolExecutor   — calls tool_registry.execute() directly (Web Chat, CLI, Cloud Server mode)
    └── RemoteToolExecutor  — routes LOCAL tools to VS Code via WebSocket, SERVER tools locally

The AgentLoop delegates ALL tool execution through a ToolExecutor instance.
This decouples the agent brain from WHERE tools actually run.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger("agent.tool_executor")


class _ReadResultCache:
    """Short-TTL cache for pure read-tool results.

    Saves tokens + latency when the model re-reads the same file/query within
    a turn. Invalidated wholesale whenever a mutating (write/irreversible)
    tool runs, so a cached read can never go stale behind a write.
    """

    def __init__(self, ttl_s: float = 60.0, max_entries: int = 128):
        self._ttl = ttl_s
        self._max = max_entries
        self._store: Dict[str, tuple] = {}

    @staticmethod
    def _key(tool_name: str, args: Dict[str, Any]) -> str:
        try:
            canon = json.dumps(args, sort_keys=True, default=str)
        except Exception:
            canon = str(args)
        return f"{tool_name}::{canon}"

    def get(self, tool_name: str, args: Dict[str, Any]):
        key = self._key(tool_name, args)
        hit = self._store.get(key)
        if not hit:
            return None
        expires_at, result = hit
        if time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return result

    def put(self, tool_name: str, args: Dict[str, Any], result: str):
        if len(self._store) >= self._max:
            self._store.pop(next(iter(self._store)), None)
        self._store[self._key(tool_name, args)] = (time.time() + self._ttl, result)

    def invalidate_all(self):
        self._store.clear()


LOCAL_TOOLS = frozenset({
    "view_file",
    "write_to_file",
    "list_dir",
    "grep_search",
    "replace_file_content",
    "multi_replace_file_content",
    "run_command",
    "ask_question",
    "page_observe",
    "page_act",
    "page_navigate",
    "page_wait",
    "page_read",
})


class _ToolFailure(Exception):
    """A tool run that failed but has a user-facing message (e.g. a timeout).

    Raised by the execution core so it propagates PAST the read-cache (a
    failure is never memoised) and is converted back to a plain string at the
    execute() boundary — the AgentLoop still sees a string, never an exception.
    """
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ToolExecutor(ABC):
    """Abstract base for tool execution strategies."""

    DUPLICATE_GUARD_S = 60.0

    @abstractmethod
    async def execute(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a tool and return its result as a string."""
        ...

    @staticmethod
    def _call_signature(tool_name: str, tool_args: Dict[str, Any]) -> str:
        scope = ""
        try:
            from core.tenant_context import current_tenant
            from core.user_context import current_user_context
            tid = getattr(current_tenant(), "tenant_id", "default")
            uctx = current_user_context.get()
            uid = getattr(uctx, "user_id", "") if uctx else ""
            scope = f"{tid}:{uid}:"
        except Exception:
            scope = ""
        try:
            return f"{scope}{tool_name}::{json.dumps(tool_args, sort_keys=True, default=str)}"
        except Exception:
            return f"{scope}{tool_name}::{tool_args}"

    @staticmethod
    def _role_authz_block(tool_name: str) -> Optional[str]:
        """Per-tool role authorization (I.4). Returns a denial string or None.

        Reads the active user from the ContextVar the agent loop binds. System
        / automation contexts (cron, CLI, headless) bypass — they have no human
        role and are already trusted. Default-open: returns None unless
        `security.tool_min_role` is configured.
        """
        try:
            from core.user_context import current_user_context
            uctx = current_user_context.get()
            if uctx is None or getattr(uctx, "is_system", False):
                return None
            from core.tool_authz import check_tool_role
            allowed, reason = check_tool_role(tool_name, getattr(uctx, "role", ""))
            if not allowed:
                logger.warning("Role authz blocked %s: %s", tool_name, reason)
                return (f"[ACCESS DENIED] Tool '{tool_name}' {reason}. "
                        f"The action was NOT executed.")
        except Exception:
            return None
        return None

    @staticmethod
    def _dup_redis_key(sig: str) -> str:
        import hashlib
        return "ag:dup:" + hashlib.sha256(sig.encode()).hexdigest()[:32]

    def _record_irreversible(self, tool_name: str, tool_args: Dict[str, Any]):
        """Arm the double-send guard for this exact call (local + Redis).

        With Redis configured the guard holds across ALL workers (K.2) —
        a retry hitting a different process is still blocked. Local dict
        is always written as the fallback.
        """
        sig = self._call_signature(tool_name, tool_args)
        self._recent_irreversible[sig] = time.time()
        try:
            from core.redis_client import get_redis
            r = get_redis()
            if r is not None:
                r.set(self._dup_redis_key(sig), "1",
                      ex=int(self.DUPLICATE_GUARD_S))
        except Exception:
            pass

    def _check_duplicate_irreversible(self, tool_name: str,
                                      tool_args: Dict[str, Any]) -> Optional[str]:
        """Block byte-identical irreversible calls repeated within the window.

        Guards against the classic accident: a send times out client-side,
        the model retries, and the email/message goes out twice. A repeat
        AFTER the window (a genuine new request) is allowed. Requires the
        subclass to maintain a `_recent_irreversible` dict; checks Redis
        first when configured so the guard is cross-worker.
        """
        sig = self._call_signature(tool_name, tool_args)
        now = time.time()

        try:
            from core.redis_client import get_redis
            r = get_redis()
            if r is not None:
                ttl = r.ttl(self._dup_redis_key(sig))
                if ttl and ttl > 0:
                    logger.warning("Duplicate irreversible call blocked (redis): %s", tool_name)
                    return (
                        f"[DUPLICATE BLOCKED] An identical {tool_name} call was "
                        f"already executed recently, so this repeat was NOT executed "
                        f"(its earlier result still stands). Assume the first call "
                        f"succeeded. If you truly need to run it again, wait "
                        f"{int(ttl)}s or change the arguments."
                    )
        except Exception:
            pass
        expired = [s for s, ts in self._recent_irreversible.items()
                   if now - ts > self.DUPLICATE_GUARD_S]
        for s in expired:
            self._recent_irreversible.pop(s, None)
        ts = self._recent_irreversible.get(sig)
        if ts is not None:
            age = int(now - ts)
            logger.warning("Duplicate irreversible call blocked: %s (%.0fs ago)", tool_name, age)
            return (
                f"[DUPLICATE BLOCKED] An identical {tool_name} call was already "
                f"executed {age}s ago, so this repeat was NOT executed (its earlier "
                f"result still stands). Assume the first call succeeded. If you truly "
                f"need to run it again, wait "
                f"{int(self.DUPLICATE_GUARD_S - age)}s or change the arguments."
            )
        return None

    def _policy_for(self, tool_name: str):
        """Resolve the ToolPolicy for a tool from the registry (None if unknown)."""
        getter = getattr(getattr(self, "_registry", None), "get_policy", None)
        if callable(getter):
            try:
                return getter(tool_name)
            except Exception:
                return None
        return None

    # ── Read-result cache (shared by EVERY executor) ────────────────────────
    # A tool's declared policy (cacheable read / mutating write) must take
    # effect wherever the tool runs — locally OR over a remote bridge. Keeping
    # the cache only in LocalToolExecutor silently exempted the web-copilot /
    # VS Code page_* tools (they run through RemoteToolExecutor), so a repeated
    # "read the same screen" re-ran a full round-trip. This lives in the base
    # so both executors get it for free.
    def _read_cache(self):
        # INVARIANT: one executor instance per security scope (chat = per
        # session; cron = per run; copilot = per WS connection). The cache key
        # is (tool_name, args) WITHOUT tenant/user — correct only because the
        # cache lives on the per-scope instance. Never share an executor across
        # tenants/users, or add scope to the key first.
        c = getattr(self, "_cache", None)
        if c is None:
            c = _ReadResultCache()
            self._cache = c
        return c

    async def _read_through_cache(self, tool_name: str, tool_args: Dict[str, Any],
                                  policy, run):
        """Memoise cacheable reads; invalidate the read-cache on any write.

        `run` is an async callable performing the real execution. It RETURNS on
        success and RAISES on failure — so a failure never reaches ``put`` and
        is never memoised (no result-string sniffing). A write invalidates in a
        ``finally``, so even a write that errors/times out (and may have
        partially mutated state) can never leave a stale read cached.
        """
        cache = self._read_cache()
        cacheable = policy is not None and getattr(policy, "cacheable", False)
        invalidates = (policy is not None
                       and getattr(policy, "side_effect", None) != "read")
        if cacheable:
            hit = cache.get(tool_name, tool_args)
            if hit is not None:
                try:
                    from core.telemetry import telemetry
                    telemetry.incr("tool_cache_hits_total", tool=tool_name)
                except Exception:
                    pass
                return hit
        try:
            result = await run()
        finally:
            if invalidates:
                cache.invalidate_all()
        if cacheable:
            cache.put(tool_name, tool_args, result)
        return result

    def _audit_action(self, tool_name: str, policy, tool_args: Dict[str, Any],
                      approved, status: str):
        """Record a side-effecting action to the audit sink (best-effort)."""
        audit = getattr(self, "_audit", None)
        if audit is None or policy is None or getattr(policy, "side_effect", None) == "read":
            return
        try:
            audit(tool_name, policy.side_effect, str(tool_args)[:500], approved, status)
        except Exception:
            pass

    async def _pre_execution_gate(self, tool_name: str, tool_args: Dict[str, Any],
                                  policy) -> Optional[str]:
        """The SINGLE pre-execution security gate, shared by every executor.

        Returns a user-facing block string to short-circuit, or None to proceed.
        This is the one place tool-call policy is enforced — the agent loop no
        longer gates separately. Order (first hit wins, fail-closed):

          1. Guardrail engine — execution policy + embed-action danger. A danger
             match that allows human override becomes an approval gate here.
          2. Per-tool role authorization (I.4).
          3. Irreversible-call protection — double-send dedup + HITL approval.
        """
        guard = getattr(self, "_guardrail", None)
        if guard is not None:
            from core.guard_pipeline import GuardContext
            decision = guard.evaluate(GuardContext(
                tool_name=tool_name, args=tool_args, policy=policy))
            if decision.needs_approval:
                approved = await self._request_approval(tool_name, tool_args)
                if approved is False:
                    return f"[APPROVAL DENIED] {decision.reason}. The action was NOT performed."
                if approved is None:
                    return f"[GUARDRAIL BLOCKED] {decision.reason}"
            elif not decision.allow:
                return f"[GUARDRAIL BLOCKED] {decision.reason}"

        denied = self._role_authz_block(tool_name)
        if denied:
            return denied

        if policy is not None and getattr(policy, "requires_approval", False):
            if getattr(policy, "dedup_guard", True):
                blocked = self._check_duplicate_irreversible(tool_name, tool_args)
                if blocked:
                    self._audit_action(tool_name, policy, tool_args, None, "blocked")
                    return blocked
            approved = await self._request_approval(tool_name, tool_args)
            if approved is False:
                self._audit_action(tool_name, policy, tool_args, False, "denied")
                return "[APPROVAL DENIED] The action was NOT executed."
        return None

    async def _request_approval(self, tool_name: str, tool_args: Dict[str, Any]):
        """Ask the HITL approval manager. Returns True (approved), False (denied),
        or None when no approval channel is enabled (caller decides the default)."""
        approvals = getattr(self, "_approvals", None)
        if approvals is None or not getattr(approvals, "enabled", False):
            return None
        try:
            approved, _reason = await approvals.request(tool_name, tool_args)
            return bool(approved)
        except Exception as e:
            logger.error("Approval request failed for %s: %s — denying", tool_name, e)
            return False


class LocalToolExecutor(ToolExecutor):
    """Execute all tools locally via ToolRegistry.
    
    This is the default executor used by:
    - Web Chat (all tools on VPS)
    - CLI mode (all tools on the machine running CLI)
    - VS Code "Cloud Server" mode (all tools on VPS)
    """

    def __init__(self, tool_registry: Any, guardrail_engine: Any = None,
                 approval_manager: Any = None, audit_sink: Any = None):
        self._registry = tool_registry
        self._guardrail = guardrail_engine
        self._approvals = approval_manager
        self._audit = audit_sink
        self._cache = _ReadResultCache()
        self._recent_irreversible: Dict[str, float] = {}

    async def execute(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a tool. Cache/invalidation is the shared base concern; this
        method owns the retry/timeout/telemetry/audit core. A timeout surfaces
        as a string to the AgentLoop but as a raise past the cache, so it is
        never memoised."""
        policy = self._policy_for(tool_name)

        block = await self._pre_execution_gate(tool_name, tool_args, policy)
        if block is not None:
            return block

        try:
            return await self._read_through_cache(
                tool_name, tool_args, policy,
                lambda: self._execute_core(tool_name, tool_args, policy))
        except _ToolFailure as e:
            return e.message

    async def _execute_core(self, tool_name: str, tool_args: Dict[str, Any],
                            policy) -> str:
        """The real execution: retries, timeout, telemetry, audit. Returns the
        tool result on success; RAISES (_ToolFailure on timeout, the original
        exception on a hard error) on failure so nothing failed is cached."""
        try:
            from core.telemetry import telemetry
        except Exception:
            telemetry = None

        timeout_s = getattr(policy, "timeout_s", 0) or 0
        max_attempts = 1 + (getattr(policy, "retries", 0)
                            if policy is not None and policy.idempotent else 0)

        _is_async = False
        try:
            _is_async = bool(self._registry.is_async(tool_name))
        except Exception:
            pass

        _start = time.perf_counter()
        result = None
        for attempt in range(max_attempts):
            try:
                if _is_async:
                    # Async tools (sub-agents, future I/O tools) run ON the
                    # loop — no worker thread, no nested event loop; a batch
                    # of them is genuinely concurrent.
                    coro = self._registry.execute_async(tool_name, tool_args)
                else:
                    coro = asyncio.to_thread(self._registry.execute, tool_name, tool_args)
                if timeout_s > 0:
                    result = await asyncio.wait_for(coro, timeout=timeout_s)
                else:
                    result = await coro
                result = str(result) if result is not None else "Done."
                break
            except asyncio.TimeoutError:
                if telemetry:
                    telemetry.incr("tool_calls_total", tool=tool_name, status="timeout")
                if attempt + 1 >= max_attempts:
                    if policy is not None and policy.requires_approval:
                        if getattr(policy, "dedup_guard", True):
                            self._record_irreversible(tool_name, tool_args)
                        self._audit_action(tool_name, policy, tool_args, None, "timeout")
                    raise _ToolFailure(
                        f"Error: tool '{tool_name}' timed out after "
                        f"{int(timeout_s)}s (policy limit). It may still be "
                        f"running; do not assume it completed.")
            except Exception:
                if attempt + 1 >= max_attempts:
                    if telemetry:
                        telemetry.incr("tool_calls_total", tool=tool_name, status="error")
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))

        if telemetry:
            status = "error" if result.startswith("Error executing") else "ok"
            telemetry.observe("tool_duration_seconds", time.perf_counter() - _start, tool=tool_name)
            telemetry.incr("tool_calls_total", tool=tool_name, status=status)

        if policy is not None and policy.requires_approval:
            if getattr(policy, "dedup_guard", True):
                self._record_irreversible(tool_name, tool_args)
            approved = True if (self._approvals and getattr(self._approvals, "enabled", False)) else None
            self._audit_action(tool_name, policy, tool_args, approved, "executed")
        elif policy is not None and policy.side_effect == "write":
            self._audit_action(tool_name, policy, tool_args, None, "executed")

        return result


class RemoteToolExecutor(ToolExecutor):
    """Routes LOCAL tools to the VS Code extension via WebSocket.
    Server tools still execute locally on the VPS via ToolRegistry.
    
    Used by VS Code "Local Server" mode only.
    
    Protocol:
        1. Agent decides to call view_file("/Users/.../app.py")
        2. RemoteToolExecutor checks: view_file ∈ LOCAL_TOOLS → yes
        3. Sends via WebSocket: {"type": "tool_request", "requestId": "...", "toolName": "view_file", "toolArgs": {...}}
        4. Waits for VS Code to respond: {"type": "tool_result", "requestId": "...", "result": "..."}
        5. Returns the result to the AgentLoop
    """

    TOOL_REQUEST_TIMEOUT = 120

    def __init__(self, websocket: Any, tool_registry: Any,
                 result_queue_factory=None, guardrail_engine: Any = None,
                 approval_manager: Any = None,
                 allowed_local_tools: Optional[set] = None):
        """
        Args:
            websocket: The FastAPI WebSocket connection to the VS Code client.
            tool_registry: The server-side ToolRegistry for server tools.
            result_queue_factory: Optional factory for creating result queues.
            guardrail_engine: Optional GuardrailEngine for execution policies.
            approval_manager: Optional ApprovalManager for HITL gates.
            allowed_local_tools: Hard ceiling for client-executed (LOCAL) tools.
                None = allow all LOCAL_TOOLS (VS Code / trusted surfaces). The
                embed runtime passes the mode's ceiling so a chat-mode bot can
                never drive the page even if the model emits a page_* call the
                schema never offered it (defense-in-depth: the LLM schema is
                built from the sandboxed registry, but execution was previously
                routed off the FULL registry, bypassing the sandbox).
        """
        self._ws = websocket
        self._registry = tool_registry
        self._pending: Dict[str, asyncio.Future] = {}
        self._guardrail = guardrail_engine
        self._approvals = approval_manager
        self._allowed_local_tools = allowed_local_tools
        # Embed sessions: the exact tool names the session offered the model.
        # None = no limit (trusted surfaces). See restrict_to().
        self._allowed_tools: Optional[frozenset] = None
        self._recent_irreversible: Dict[str, float] = {}

    def restrict_to(self, registry: Any) -> None:
        """Run only tools in `registry` (the session's sandboxed registry), and
        run server tools FROM it. Without this, server-side tools resolved off
        the full registry passed at construction, so a model coaxed into naming
        a tool it was never offered would still execute it. Fails closed: an
        unusable registry allows nothing."""
        names = getattr(registry, "_tools", None)
        self._allowed_tools = frozenset(names) if isinstance(names, dict) else frozenset()
        if isinstance(names, dict):
            self._registry = registry

    async def execute(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Route tool execution: LOCAL tools → WebSocket, SERVER tools → local."""
        policy = self._policy_for(tool_name)

        block = await self._pre_execution_gate(tool_name, tool_args, policy)
        if block is not None:
            return block

        try:
            # Cacheable reads (page_observe/page_read/view_file …) are memoised
            # and served without a round-trip; writes invalidate the cache.
            return await self._read_through_cache(
                tool_name, tool_args, policy,
                lambda: self._dispatch(tool_name, tool_args))
        finally:
            if (policy is not None and policy.requires_approval
                    and getattr(policy, "dedup_guard", True)):
                self._record_irreversible(tool_name, tool_args)

    async def _dispatch(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        if self._allowed_tools is not None and tool_name not in self._allowed_tools:
            logger.warning("RemoteToolExecutor: blocked unoffered tool %r", tool_name)
            return f"[GUARDRAIL BLOCKED] '{tool_name}' is not available here."
        if tool_name in LOCAL_TOOLS:
            if (self._allowed_local_tools is not None
                    and tool_name not in self._allowed_local_tools):
                return (f"[GUARDRAIL BLOCKED] '{tool_name}' is not available in "
                        f"this mode. Use page_observe/page_act to read and act on "
                        f"the screen.")
            return await self._execute_remote(tool_name, tool_args)
        else:
            try:
                _is_async_fn = getattr(self._registry, "is_async", None)
                if callable(_is_async_fn) and _is_async_fn(tool_name):
                    result = await self._registry.execute_async(tool_name, tool_args)
                else:
                    result = await asyncio.to_thread(
                        self._registry.execute, tool_name, tool_args
                    )
                return str(result) if result is not None else "Done."
            except Exception:
                raise

    async def _execute_remote(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Send tool request to VS Code extension and wait for result."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future

        try:
            await self._ws.send_json({
                "type": "tool_request",
                "requestId": request_id,
                "toolName": tool_name,
                "toolArgs": tool_args,
            })
            
            logger.info("RemoteToolExecutor: Sent tool_request %s → %s(%s)",
                        request_id, tool_name, str(tool_args)[:100])

            timeout = 15 if tool_name.startswith("page_") else self.TOOL_REQUEST_TIMEOUT
            result = await asyncio.wait_for(future, timeout=timeout)
            
            logger.info("RemoteToolExecutor: Got tool_result %s (%d chars)",
                        request_id, len(result))
            
            return result

        except asyncio.TimeoutError:
            logger.error("RemoteToolExecutor: Timeout waiting for tool_result %s (%s) after %ds",
                         request_id, tool_name, timeout)
            return f"Error: VS Code extension did not respond within {timeout}s for {tool_name}"
        except Exception as e:
            logger.error("RemoteToolExecutor: Error in remote execution %s: %s",
                         request_id, e)
            return f"Error: Remote tool execution failed for {tool_name}: {e}"
        finally:
            self._pending.pop(request_id, None)

    def resolve_result(self, request_id: str, result: str):
        """Called when a tool_result message arrives from VS Code.
        
        This resolves the pending Future so _execute_remote() can continue.
        Must be called from the WebSocket message handler.
        """
        future = self._pending.get(request_id)
        if future and not future.done():
            future.set_result(result)
            logger.debug("RemoteToolExecutor: Resolved %s", request_id)
        else:
            logger.warning("RemoteToolExecutor: No pending request for %s", request_id)

    def resolve_error(self, request_id: str, error: str):
        """Called when VS Code reports an error executing a tool."""
        future = self._pending.get(request_id)
        if future and not future.done():
            future.set_result(f"Error (client-side): {error}")
        else:
            logger.warning("RemoteToolExecutor: No pending request for error %s", request_id)

    def cancel_all(self):
        """Cancel all pending tool requests (e.g., on WebSocket disconnect)."""
        for req_id, future in self._pending.items():
            if not future.done():
                future.set_result("Error: WebSocket connection lost")
        self._pending.clear()
