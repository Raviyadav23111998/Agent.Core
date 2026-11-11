"""
Human-in-the-Loop Approval Gates.
═══════════════════════════════════

Pauses the agent before IRREVERSIBLE tool calls (send email, run shell,
delete, push — classified by ToolPolicy.requires_approval) and asks the
human to approve or deny. The agent loop simply awaits the decision; the
SSE/WS event streams carry the request to the UI, and a REST endpoint
(or WS frame) resolves it.

Flow:
    executor → ApprovalManager.request()
        → publish 'approval_request' on the conversation EventBus
        → SSE "approval.request" / WS frame reaches the client
    client → POST /v1/chat/approvals/{id}  (or WS approval_response)
        → resolve_approval(id, approved)
        → the awaiting Future completes → tool runs or is blocked

Modes (config `agent.approval_mode`, persona key `approval_mode`):
    "off"          — never ask (default; preserves existing behavior)
    "irreversible" — ask for tools whose policy.requires_approval is True

Timeout denies by default (fail-closed): an unattended approval must not
block forever NOR silently execute.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("agent.approvals")

_PENDING: Dict[str, Dict[str, Any]] = {}


def resolve_approval(approval_id: str, approved: bool,
                     user_id: Optional[int] = None, reason: str = "") -> bool:
    """Resolve a pending approval (called from REST/WS handlers).

    If user_id is given it must match the requesting conversation's user —
    one user must not be able to approve another user's actions.
    Returns True if a pending approval was resolved.
    """
    entry = _PENDING.get(approval_id)
    if not entry:
        return False
    if user_id is not None and entry.get("user_id") not in (None, user_id):
        logger.warning("approval.user_mismatch: id=%s", approval_id)
        return False
    future: asyncio.Future = entry["future"]
    if not future.done():
        future.get_loop().call_soon_threadsafe(
            lambda: None if future.done() else future.set_result((approved, reason))
        )
    return True


class ApprovalManager:
    """Per-conversation approval broker.

    Publishes approval requests on the conversation's EventBus and awaits
    the human decision with a deny-on-timeout policy.
    """

    DEFAULT_TIMEOUT_S = 300

    def __init__(self, event_bus: Any, user_id: Optional[int] = None,
                 mode: str = "off", timeout_s: float = None):
        self._bus = event_bus
        self._user_id = user_id
        self.mode = (mode or "off").strip().lower()
        self._timeout = timeout_s or self.DEFAULT_TIMEOUT_S

    @property
    def enabled(self) -> bool:
        return self.mode == "irreversible"

    async def request(self, tool_name: str,
                      tool_args: Dict[str, Any]) -> Tuple[bool, str]:
        """Ask the human to approve this tool call. Deny on timeout."""
        approval_id = f"appr_{uuid.uuid4().hex[:12]}"
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        _PENDING[approval_id] = {
            "future": future,
            "user_id": self._user_id,
            "tool": tool_name,
            "created": time.time(),
        }
        try:
            await self._bus.publish("approval_request", {
                "approval_id": approval_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "timeout_s": self._timeout,
            })
            logger.info("approval.requested: id=%s tool=%s", approval_id, tool_name)
            try:
                approved, reason = await asyncio.wait_for(future, timeout=self._timeout)
            except asyncio.TimeoutError:
                logger.warning("approval.timeout: id=%s tool=%s", approval_id, tool_name)
                return False, f"approval timed out after {int(self._timeout)}s (denied)"
            logger.info("approval.resolved: id=%s approved=%s", approval_id, approved)
            return approved, reason or ("approved" if approved else "denied by user")
        finally:
            _PENDING.pop(approval_id, None)
