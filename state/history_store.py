"""Conversation-history persistence — the resume mechanism.
═══════════════════════════════════════════════════════════

Claude Code treats resume as an engine feature: a process restart or
session-pool eviction must not amnesia a conversation. This store persists
the loop's exact message list (including `_thinking` blocks and `_is_error`
flags needed for faithful replay) and rehydrates it on session recreation.

Storage reuses `ag_system.checkpoints` (jsonb) with a sentinel iteration of
-1 and one row per session — no schema migration required. Everything is
fail-soft: a missing table or DB just disables resume with a debug log.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

logger = logging.getLogger("agent.history_store")

SENTINEL_ITERATION = -1
_PAYLOAD_TYPE = "conversation_history"
MAX_MESSAGES = 2000


class HistoryStore:
    def __init__(self, db: Any, session_id: str, user_id: str = None):
        self._db = db if not isinstance(db, str) else None
        self._session_id = session_id or "default"
        self._user_id = user_id

    @property
    def available(self) -> bool:
        return self._db is not None and self._session_id != "default"

    def save(self, messages: List[dict]) -> bool:
        """Persist the full message list (replaces the previous snapshot)."""
        if not self.available:
            return False
        try:
            payload = json.dumps(
                {"type": _PAYLOAD_TYPE, "messages": messages[-MAX_MESSAGES:]},
                ensure_ascii=False, default=str)
            self._db.execute(
                "DELETE FROM ag_system.checkpoints "
                "WHERE session_id = %s AND iteration = %s",
                (self._session_id, SENTINEL_ITERATION))
            self._db.execute(
                "INSERT INTO ag_system.checkpoints "
                "(session_id, user_id, iteration, checkpoint_data) "
                "VALUES (%s, %s, %s, %s::jsonb)",
                (self._session_id, self._user_id, SENTINEL_ITERATION, payload))
            return True
        except Exception as e:
            logger.debug("history save skipped (%s): %s", self._session_id, e)
            return False

    def load(self) -> Optional[List[dict]]:
        """Return the persisted message list, or None."""
        if not self.available:
            return None
        try:
            row = self._db.fetch_one(
                "SELECT checkpoint_data FROM ag_system.checkpoints "
                "WHERE session_id = %s AND iteration = %s",
                (self._session_id, SENTINEL_ITERATION))
            if not row:
                return None
            data = row.get("checkpoint_data")
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, dict) or data.get("type") != _PAYLOAD_TYPE:
                return None
            messages = data.get("messages")
            return messages if isinstance(messages, list) else None
        except Exception as e:
            logger.debug("history load skipped (%s): %s", self._session_id, e)
            return None
