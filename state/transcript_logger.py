"""
Conversation transcript logger — PostgreSQL-backed (ag_system.transcript_logs).

Migrated from .ag_data/users/<id>/logs/transcript_*.jsonl to PostgreSQL.
Every agent action is logged as a row for searchable audit trail.
"""

import json
import uuid
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class TranscriptLogger:
    """Manages conversation history and raw LLM traces.
    
    Writes every action to ag_system.transcript_logs so the agent
    can search its own memory and provide an audit trail.
    
    PostgreSQL-backed via TenantDB.
    """

    def __init__(self, db, user_id: str = None, **kwargs):
        """
        Args:
            db: TenantDB instance (or legacy storage_dir string — ignored)
            user_id: User identifier for scoping logs
        """
        if isinstance(db, str):
            self._db = kwargs.get('tenant_db')
            self._user_id = user_id
        else:
            self._db = db
            self._user_id = user_id

        self.conversation_id = str(uuid.uuid4())
        self.step_index = 0

    def log_step(self, source: str, step_type: str, content: str, tool_calls: list = None):
        """Append a single step to the transcript.
        
        Args:
            source: USER, MODEL, SYSTEM
            step_type: USER_INPUT, LLM_RESPONSE, VIEW_FILE, etc.
            content: The text content of this step
            tool_calls: Optional list of tool call dicts
        """
        if not self._db:
            return self.step_index

        try:
            self._db.execute("""
                INSERT INTO ag_system.transcript_logs
                    (session_id, user_id, step_index, source, step_type, content, tool_calls)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            """, (
                self.conversation_id,
                self._user_id,
                self.step_index,
                source,
                step_type,
                content[:50000] if content else None,
                json.dumps(tool_calls) if tool_calls else None,
            ))
        except Exception as e:
            logger.warning("transcript_logger.write_failed: %s", e)

        self.step_index += 1
        return self.step_index

    def get_recent_history(self, num_steps: int = 10) -> List[Dict[str, Any]]:
        """Get recent transcript steps for this session."""
        if not self._db:
            return []

        rows = self._db.fetch_all("""
            SELECT step_index, source, step_type AS type, content, tool_calls, created_at
            FROM ag_system.transcript_logs
            WHERE session_id = %s
            ORDER BY step_index DESC
            LIMIT %s
        """, (self.conversation_id, num_steps))

        return list(reversed(rows))
