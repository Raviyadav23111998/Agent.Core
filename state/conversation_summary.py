"""
Conversation Summary Manager — PostgreSQL-backed (ag_system.conversation_summaries).

Migrated from .ag_data/conversation_summaries/*.json to PostgreSQL.
"""

import json
import logging

logger = logging.getLogger(__name__)


class ConversationSummaryManager:
    """Auto-generates and persists conversation summaries for cross-session continuity.
    
    Architecture:
    - At session end, generates a structured summary of what was accomplished
    - Summary stored in ag_system.conversation_summaries
    - At next session start, recent summaries loaded as context
    - This is how the agent "remembers" across conversations
    """

    MAX_SUMMARIES = 50

    def __init__(self, db, user_id: int = None):
        """
        Args:
            db: TenantDB instance
            user_id: If provided, scopes summaries to this user
        """
        self._db = db
        self._user_id = user_id

    def save_summary(self, summary: dict) -> str:
        """Persist a conversation summary to the database."""
        if not summary or not summary.get("objective"):
            return "No meaningful conversation to summarize."

        summary_text = json.dumps(summary, ensure_ascii=False)
        message_count = summary.get("messages_count", 0)

        self._db.execute("""
            INSERT INTO ag_system.conversation_summaries
                (user_id, summary, message_count)
            VALUES (%s, %s, %s)
        """, (self._user_id, summary_text, message_count))

        self._evict_old_summaries()

        return "Conversation summary saved to database."

    def load_recent_summaries(self, count: int = 5) -> str:
        """Load recent conversation summaries for context injection.
        
        Returns a compact string suitable for system prompt injection.
        """
        sql = """
            SELECT summary, created_at FROM ag_system.conversation_summaries
        """
        params = []
        if self._user_id is not None:
            sql += " WHERE user_id = %s"
            params.append(self._user_id)
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(count)

        rows = self._db.fetch_all(sql, tuple(params))
        if not rows:
            return ""

        lines = []
        for row in rows:
            try:
                s = json.loads(row.get("summary", "{}"))
                objective = s.get("objective", "Unknown")[:100]
                outcome = s.get("outcome", "Unknown")[:100]
                files_mod = ", ".join(s.get("files_modified", [])[:5])
                ts = str(row.get("created_at", "?"))[:10]
                lines.append(
                    f"- [{ts}] "
                    f"{objective} → {outcome}"
                    f"{' (files: ' + files_mod + ')' if files_mod else ''}"
                )
            except Exception:
                continue

        return "\n".join(lines)

    def _evict_old_summaries(self):
        """Remove oldest summaries when over capacity."""
        if self._user_id is None:
            return
        count = self._db.fetch_scalar(
            "SELECT COUNT(*) FROM ag_system.conversation_summaries WHERE user_id = %s",
            (self._user_id,)
        ) or 0
        if count > self.MAX_SUMMARIES:
            overflow = count - self.MAX_SUMMARIES
            self._db.execute("""
                DELETE FROM ag_system.conversation_summaries
                WHERE summary_id IN (
                    SELECT summary_id FROM ag_system.conversation_summaries
                    WHERE user_id = %s
                    ORDER BY created_at ASC
                    LIMIT %s
                )
            """, (self._user_id, overflow))
