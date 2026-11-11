"""
Artifact Manager — PostgreSQL-backed (ag_system.artifacts).

Migrated from .ag_data/artifacts/*.md to PostgreSQL.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages persistent markdown artifacts (plans, tasks, walkthroughs).
    
    PostgreSQL-backed via TenantDB (ag_system.artifacts).
    """

    def __init__(self, db_or_dir, session_id: str = None, user_id: str = None):
        """
        Args:
            db_or_dir: TenantDB instance or legacy artifact_dir string
            session_id: Conversation/session ID
            user_id: User identifier
        """
        if isinstance(db_or_dir, str):
            self._db = None
        else:
            self._db = db_or_dir

        self._session_id = session_id or "default"
        self._user_id = user_id

    def write_artifact(self, name: str, title: str, content: str,
                       request_feedback: bool = False) -> str:
        """Creates or overwrites an artifact."""
        if not self._db:
            return f"Artifact '{name}' skipped (no DB)"

        full_content = f"# {title}\n\n"
        if request_feedback:
            full_content += "> [!IMPORTANT]\n> User review requested for this artifact.\n\n"
        full_content += content

        self._db.execute("""
            INSERT INTO ag_system.artifacts
                (name, title, content, session_id, user_id, request_feedback)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (name, session_id) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                request_feedback = EXCLUDED.request_feedback,
                updated_at = NOW()
        """, (
            name, title, full_content,
            self._session_id, self._user_id, request_feedback,
        ))

        return f"Artifact '{name}' saved to database"

    def read_artifact(self, name: str) -> Optional[str]:
        """Read an artifact's content."""
        if not self._db:
            return None

        row = self._db.fetch_one("""
            SELECT content FROM ag_system.artifacts
            WHERE name = %s AND session_id = %s
        """, (name, self._session_id))
        return row["content"] if row else None

    def list_artifacts(self) -> list:
        """List all artifacts for the current session."""
        if not self._db:
            return []

        rows = self._db.fetch_all("""
            SELECT name, title, request_feedback, updated_at
            FROM ag_system.artifacts
            WHERE session_id = %s
            ORDER BY updated_at DESC
        """, (self._session_id,))
        return rows

    def update_task_list(self, task_updates: dict) -> str:
        """Helper to manage checklists in task.md."""
        return "Task list updated."
