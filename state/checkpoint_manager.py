"""
Conversation checkpointing — PostgreSQL-backed (ag_system.checkpoints).

Migrated from .ag_data/users/<id>/checkpoints/checkpoint_*.json to PostgreSQL.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Conversation checkpointing for long multi-step tasks.
    
    Architecture:
    - Every CHECKPOINT_INTERVAL tool iterations, generates a checkpoint summary
    - Checkpoint includes: accomplishments, decisions, findings, unresolved items
    - When pruning, checkpoint summary injected after task anchor
    - Prevents loss of critical context during very long conversations
    
    PostgreSQL-backed via TenantDB (ag_system.checkpoints).
    """
    
    CHECKPOINT_INTERVAL = 15
    
    def __init__(self, db_or_dir, session_id: str = None, user_id: str = None):
        """
        Args:
            db_or_dir: TenantDB instance or legacy storage_dir string
            session_id: Conversation/session ID for scoping checkpoints
            user_id: User identifier
        """
        if isinstance(db_or_dir, str):
            self._db = None
        else:
            self._db = db_or_dir

        self._session_id = session_id or "default"
        self._user_id = user_id

        self.checkpoints: List[dict] = []
        self._iteration_count = 0
        self._files_touched: set = set()
        self._actions_log: List[str] = []
        self._decisions: List[str] = []
        self._accomplishments: List[str] = []
        self._unresolved: List[str] = []
    
    def record_tool_call(self, tool_name: str, result_summary: str):
        """Record a tool call for checkpoint generation."""
        self._iteration_count += 1
        
        if tool_name in ('write_to_file', 'replace_file_content', 'multi_replace_file_content'):
            self._files_touched.add(result_summary.split('/')[-1][:50] if '/' in result_summary else tool_name)
            self._accomplishments.append(f"Modified: {result_summary[:60]}")
        
        if tool_name in ('run_command',):
            self._accomplishments.append(f"Ran: {result_summary[:60]}")
        
        action = f"{tool_name}: {result_summary[:80]}"
        self._actions_log.append(action)
    
    def record_decision(self, decision: str):
        """Record a decision made during the conversation."""
        if decision and decision not in self._decisions:
            self._decisions.append(decision[:120])
    
    def record_unresolved(self, item: str):
        """Record an unresolved issue."""
        if item and item not in self._unresolved:
            self._unresolved.append(item[:120])
    
    def should_checkpoint(self) -> bool:
        """Check if it's time to create a checkpoint."""
        return (self._iteration_count > 0 and 
                self._iteration_count % self.CHECKPOINT_INTERVAL == 0)
    
    def create_checkpoint(self) -> dict:
        """Generate a rich checkpoint summary from accumulated context."""
        checkpoint = {
            "iteration": self._iteration_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_calls_count": self._iteration_count,
            "files_modified": list(self._files_touched),
            "accomplishments": self._accomplishments[-self.CHECKPOINT_INTERVAL:],
            "decisions": list(self._decisions),
            "unresolved": list(self._unresolved),
            "actions": self._actions_log[-self.CHECKPOINT_INTERVAL:],
        }
        
        self.checkpoints.append(checkpoint)
        self._save_checkpoint(checkpoint)
        
        self._accomplishments = self._accomplishments[-3:]
        
        return checkpoint
    
    def get_checkpoint_summary(self) -> str:
        """Get a compact summary of the latest checkpoint for prompt injection."""
        if not self.checkpoints:
            return ""
        
        latest = self.checkpoints[-1]
        files = latest.get("files_modified", [])
        accomplishments = latest.get("accomplishments", [])
        decisions = latest.get("decisions", [])
        unresolved = latest.get("unresolved", [])
        
        summary = f"[Checkpoint at iteration {latest['iteration']}]\n"
        
        if accomplishments:
            summary += "Accomplished:\n"
            for a in accomplishments[-6:]:
                summary += f"  - {a}\n"
        
        if decisions:
            summary += "Decisions:\n"
            for d in decisions[-4:]:
                summary += f"  - {d}\n"
        
        if unresolved:
            summary += "Unresolved:\n"
            for u in unresolved[-3:]:
                summary += f"  - {u}\n"
        
        if files:
            summary += f"Files touched: {', '.join(files)}\n"
        
        return summary
    
    def get_all_checkpoints_summary(self) -> str:
        """Get summary of all checkpoints (for very long conversations)."""
        if not self.checkpoints:
            return "No checkpoints yet."
        
        lines = []
        for cp in self.checkpoints:
            lines.append(
                f"[Checkpoint {cp['iteration']}] "
                f"{cp['tool_calls_count']} tools, "
                f"files: {', '.join(cp.get('files_modified', []))}"
            )
        return "\n".join(lines)
    
    def _save_checkpoint(self, checkpoint: dict):
        """Persist checkpoint to PostgreSQL."""
        if not self._db:
            return
        try:
            self._db.execute("""
                INSERT INTO ag_system.checkpoints
                    (session_id, user_id, iteration, checkpoint_data)
                VALUES (%s, %s, %s, %s::jsonb)
            """, (
                self._session_id,
                self._user_id,
                checkpoint["iteration"],
                json.dumps(checkpoint, ensure_ascii=False),
            ))
        except Exception as e:
            logger.warning("checkpoint_manager.save_failed: %s", e)
    
    def reset(self):
        """Reset for a new conversation."""
        self._iteration_count = 0
        self._files_touched.clear()
        self._actions_log.clear()
        self._decisions.clear()
        self._accomplishments.clear()
        self._unresolved.clear()
        self.checkpoints.clear()
