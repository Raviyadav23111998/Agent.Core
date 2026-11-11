"""
Retention sweeper — bounded growth for high-volume tables (roadmap L).
═══════════════════════════════════════════════════════════════════════

Chat messages, transcript logs, cron outputs, and audit rows grow without
bound. This deletes rows older than a per-table retention window, in small
batches to avoid long locks. Run periodically (the gateway calls it once a
day from the cron ticker).

SAFE BY DEFAULT: every window defaults to 0 = keep forever. Nothing is
deleted until an operator sets a `retention.*_days` config > 0, so deploying
this changes nothing on its own. Audit tables get their own (longer) knob so
compliance retention isn't coupled to chat retention.

Config keys (days; 0 = disabled):
    retention.messages_days
    retention.transcripts_days
    retention.cron_outputs_days
    retention.audit_days          (covers audit_log + action_audit)
"""

from __future__ import annotations

import logging

logger = logging.getLogger("agent.retention")

_TABLES = (
    ("ag_system.messages", "created_at", "retention.messages_days"),
    ("ag_system.transcript_logs", "created_at", "retention.transcripts_days"),
    ("ag_agents.cron_job_outputs", "created_at", "retention.cron_outputs_days"),
    ("ag_system.audit_log", "created_at", "retention.audit_days"),
    ("ag_system.action_audit", "created_at", "retention.audit_days"),
)

_BATCH = 5000
_MAX_BATCHES = 50


class RetentionManager:
    """Deletes rows past their configured retention window, in batches."""

    def __init__(self, db):
        self._db = db

    def _days(self, key: str) -> int:
        try:
            from core.config import app_config
            return app_config.get_int(key, 0)
        except Exception:
            return 0

    def sweep(self) -> dict:
        """Run one retention pass. Returns {table: rows_deleted}."""
        deleted = {}
        if self._db is None:
            return deleted
        for table, ts_col, cfg_key in _TABLES:
            days = self._days(cfg_key)
            if days <= 0:
                continue
            total = self._purge_table(table, ts_col, days)
            if total:
                deleted[table] = total
        if deleted:
            logger.info("retention.sweep deleted=%s", deleted)
        return deleted

    def _purge_table(self, table: str, ts_col: str, days: int) -> int:
        """Batch-delete rows older than `days` from one table."""
        total = 0
        sql = (
            f"DELETE FROM {table} WHERE ctid IN ("
            f"  SELECT ctid FROM {table} "
            f"  WHERE {ts_col} < NOW() - (%s || ' days')::interval "
            f"  LIMIT {_BATCH})"
        )
        for _ in range(_MAX_BATCHES):
            try:
                n = self._db.execute(sql, (days,))
            except Exception as e:
                logger.warning("retention.purge_failed table=%s err=%s", table, e)
                break
            total += n or 0
            if not n or n < _BATCH:
                break
        return total
