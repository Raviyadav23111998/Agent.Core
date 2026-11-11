"""
Cron job scheduler — executes due jobs via the agent loop.

Migrated from OS file locks to PostgreSQL advisory locks.
The gateway calls tick() every 60 seconds.
"""

import logging
from typing import Optional, Callable, Awaitable

from cron.jobs import get_due_jobs, mark_job_run, save_job_output, _get_db

logger = logging.getLogger(__name__)

SILENT_MARKER = "[SILENT]"

CRON_TICK_LOCK_ID = 99001


async def tick(
    run_agent: Callable[..., Awaitable[str]],
    deliver_fn: Optional[Callable] = None,
):
    """Check for due jobs and run them.

    Uses PostgreSQL advisory lock to prevent concurrent ticks across processes.

    Args:
        run_agent: Async function ``(prompt, app_id=...)`` returning the agent's
                   response. The job's app_id is threaded through so the run
                   authenticates with its owning app's connections.
        deliver_fn: Optional async function(job, content) to deliver results externally.
    """
    db = _get_db()

    with db.connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (CRON_TICK_LOCK_ID,))
            locked = cur.fetchone()[0]
            conn.commit()

            if not locked:
                logger.debug("Cron tick: advisory lock held by another process, skipping")
                return

            try:
                due_jobs = get_due_jobs()
                if not due_jobs:
                    return

                logger.info("Cron tick: %d job(s) due", len(due_jobs))

                for job in due_jobs:
                    await _run_job(job, run_agent, deliver_fn)
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s)", (CRON_TICK_LOCK_ID,))
                cur.fetchone()
                conn.commit()
        finally:
            cur.close()


async def _run_job(
    job: dict,
    run_agent: Callable[..., Awaitable[str]],
    deliver_fn: Optional[Callable] = None,
):
    """Run a single cron job via the agent and deliver results.
    
    Circuit breaker: if the job has failed >= max_failures times
    consecutively, skip the LLM call entirely and just reschedule.
    This prevents burning tokens on misconfigured integrations.
    """
    job_id = job["id"]
    job_name = job.get("name", job_id)
    prompt = job.get("prompt", "")

    if not prompt:
        mark_job_run(job_id, False, error="Empty prompt")
        return

    db = _get_db()
    db_id = job.get("db_id")
    if db_id:
        row = db.fetch_one(
            "SELECT consecutive_failures, COALESCE(max_failures, 3) AS max_failures, last_error "
            "FROM ag_agents.cron_jobs WHERE job_id = %s", (db_id,)
        )
        if row:
            failures = row.get("consecutive_failures", 0) or 0
            max_fail = row.get("max_failures", 3) or 3
            if failures >= max_fail:
                last_err = (row.get("last_error") or "unknown")[:80]
                logger.warning(
                    "Circuit breaker OPEN for '%s' — %d consecutive failures "
                    "(last: %s). Skipping LLM call to save tokens. "
                    "Fix the root cause and reset via: UPDATE ag_agents.cron_jobs "
                    "SET consecutive_failures = 0 WHERE job_id = %s",
                    job_name, failures, last_err, db_id
                )
                mark_job_run(job_id, False, error=f"Circuit breaker: {failures} consecutive failures — {last_err}")
                return

    logger.info("Running cron job '%s' (%s)", job_name, job_id)

    try:
        # Thread the job's owning app_id so the agent authenticates with that
        # app's connections (app-scoped Jira/M365/… credentials) instead of
        # falling back to org-only resolution.
        result = await run_agent(prompt, app_id=job.get("app_id"),
                                 persona_id=job.get("persona_id"))

        if not result or not result.strip():
            result = "(No output)"

        save_job_output(job_id, result)

        should_deliver = not result.strip().startswith(SILENT_MARKER)
        deliver_target = job.get("deliver", "local")

        if should_deliver and deliver_target != "local" and deliver_fn:
            try:
                await deliver_fn(job, result)
            except Exception as e:
                logger.error("Delivery failed for job '%s': %s", job_name, e)
                mark_job_run(job_id, True, error=f"Delivery failed: {e}")
                return

        mark_job_run(job_id, True)
        logger.info("Cron job '%s' completed successfully", job_name)

    except Exception as e:
        error_msg = str(e)
        logger.error("Cron job '%s' failed: %s", job_name, error_msg)
        save_job_output(job_id, f"ERROR: {error_msg}")
        mark_job_run(job_id, False, error=error_msg)

