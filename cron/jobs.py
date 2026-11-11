"""
Cron job storage and management — PostgreSQL-backed.

Migrated from JSON file storage to ag_agents.cron_jobs table.
Schedule parsing logic is unchanged.

Jobs stored in: ag_agents.cron_jobs (PostgreSQL)
"""

import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any

try:
    from croniter import croniter
    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False

logger = logging.getLogger(__name__)


_db = None

def init_db(db):
    """Inject the TenantDB instance. Must be called before any CRUD."""
    global _db
    _db = db
    logger.info("cron.jobs: PostgreSQL backend initialized")

def _get_db():
    if _db is None:
        raise RuntimeError("cron.jobs.init_db() not called. No database connection.")
    return _db

ONESHOT_GRACE_SECONDS = 120


def _now() -> datetime:
    """Current time as timezone-aware datetime."""
    return datetime.now(timezone.utc).astimezone()


def parse_duration(s: str) -> int:
    """Parse duration string into minutes. '30m' → 30, '2h' → 120, '1d' → 1440."""
    s = s.strip().lower()
    match = re.match(r'^(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$', s)
    if not match:
        raise ValueError(f"Invalid duration: '{s}'. Use format like '30m', '2h', or '1d'")
    value = int(match.group(1))
    unit = match.group(2)[0]
    return value * {'m': 1, 'h': 60, 'd': 1440}[unit]


def parse_schedule(schedule: str) -> Dict[str, Any]:
    """Parse schedule string into structured format.
    
    Supports:
        "30m"              → once in 30 minutes
        "every 30m"        → recurring every 30 minutes
        "0 9 * * *"        → cron expression
        "2026-02-03T14:00" → once at timestamp
    """
    schedule = schedule.strip()
    original = schedule
    schedule_lower = schedule.lower()

    if schedule_lower.startswith("every "):
        minutes = parse_duration(schedule[6:].strip())
        return {"kind": "interval", "minutes": minutes, "display": f"every {minutes}m"}

    parts = schedule.split()
    if len(parts) >= 5 and all(re.match(r'^[\d\*\-,/]+$', p) for p in parts[:5]):
        if not HAS_CRONITER:
            raise ValueError("Cron expressions require 'croniter'. Install: pip install croniter")
        try:
            croniter(schedule)
        except Exception as e:
            raise ValueError(f"Invalid cron expression '{schedule}': {e}")
        return {"kind": "cron", "expr": schedule, "display": schedule}

    if 'T' in schedule or re.match(r'^\d{4}-\d{2}-\d{2}', schedule):
        try:
            dt = datetime.fromisoformat(schedule.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.astimezone()
            return {"kind": "once", "run_at": dt.isoformat(),
                    "display": f"once at {dt.strftime('%Y-%m-%d %H:%M')}"}
        except ValueError as e:
            raise ValueError(f"Invalid timestamp '{schedule}': {e}")

    try:
        minutes = parse_duration(schedule)
        run_at = _now() + timedelta(minutes=minutes)
        return {"kind": "once", "run_at": run_at.isoformat(), "display": f"once in {original}"}
    except ValueError:
        pass

    raise ValueError(
        f"Invalid schedule '{original}'. Use:\n"
        f"  - Duration: '30m', '2h' (one-shot)\n"
        f"  - Interval: 'every 30m' (recurring)\n"
        f"  - Cron: '0 9 * * *'\n"
        f"  - Timestamp: '2026-02-03T14:00:00'"
    )


def compute_next_run(schedule: Dict[str, Any], last_run_at: Optional[str] = None) -> Optional[str]:
    """Compute next run time. Returns ISO string or None."""
    now = _now()

    if schedule["kind"] == "once":
        if last_run_at:
            return None
        run_at = schedule.get("run_at")
        if run_at:
            run_at_dt = datetime.fromisoformat(run_at)
            if run_at_dt.tzinfo is None:
                run_at_dt = run_at_dt.astimezone()
            if run_at_dt >= now - timedelta(seconds=ONESHOT_GRACE_SECONDS):
                return run_at
        return None

    elif schedule["kind"] == "interval":
        minutes = schedule["minutes"]
        if last_run_at:
            last = datetime.fromisoformat(last_run_at)
            return (last + timedelta(minutes=minutes)).isoformat()
        return (now + timedelta(minutes=minutes)).isoformat()

    elif schedule["kind"] == "cron":
        if not HAS_CRONITER:
            return None
        base = datetime.fromisoformat(last_run_at) if last_run_at else now
        cron = croniter(schedule["expr"], base)
        return cron.get_next(datetime).isoformat()

    return None


def create_job(
    prompt: str,
    schedule: str,
    name: Optional[str] = None,
    repeat: Optional[int] = None,
    deliver: Optional[str] = None,
    origin: Optional[Dict[str, Any]] = None,
    app_id: Optional[int] = None,
    persona_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a new cron job in PostgreSQL.

    ``persona_id`` assigns a SPECIFIC persona to this task (per-task
    governance); NULL means the job inherits the app's default persona at run
    time (resolved by the runner's persona applicator)."""
    db = _get_db()
    parsed = parse_schedule(schedule)

    if repeat is not None and repeat <= 0:
        repeat = None
    if parsed["kind"] == "once" and repeat is None:
        repeat = 1
    if deliver is None:
        deliver = "origin" if origin else "local"

    job_uuid = uuid.uuid4().hex[:12]
    next_run = compute_next_run(parsed)
    job_name = name or prompt[:50].strip()

    row = db.execute_returning("""
        INSERT INTO ag_agents.cron_jobs (
            job_uuid, name, prompt, schedule, schedule_kind, schedule_json,
            is_active, state, next_run_at, repeat_times, repeat_completed,
            deliver, origin_json, app_id, persona_id
        ) VALUES (
            %s, %s, %s, %s, %s, %s::jsonb,
            TRUE, 'scheduled', %s, %s, 0,
            %s, %s::jsonb, %s, %s
        ) RETURNING *
    """, (
        job_uuid, job_name, prompt, parsed.get("display", schedule),
        parsed["kind"], json.dumps(parsed),
        next_run, repeat,
        deliver, json.dumps(origin) if origin else None,
        app_id or 1, persona_id
    ))

    return _row_to_job(row)


def update_job(job_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Update an existing cron job."""
    db = _get_db()
    job = get_job(job_id)
    if not job:
        return None

    db_id = job.get("db_id")
    updates = []
    params = []

    if "prompt" in kwargs:
        updates.append("prompt = %s")
        params.append(kwargs["prompt"])
    if "name" in kwargs:
        updates.append("name = %s")
        params.append(kwargs["name"])
    if "deliver" in kwargs:
        updates.append("deliver = %s")
        params.append(kwargs["deliver"])
    if kwargs.get("app_id") is not None:
        # The owning application. Without this branch the field was silently
        # dropped, so the admin UI's "change application" never persisted —
        # and (with the identity fix) it is the app whose connections the
        # cron authenticates with.
        updates.append("app_id = %s")
        params.append(kwargs["app_id"])
    if "persona_id" in kwargs:
        # Per-task persona assignment. Explicit None clears it → the job falls
        # back to the app's default persona at run time.
        updates.append("persona_id = %s")
        params.append(kwargs["persona_id"])
    if "enabled" in kwargs:
        updates.append("is_active = %s")
        params.append(kwargs["enabled"])
    if "repeat" in kwargs:
        repeat = kwargs["repeat"]
        if repeat is not None and repeat <= 0:
            repeat = None
        updates.append("repeat_times = %s")
        params.append(repeat)
    
    if "schedule" in kwargs and kwargs["schedule"] is not None:
        schedule = kwargs["schedule"]
        parsed = parse_schedule(schedule)
        next_run = compute_next_run(parsed)
        updates.extend([
            "schedule = %s",
            "schedule_kind = %s",
            "schedule_json = %s::jsonb",
            "next_run_at = %s",
            "state = 'scheduled'"
        ])
        params.extend([
            parsed.get("display", schedule),
            parsed["kind"],
            json.dumps(parsed),
            next_run
        ])

    if not updates:
        return job

    updates.append("updated_at = NOW()")
    query = f"""
        UPDATE ag_agents.cron_jobs 
        SET {', '.join(updates)}
        WHERE job_id = %s
        RETURNING *
    """
    params.append(db_id)
    
    row = db.execute_returning(query, tuple(params))
    return _row_to_job(row)


def _row_to_job(row: Dict) -> Dict[str, Any]:
    """Convert a DB row to the legacy job dict format."""
    if not row:
        return {}
    schedule_json = row.get("schedule_json") or {}
    if isinstance(schedule_json, str):
        schedule_json = json.loads(schedule_json)
    origin = row.get("origin_json") or {}
    if isinstance(origin, str):
        origin = json.loads(origin)

    return {
        "id": row.get("job_uuid", str(row.get("job_id", ""))),
        "db_id": row.get("job_id"),
        "name": row.get("name", ""),
        "prompt": row.get("prompt", ""),
        "schedule": schedule_json,
        "schedule_display": row.get("schedule", ""),
        "repeat": {
            "times": row.get("repeat_times"),
            "completed": row.get("repeat_completed", 0),
        },
        "enabled": row.get("is_active", True),
        "state": row.get("state", "scheduled"),
        "created_at": str(row["created_at"]) if row.get("created_at") else "",
        "next_run_at": str(row["next_run_at"]) if row.get("next_run_at") else None,
        "last_run_at": str(row["last_run_at"]) if row.get("last_run_at") else None,
        "last_status": row.get("last_status") or "",
        "last_error": row.get("last_error") or "",
        "deliver": row.get("deliver", "local"),
        "app_id": row.get("app_id"),
        "persona_id": row.get("persona_id"),
        "origin": origin,
    }


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a job by UUID or DB ID."""
    db = _get_db()
    row = db.fetch_one(
        "SELECT * FROM ag_agents.cron_jobs WHERE job_uuid = %s OR job_id::text = %s",
        (job_id, job_id)
    )
    return _row_to_job(row) if row else None


def list_jobs(include_disabled: bool = False) -> List[Dict[str, Any]]:
    """List all jobs."""
    db = _get_db()
    if include_disabled:
        rows = db.fetch_all("SELECT * FROM ag_agents.cron_jobs ORDER BY created_at DESC")
    else:
        rows = db.fetch_all(
            "SELECT * FROM ag_agents.cron_jobs WHERE is_active = TRUE ORDER BY created_at DESC"
        )
    return [_row_to_job(r) for r in rows]


def remove_job(job_id: str) -> bool:
    """Remove a job by UUID or DB ID."""
    db = _get_db()
    count = db.execute(
        "DELETE FROM ag_agents.cron_jobs WHERE job_uuid = %s OR job_id::text = %s",
        (job_id, job_id)
    )
    return count > 0


def get_due_jobs() -> List[Dict[str, Any]]:
    """Get all jobs that are due to run now."""
    db = _get_db()
    rows = db.fetch_all("""
        SELECT * FROM ag_agents.cron_jobs
        WHERE is_active = TRUE
          AND state != 'paused'
          AND next_run_at IS NOT NULL
          AND next_run_at <= NOW()
        ORDER BY next_run_at ASC
    """)
    return [_row_to_job(r) for r in rows]


def mark_job_run(job_id: str, success: bool, error: Optional[str] = None):
    """Mark a job as having been run. Updates next_run, increments count.
    
    Tracks consecutive_failures for circuit breaker:
    - On success: resets consecutive_failures to 0
    - On failure: increments consecutive_failures
    """
    db = _get_db()
    job = get_job(job_id)
    if not job:
        return

    now_iso = _now().isoformat()
    db_id = job.get("db_id")

    repeat_times = job["repeat"]["times"]
    repeat_completed = job["repeat"]["completed"] + 1

    if repeat_times is not None and repeat_times > 0 and repeat_completed >= repeat_times:
        db.execute("DELETE FROM ag_agents.cron_jobs WHERE job_id = %s", (db_id,))
        return

    schedule = job.get("schedule", {})
    next_run = compute_next_run(schedule, now_iso)

    new_state = "scheduled" if next_run else "completed"
    is_active = next_run is not None

    db.execute("""
        UPDATE ag_agents.cron_jobs SET
            last_run_at = NOW(),
            last_status = %s,
            last_error = %s,
            repeat_completed = %s,
            next_run_at = %s,
            state = %s,
            is_active = %s,
            run_count = run_count + 1,
            consecutive_failures = CASE WHEN %s THEN 0 ELSE COALESCE(consecutive_failures, 0) + 1 END,
            updated_at = NOW()
        WHERE job_id = %s
    """, (
        "ok" if success else "error",
        error if not success else None,
        repeat_completed,
        next_run,
        new_state,
        is_active,
        success,
        db_id,
    ))


def save_job_output(job_id: str, content: str):
    """Save job output to the cron_job_outputs table."""
    db = _get_db()
    job = get_job(job_id)
    if not job:
        return
    db_id = job.get("db_id")
    db.execute("""
        INSERT INTO ag_agents.cron_job_outputs (job_id, content)
        VALUES (%s, %s)
    """, (db_id, content))
