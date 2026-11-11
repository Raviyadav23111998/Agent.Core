"""Cron jobs must run under their owning app's identity so that app-scoped
connections (Jira, M365, SMTP…) resolve — the fix for the fleet Jira cron
that failed with user_id=None app_id=None while the bot worked.
"""

import asyncio

from core.user_context import create_cron_user, create_agent_user, CRON_USER


# ── identity plumbing ────────────────────────────────────────────────

def test_cron_user_carries_app_id():
    ctx = create_cron_user(app_id=4)
    assert ctx.user_id == "__cron__"
    assert ctx.app_id == "4"
    assert ctx.platform == "cron"
    # legacy singleton still has none (proves the old behaviour was the bug)
    assert CRON_USER.app_id is None


def test_cron_user_without_app_id_is_none():
    assert create_cron_user().app_id is None
    assert create_cron_user(app_id=None).app_id is None


def test_agent_user_takes_explicit_app_id_over_parent():
    parent = create_cron_user(app_id=9)
    ctx = create_agent_user("email-to-jira-bot", parent_context=parent, app_id=4)
    assert ctx.app_id == "4"
    assert ctx.user_id == "__agent__email-to-jira-bot__"


def test_agent_user_falls_back_to_parent_app_id():
    parent = create_cron_user(app_id=4)
    ctx = create_agent_user("bot", parent_context=parent)
    assert ctx.app_id == "4"


# ── scheduler threads job.app_id into run_agent ──────────────────────

def test_scheduler_passes_app_id_to_run_agent(monkeypatch):
    from cron import scheduler as sched

    captured = {}

    # Contract: the scheduler threads BOTH the job's app_id (credential scope)
    # and persona_id (per-task governance) into the runner.
    async def fake_run_agent(prompt, app_id=None, persona_id=None):
        captured["prompt"] = prompt
        captured["app_id"] = app_id
        captured["persona_id"] = persona_id
        return "ok"

    monkeypatch.setattr(sched, "_get_db", lambda: None)
    monkeypatch.setattr(sched, "save_job_output", lambda *a, **k: None)
    marks = {}
    monkeypatch.setattr(sched, "mark_job_run",
                        lambda jid, ok, error=None: marks.update(ok=ok))

    job = {"id": "j1", "name": "fleet", "prompt": "[AGENT:x] do it",
           "app_id": 4, "persona_id": 9}
    asyncio.run(sched._run_job(job, fake_run_agent))

    assert captured["app_id"] == 4
    assert captured["persona_id"] == 9
    assert marks["ok"] is True


# ── update_job persists app_id (admin UI "change application") ───────

def test_update_job_persists_app_id(monkeypatch):
    from cron import jobs

    captured = {}

    class _FakeDB:
        def execute_returning(self, query, params):
            captured["query"] = query
            captured["params"] = params
            return {"job_id": 7, "job_uuid": "u7", "app_id": params[0],
                    "name": "fleet", "prompt": "p", "schedule": "every 1h",
                    "schedule_kind": "interval", "schedule_json": {},
                    "is_active": True, "deliver": "local", "repeat_times": None,
                    "repeat_completed": 0, "state": "scheduled",
                    "next_run_at": None, "last_run_at": None, "last_status": None,
                    "run_count": 0, "origin_json": None}

    monkeypatch.setattr(jobs, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(jobs, "get_job",
                        lambda jid: {"db_id": 7, "id": "u7"})

    result = jobs.update_job("u7", app_id=4)
    assert "app_id = %s" in captured["query"]
    assert 4 in captured["params"]
    assert result["app_id"] == 4


def test_update_job_omits_app_id_when_not_given(monkeypatch):
    from cron import jobs

    captured = {}

    class _FakeDB:
        def execute_returning(self, query, params):
            captured["query"] = query
            return {"job_id": 7, "app_id": 1, "name": "x", "prompt": "p",
                    "schedule": "every 1h", "schedule_kind": "interval",
                    "schedule_json": {}, "is_active": True, "deliver": "local",
                    "repeat_times": None, "repeat_completed": 0,
                    "state": "scheduled", "next_run_at": None,
                    "last_run_at": None, "last_status": None, "run_count": 0,
                    "job_uuid": "u7", "origin_json": None}

    monkeypatch.setattr(jobs, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(jobs, "get_job", lambda jid: {"db_id": 7, "id": "u7"})

    jobs.update_job("u7", name="renamed")
    assert "app_id = %s" not in captured["query"]


# ── Jira credential resolution: cron context now reaches app scope ───

class _Cred:
    def __init__(self, config, secret):
        self.config = config
        self.secret = secret


class _FakeRegistry:
    """Mimics Connections.resolve's user→app→org chain for a direct provider.
    Only an APP-scoped Jira connection exists (owner app_id=4), like prod."""

    def __init__(self):
        self._app_conn = {4: _Cred(
            {"server_url": "https://acme.atlassian.net", "email": "bot@acme.com"},
            "APITOKEN")}

    def resolve(self, provider, *, user_id=None, app_id=None):
        from core.connections import ConnectionNotConfigured
        # user scope: none configured
        # app scope:
        if app_id is not None:
            key = int(app_id) if str(app_id).isdigit() else None
            if key in self._app_conn:
                return self._app_conn[key]
        # org scope: none configured
        raise ConnectionNotConfigured(provider)


def _resolve_with_context(ctx):
    from tools.jira_tools import JiraActionTools
    tools = JiraActionTools(credential_registry=_FakeRegistry(),
                            user_context_resolver=lambda: ctx)
    return tools._resolve_credentials()


def test_jira_resolves_for_cron_with_app_id():
    ctx = create_agent_user("email-to-jira-bot",
                            parent_context=None, app_id=4)
    server, email, token = _resolve_with_context(ctx)
    assert server == "https://acme.atlassian.net"
    assert email == "bot@acme.com"
    assert token == "APITOKEN"


def test_jira_fails_for_bare_cron_without_app_id():
    # The pre-fix behaviour: bare CRON_USER → app_id None → org-only → fail.
    server, email, token = _resolve_with_context(CRON_USER)
    assert server is None and email is None and token is None
