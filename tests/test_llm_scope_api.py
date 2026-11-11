"""Phase 2 — LLM scope service (agent.api): resolution order, budget
messages, ledger batching. Fake-DB level (no Postgres)."""

import os
import sys
import time
import queue

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'agent.api'))

from api.services.llm_scope import LLMScopeService, numeric_user_id  # noqa: E402


class FakeDB:
    """Answers the exact queries LLMProfileRepository issues."""

    def __init__(self):
        self.profiles = {}          # profile_id -> row
        self.app_binding = {}       # app_id -> profile_id
        self.user_binding = {}      # user_id -> profile_id
        self.budgets = []
        self.spend = {"tokens": 0, "usd": 0.0}
        self.inserted = []

    def fetch_one(self, sql, params=None):
        if "JOIN ag_system.applications" in sql:
            pid = self.app_binding.get(params[0])
            return self.profiles.get(pid)
        if "JOIN ag_auth.users" in sql:
            pid = self.user_binding.get(params[0])
            return self.profiles.get(pid)
        if "is_default AND is_active" in sql:
            return next((p for p in self.profiles.values()
                         if p.get("is_default")), None)
        if "name = %s OR model = %s" in sql:
            return next((p for p in self.profiles.values()
                         if p["name"] == params[0] or p["model"] == params[1]),
                        None)
        if "SUM(prompt_tokens + completion_tokens)" in sql:
            return dict(self.spend)
        return None

    def fetch_all(self, sql, params=None):
        if "FROM ag_system.budgets" in sql:
            return list(self.budgets)
        return []

    def execute(self, sql, params=None):
        if "INSERT INTO ag_system.llm_usage" in sql:
            self.inserted.append(params)


def _profile(pid, name, model, default=False):
    return {"profile_id": pid, "name": name, "provider": "deepseek",
            "model": model, "base_url": "https://x/v1/chat/completions",
            "api_key": "k", "is_default": default,
            "updated_at": "2026-07-04"}


def make_service(db):
    svc = LLMScopeService.__new__(LLMScopeService)  # skip writer thread
    from api.repositories.llm_profile_repo import LLMProfileRepository
    import threading
    svc._db = db
    svc._repo = LLMProfileRepository(db)
    svc._budget_cache = {}
    svc._budget_lock = threading.Lock()
    return svc


def test_resolution_order_request_user_app_default():
    db = FakeDB()
    db.profiles = {1: _profile(1, "default", "deepseek-v4-flash", default=True),
                   2: _profile(2, "coding-glm", "glm-5.2"),
                   3: _profile(3, "user-pref", "deepseek-v4-pro")}
    db.app_binding = {10: 2}
    db.user_binding = {5: 3}
    svc = make_service(db)

    # request param (profile name or model id) wins over everything
    assert svc.resolve_profile(user_id=5, app_id=10,
                               requested_model="coding-glm")["profile_id"] == 2
    assert svc.resolve_profile(requested_model="glm-5.2")["profile_id"] == 2
    # unknown request model falls through (never a free-form endpoint)
    assert svc.resolve_profile(user_id=5, app_id=10,
                               requested_model="gpt-99")["profile_id"] == 3
    # user > app
    assert svc.resolve_profile(user_id=5, app_id=10)["profile_id"] == 3
    # app > default
    assert svc.resolve_profile(user_id=99, app_id=10)["profile_id"] == 2
    # default floor
    assert svc.resolve_profile(user_id=99, app_id=99)["profile_id"] == 1


def test_budget_block_message_and_warn_passthrough():
    db = FakeDB()
    svc = make_service(db)
    db.budgets = [{"scope_type": "user", "scope_id": 5, "win": "day",
                   "max_tokens": 1000, "max_usd": 0, "action": "block"}]
    db.spend = {"tokens": 1500, "usd": 0.10}
    msg = svc.check_budgets(user_id=5, app_id=None)
    assert msg and "limit" in msg and "Resets at" in msg and "1,000" in msg

    # warn-only budget logs but never blocks
    db.budgets[0]["action"] = "warn"
    svc._budget_cache.clear()
    assert svc.check_budgets(user_id=5, app_id=None) is None

    # under cap → allowed
    db.budgets[0]["action"] = "block"
    db.spend = {"tokens": 10, "usd": 0.0}
    svc._budget_cache.clear()
    assert svc.check_budgets(user_id=5, app_id=None) is None


def test_money_cap_blocks_independently():
    db = FakeDB()
    svc = make_service(db)
    db.budgets = [{"scope_type": "app", "scope_id": 7, "win": "month",
                   "max_tokens": 0, "max_usd": 5.0, "action": "block"}]
    db.spend = {"tokens": 100, "usd": 6.25}
    msg = svc.check_budgets(user_id=None, app_id=7)
    assert msg and "$6.25/$5.00" in msg


def test_usage_sink_batches_and_strips_tenant_ctx():
    db = FakeDB()
    svc = make_service(db)
    import queue
    svc._queue = queue.Queue()
    sink = svc.make_usage_sink(user_id=5, app_id=10, run_kind="chat",
                               profile_id=2)
    sink({"model": "glm-5.2", "prompt_tokens": 100, "completion_tokens": 20,
          "cached_tokens": 80, "cost_usd": 0.0001, "conversation_id": "c1"})
    rows = [svc._queue.get_nowait()]
    assert rows[0]["user_id"] == 5 and rows[0]["run_kind"] == "chat"
    svc._flush_batch(rows)
    assert len(db.inserted) == 1
    flat = db.inserted[0]
    assert 5 in flat and 10 in flat and "glm-5.2" in flat


def test_numeric_user_id():
    assert numeric_user_id("user_5") == 5
    assert numeric_user_id(7) == 7
    assert numeric_user_id("7") == 7
    assert numeric_user_id("embed_abc") is None
    assert numeric_user_id("agent_researcher") is None
    assert numeric_user_id(None) is None


def test_sink_normalizes_raw_context_ids():
    """'user_5' and bot-site ids must never reach the INTEGER ledger column
    raw — one bad row would fail the whole batched INSERT."""
    db = FakeDB()
    svc = LLMScopeService.__new__(LLMScopeService)
    svc._db = db
    from api.repositories.llm_profile_repo import LLMProfileRepository
    svc._repo = LLMProfileRepository(db)
    svc._queue = queue.Queue()

    sink = svc.make_usage_sink(user_id="user_5", app_id=4, run_kind="subagent")
    sink({"model": "m", "prompt_tokens": 1, "completion_tokens": 1,
          "cached_tokens": 0, "cost_usd": 0.0, "conversation_id": "c"})
    row = svc._queue.get_nowait()
    assert row["user_id"] == 5

    sink = svc.make_usage_sink(user_id="site-user-abc", app_id=4, run_kind="bot")
    sink({"model": "m", "prompt_tokens": 1, "completion_tokens": 1,
          "cached_tokens": 0, "cost_usd": 0.0, "conversation_id": "c"})
    assert svc._queue.get_nowait()["user_id"] is None


def test_flush_batch_salvages_good_rows():
    """One malformed row must not erase the rest of the batch."""
    class ExplodingOnBatchDB(FakeDB):
        def execute(self, sql, params=None):
            if "INSERT INTO ag_system.llm_usage" in sql:
                # multi-row VALUES → simulate whole-batch failure once
                if sql.count("(%s") > 1:
                    raise RuntimeError("bad row in batch")
                super().execute(sql, params)

    db = ExplodingOnBatchDB()
    svc = LLMScopeService.__new__(LLMScopeService)
    svc._db = db
    from api.repositories.llm_profile_repo import LLMProfileRepository
    svc._repo = LLMProfileRepository(db)
    rows = [{"model": "m", "prompt_tokens": 1, "completion_tokens": 1,
             "cached_tokens": 0, "cost_usd": 0.0, "conversation_id": f"c{i}",
             "user_id": 1, "app_id": 2, "run_kind": "chat",
             "profile_id": None, "_tenant_ctx": None} for i in range(3)]
    svc._flush_batch(rows)
    assert len(db.inserted) == 3  # salvaged row-by-row
