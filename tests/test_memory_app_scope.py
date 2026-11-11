"""Memory and skills are scoped to their APPLICATION, not only their user.

The leak these pin was real and is worth stating plainly, because the shape of
it is what makes the tests below the right ones.

ag_memory.memories has always had an app_id column. Nothing ever read it, and
nothing ever wrote it either — it is NOT NULL DEFAULT 1, so every row from
every application claimed to be application 1. Isolation therefore rested
entirely on user_id, which holds right up until a SERVICE KEY is involved: a
key spends as the human who minted it, so an application calling the platform
for its own end users runs as that person and reads their private memories.

Observed in production: an exam-question authoring app recalling a platform
admin's personal research, including a named private individual's profile, into
prompts that write questions for schoolchildren.

UserContext already carried app_id (cron needed it for credentials — see
test_cron_app_identity.py). Nothing needed plumbing. The identity arrived
intact at a store that discarded it.
"""

import pytest

from state.memory_store import MemoryStore, GLOBAL_USER


class FakeDB:
    """Records the SQL and params a call would have issued.

    A fake rather than a real database because the claim under test is about
    the WHERE clause, and a fake makes an unscoped query a visible assertion
    instead of an empty result set that could mean anything.
    """

    def __init__(self, rows=None):
        self.rows = rows or []
        self.queries = []

    def _record(self, sql, params=None):
        self.queries.append((" ".join(sql.split()), params or ()))

    def fetch_one(self, sql, params=None):
        self._record(sql, params)
        return None

    def fetch_all(self, sql, params=None):
        self._record(sql, params)
        return list(self.rows)

    def execute(self, sql, params=None):
        self._record(sql, params)
        return 1

    def execute_returning(self, sql, params=None):
        self._record(sql, params)
        return {"memory_id": 1}

    @property
    def last(self):
        return self.queries[-1]

    def any_query_containing(self, needle):
        return [q for q in self.queries if needle in q[0]]


# ── reads are confined to the calling application ────────────────────────

def test_recall_filters_by_app_id():
    db = FakeDB()
    MemoryStore(db=db).recall("kinematics", user_id="user_1", app_id=7)

    scoped = db.any_query_containing("app_id = %s")
    assert scoped, "recall issued no app-scoped query — this is the leak"
    assert all(7 in q[1] for q in scoped)


def test_recall_without_an_app_is_unscoped_not_empty():
    # The CLI has no application and one operator. Failing it closed would
    # break local use to fix a server-side leak, so app_id=None means
    # "unscoped" — every caller that HAS an app passes it, and that is the
    # case that leaked.
    db = FakeDB()
    MemoryStore(db=db).recall("anything", user_id="__cli__")

    assert not db.any_query_containing("app_id = %s")


def test_the_actual_leak_one_app_cannot_read_anothers_user_rows():
    # The precise production shape: a service key spends as user 1, so the
    # authoring app (7) runs as "user_1" — the same user_id whose bucket holds
    # the admin's private notes. Only the app dimension separates them.
    db = FakeDB()
    MemoryStore(db=db).recall("highway", user_id="user_1", app_id=7)

    sql, params = db.any_query_containing("app_id = %s")[0]
    assert "user_id" in sql and "app_id = %s" in sql
    assert 7 in params, "the app filter must carry the CALLER's app, not the row's"


def test_recall_by_category_is_scoped_too():
    # A second hand-written copy of the WHERE clause is exactly how one read
    # path ends up scoped and its neighbour does not.
    db = FakeDB()
    MemoryStore(db=db).recall_by_category(user_id="user_1", category="fact", app_id=7)

    assert db.any_query_containing("app_id = %s")


def test_get_user_context_is_scoped_too():
    # The widest read in the class — everything, unfiltered by query — so a
    # missing app filter here leaks the most in a single call.
    db = FakeDB(rows=[{"category": "fact", "key": "k", "value": "v"}])
    MemoryStore(db=db).get_user_context(user_id="user_1", app_id=7)

    assert db.any_query_containing("app_id = %s")


def test_get_context_memories_is_scoped_too():
    # The INJECTION path: what comes back is pasted into a prompt without
    # anybody choosing it, so an unscoped read is worse here than in a tool.
    db = FakeDB()
    MemoryStore(db=db).get_context_memories("query", user_id="user_1", app_id=7)

    assert db.any_query_containing("app_id = %s")


def test_visible_scopes_still_apply_alongside_the_app_filter():
    # The Teams-style group model must survive: app scoping narrows WITHIN the
    # user dimension, it does not replace it.
    db = FakeDB()
    MemoryStore(db=db).recall("q", user_id="user_1",
                              visible_scopes=["user_1", GLOBAL_USER], app_id=7)

    sql, params = db.any_query_containing("app_id = %s")[0]
    assert "user_id = ANY(%s)" in sql
    assert ["user_1", GLOBAL_USER] in params and 7 in params


# ── writes stamp the real application ────────────────────────────────────

def test_remember_writes_the_app_id():
    db = FakeDB()
    MemoryStore(db=db).remember("k", "v", user_id="user_1", app_id=7)

    sql, params = db.last
    assert "app_id" in sql
    assert 7 in params


def test_remember_without_an_app_falls_back_to_the_column_default():
    # COALESCE, not a Python default: the column is NOT NULL, so passing None
    # must let the database's own default apply rather than failing the insert.
    db = FakeDB()
    MemoryStore(db=db).remember("k", "v", user_id="__cli__")

    sql, params = db.last
    assert "COALESCE(%s, 1)" in sql
    assert None in params


def test_upsert_conflict_target_includes_the_app():
    # (user_id, key) meant two applications could not hold the same key for one
    # person — the second would silently UPDATE the first's memory through
    # ON CONFLICT. The constraint has to follow the scope.
    db = FakeDB()
    MemoryStore(db=db).remember("pref.tone", "v", user_id="user_1", app_id=7)

    sql, _ = db.last
    assert "ON CONFLICT (app_id, user_id, key)" in sql


# ── the identity the platform already establishes ────────────────────────

def test_app_id_is_read_from_the_ambient_user_context():
    # The fix depends on UserContext.app_id being populated by every entry
    # point. It is — this pins that the memory tool reads the same contextvar
    # skill_service._ambient_scope does, rather than inventing a second answer
    # to "which app is this?".
    from core.user_context import UserContext, current_user_context
    from tools.memory_mcp import MemoryMCPTools

    tools = MemoryMCPTools(memory_store=None)
    token = current_user_context.set(UserContext(user_id="user_1", app_id="7"))
    try:
        assert tools._resolve_app_id() == 7
    finally:
        current_user_context.reset(token)


def test_no_ambient_context_resolves_to_no_app():
    from core.user_context import current_user_context
    from tools.memory_mcp import MemoryMCPTools

    token = current_user_context.set(None)
    try:
        assert MemoryMCPTools(memory_store=None)._resolve_app_id() is None
    finally:
        current_user_context.reset(token)
