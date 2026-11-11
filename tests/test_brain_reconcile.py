"""
Brain ingestion reconciliation — per-pillar state + idempotent reindex + the
self-healing reconciler loop. Validated with injected fakes (no live PG/ES/Neo4j).
"""
import os
import sys
import asyncio
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brain.storage.index_state import _backoff_seconds
from brain.storage.manager import StorageManager
from brain.ingestion.reconciler import BrainReconciler


def test_backoff_grows_and_caps():
    assert _backoff_seconds(1) == 60
    assert _backoff_seconds(2) == 120
    assert _backoff_seconds(3) == 240
    assert _backoff_seconds(100) == 3600


class FakeState:
    def __init__(self):
        self.marks = []
        self.seeded = None
        self._due = []

    def seed(self, doc_id, configured):
        self.seeded = (doc_id, configured)

    def mark(self, doc_id, pillar, state, error=None):
        self.marks.append((doc_id, pillar, state))

    def due(self, limit=50):
        return self._due[:limit]

    def health(self):
        return {}


def _mgr(**attrs):
    """A StorageManager with __init__ bypassed — only the attrs the
    orchestration methods touch are set, so we test logic without a DB."""
    m = StorageManager.__new__(StorageManager)
    m.lexical = None
    m.graph = None
    m.embeddings = None
    m.extractor = None
    m.documents = None
    m.index_state = FakeState()
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def test_configured_pillars():
    assert _mgr()._configured_pillars() == {"lexical": False, "graph": False, "vector": False}
    m = _mgr(lexical=object(), graph=object(), embeddings=SimpleNamespace(available=True))
    assert m._configured_pillars() == {"lexical": True, "graph": True, "vector": True}
    m2 = _mgr(embeddings=SimpleNamespace(available=False))
    assert m2._configured_pillars()["vector"] is False


def test_attempt_pillar_marks_ok_and_failed():
    m = _mgr()
    assert m._attempt_pillar("d", "lexical", lambda: None) is True

    def boom():
        raise RuntimeError("ES down")
    assert m._attempt_pillar("d", "lexical", boom) is False
    assert ("d", "lexical", "ok") in m.index_state.marks
    assert ("d", "lexical", "failed") in m.index_state.marks


def test_reindex_unconfigured_marks_skipped():
    m = _mgr()
    ok, err = m.reindex("d", "vector")
    assert ok and ("d", "vector", "skipped") in m.index_state.marks


def test_reindex_vector_writes_and_marks_ok():
    class Emb:
        available = True
        model = "nomic-embed-text"

        def embed_one(self, text):
            return [0.01] * 768

    class Docs:
        def __init__(self):
            self.saved = None

        def get_document(self, doc_id):
            return SimpleNamespace(id=doc_id, title="t", content="body",
                                   owner_id="o", app_id=1, metadata={},
                                   document_type=SimpleNamespace(value="document"))

        def update_embedding(self, doc_id, vector, model=""):
            self.saved = (doc_id, len(vector), model)

    docs = Docs()
    m = _mgr(embeddings=Emb(), documents=docs)
    m._chunking_enabled = lambda: False

    ok, err = m.reindex("d1", "vector")
    assert ok and err == ""
    assert docs.saved == ("d1", 768, "nomic-embed-text")
    assert ("d1", "vector", "ok") in m.index_state.marks


def test_reindex_failure_marks_failed():
    class Emb:
        available = True
        model = "m"

        def embed_one(self, text):
            raise RuntimeError("ollama unreachable")

    class Docs:
        def get_document(self, doc_id):
            return SimpleNamespace(id=doc_id, title="t", content="b", owner_id="o",
                                   app_id=1, metadata={},
                                   document_type=SimpleNamespace(value="document"))

    m = _mgr(embeddings=Emb(), documents=Docs())
    ok, err = m.reindex("d1", "vector")
    assert ok is False and "ollama" in err
    assert ("d1", "vector", "failed") in m.index_state.marks


def test_reconciler_heals_due_and_reports():
    class FakeSM:
        def __init__(self):
            self.index_state = FakeState()
            self.index_state._due = [("d1", "vector"), ("d2", "lexical")]
            self.calls = []

        def reindex(self, doc_id, pillar):
            self.calls.append((doc_id, pillar))
            return (pillar == "vector", "" if pillar == "vector" else "boom")

    sm = FakeSM()
    stats = asyncio.run(BrainReconciler(sm, batch_size=10).tick())
    assert stats == {"processed": 2, "healed": 1, "failed": 1}
    assert sm.calls == [("d1", "vector"), ("d2", "lexical")]


def test_reconciler_noop_when_nothing_due():
    class FakeSM:
        index_state = FakeState()
    assert asyncio.run(BrainReconciler(FakeSM()).tick())["processed"] == 0


def test_health_probe_real_status_per_pillar():
    class Client:
        def ping(self):
            return True

        def count(self, index):
            return {"count": 7}

    class Lex:
        _client = Client()
        index_name = "brain_x"

    class GraphBad:
        _driver = None

        def connect(self):
            raise RuntimeError("Invalid credential")

    class Emb:
        available = True
        model = "nomic-embed-text"

        def embed_one(self, t):
            return [0.0] * 768

    class Docs:
        def count_documents(self):
            return {"total_documents": 10}

    class State2(FakeState):
        def health(self):
            return {"lexical": {"ok": 7, "pending": 3},
                    "graph": {"pending": 10},
                    "vector": {"ok": 10}}

    m = _mgr(lexical=Lex(), graph=GraphBad(), embeddings=Emb(), documents=Docs())
    m._db = SimpleNamespace(fetch_one=lambda sql, params=None: {"x": 1})
    m.index_state = State2()

    rep = m.health_probe()
    assert rep["documents"] == 10
    assert rep["components"]["document_store"]["status"] == "healthy"
    assert rep["components"]["vector"]["status"] == "healthy"
    assert rep["components"]["vector"]["dim"] == 768
    assert rep["components"]["lexical"]["status"] == "degraded"
    assert rep["components"]["graph"]["status"] == "unhealthy"
    assert rep["status"] == "unhealthy"
