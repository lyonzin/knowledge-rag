"""The owner's startup publishes ready only once the index is open and prepared.

In-process tests of ``server._writer_startup`` over a fake index whose opening
or embedding migration waits on an Event: meanwhile tools and the health probe
answer "repairing" and nothing else opens the index; a failure answers
"error" and never passes through ready. With the real preflight over fake
probes, the index opens only after a probe got inside and validated it.
No Chroma, no model, no sleeps.
"""

from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import MagicMock

import pytest
from support.harmless_probe import busy as BUSY
from support.harmless_probe import healthy as HEALTHY
from support.tool_calls import TOOL_CALLS

from mcp_server import _index_access, _writer_role, preflight, server
from mcp_server.health import HealthMiddleware

WAIT_SECONDS = 10
SEARCH = TOOL_CALLS["search_knowledge"]
STARTUP_THREAD = "writer-startup"  # the name _start_writer_startup gives its thread
REBUILT = {"indexed": 1, "chunks_added": 1, "errors": 0, "elapsed_seconds": 0.0}


class _ReadyRecorder(threading.Event):
    """STATE.ready that counts how often the startup publishes it."""

    def __init__(self):
        super().__init__()
        self.published = 0

    def set(self):
        self.published += 1
        super().set()


class _FakeIndex:
    """What the startup and search_knowledge use of KnowledgeOrchestrator; it notes which thread built it."""

    built_by: list[str] = []

    def __init__(self):
        type(self).built_by.append(threading.current_thread().name)
        self.collection = MagicMock()
        self.collection.count.return_value = 1  # populated: the startup indexes nothing
        self.query_cache = MagicMock()
        self.query_cache.stats.return_value = {"hit_rate": "0%"}  # what search_knowledge reports

    def _check_dimension_mismatch(self):
        return False

    def _start_bm25_background_build(self, thread_name):
        """The BM25 pre-warm (GH #321: a background build) has nothing to build here."""

    def query(self, query, **options):
        return [{"content": f"hit for {query}", "source": "notes/hit.md", "score": 1.0}]


@pytest.fixture
def owner(monkeypatch, tmp_path):
    """This process as an owner whose startup has not run yet; yields its role state."""
    monkeypatch.setattr(server.config, "data_dir", tmp_path)
    monkeypatch.setattr(server.config, "rate_limit_enabled", False)
    monkeypatch.setenv("KNOWLEDGE_RAG_WATCHER_DISABLED", "1")
    monkeypatch.setattr(server, "_orchestrator", None)
    monkeypatch.setattr(server, "_startup_thread", None)
    monkeypatch.setattr(_FakeIndex, "built_by", [])
    state = _writer_role.RoleState(ready=_ReadyRecorder())
    monkeypatch.setattr(_writer_role, "STATE", state)
    return state


def _search_status():
    return json.loads(server.search_knowledge(**SEARCH))["status"]


def _health_payload():
    """GET /health the way the HTTP transport serves it, cache statistics included."""
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.disconnect"}

    async def downstream(scope, receive, send):
        raise AssertionError("/health must not reach the MCP app")

    probe = HealthMiddleware(downstream, server.get_orchestrator)
    scope = {"type": "http", "path": "/health", "method": "GET", "headers": []}
    asyncio.new_event_loop().run_until_complete(probe(scope, receive, send))
    return json.loads(sent[-1]["body"])


def _start_and_join():
    server._start_writer_startup(lambda: False)
    server._startup_thread.join(WAIT_SECONDS)
    assert not server._startup_thread.is_alive()


def test_given_the_index_still_opening_when_tools_and_probes_ask_then_they_answer_repairing_and_never_open_it(
    owner, monkeypatch
):
    opening, may_open = threading.Event(), threading.Event()

    class SlowToOpen(_FakeIndex):
        def __init__(self):
            opening.set()
            assert may_open.wait(WAIT_SECONDS)
            super().__init__()

    monkeypatch.setattr(server, "KnowledgeOrchestrator", SlowToOpen)
    server._start_writer_startup(lambda: False)
    assert opening.wait(WAIT_SECONDS)

    health = _health_payload()
    during = (_search_status(), health["index"], health["cache"])
    may_open.set()
    server._startup_thread.join(WAIT_SECONDS)

    assert (during, _search_status(), _FakeIndex.built_by, owner.ready.published) == (
        ("repairing", "repairing", None),
        "success",
        [STARTUP_THREAD],
        1,
    )


def test_given_an_embedding_migration_running_when_tools_ask_then_they_answer_repairing_until_it_ends(
    owner, monkeypatch
):
    migrating, may_finish = threading.Event(), threading.Event()

    class Migrating(_FakeIndex):
        def _check_dimension_mismatch(self):
            return not may_finish.is_set()

        def nuclear_rebuild(self):
            migrating.set()
            assert may_finish.wait(WAIT_SECONDS)
            return REBUILT

    monkeypatch.setattr(server, "KnowledgeOrchestrator", Migrating)
    server._start_writer_startup(lambda: False)
    assert migrating.wait(WAIT_SECONDS)

    during = (_search_status(), server._orchestrator is not None, _writer_role.index_status())
    may_finish.set()
    server._startup_thread.join(WAIT_SECONDS)

    assert (during, _search_status(), _FakeIndex.built_by) == (
        ("repairing", True, "repairing"),
        "success",
        [STARTUP_THREAD],
    )


def test_given_the_index_fails_to_open_when_the_startup_ends_then_tools_answer_error_and_ready_never_came(
    owner, monkeypatch
):
    class Unopenable(_FakeIndex):
        def __init__(self):
            raise RuntimeError("chroma refused to open")

    monkeypatch.setattr(server, "KnowledgeOrchestrator", Unopenable)

    _start_and_join()

    assert (_search_status(), _writer_role.index_status(), owner.ready.published) == ("error", "error", 0)


def test_given_the_embedding_migration_fails_when_the_startup_ends_then_tools_answer_error_and_ready_never_came(
    owner, monkeypatch
):
    class FailingMigration(_FakeIndex):
        def _check_dimension_mismatch(self):
            return True

        def nuclear_rebuild(self):
            raise RuntimeError("rebuild failed")

        def index_all(self, force=False):
            raise RuntimeError("fallback index failed")

    monkeypatch.setattr(server, "KnowledgeOrchestrator", FailingMigration)

    _start_and_join()

    assert (_search_status(), _writer_role.index_status(), owner.ready.published) == ("error", "error", 0)


def _real_preflight_over(monkeypatch, tmp_path, results, events):
    """The real preflight, its probes answering ``results`` in turn; each probe and the opening go to ``events``."""
    (tmp_path / "chroma_db").mkdir()
    monkeypatch.setattr(server.config, "chroma_dir", tmp_path / "chroma_db")
    monkeypatch.setattr(preflight, "PROBE_RETRY_DELAY_SECONDS", 0)

    def probe(target, timeout_seconds=preflight.PROBE_TIMEOUT_SECONDS):
        result = results.pop(0) if len(results) > 1 else results[0]
        events.append("busy" if result is BUSY else "validated")
        return result(target)

    class RecordedIndex(_FakeIndex):
        def __init__(self):
            events.append("open")
            super().__init__()

    monkeypatch.setattr(preflight, "_probe_chroma", probe)
    monkeypatch.setattr(server, "KnowledgeOrchestrator", RecordedIndex)


def _start_real_preflight_and_join():
    server._start_writer_startup(preflight._probe_and_repair)
    server._startup_thread.join(WAIT_SECONDS)
    assert not server._startup_thread.is_alive()


def test_given_the_index_busy_then_free_when_the_owner_starts_then_it_opens_only_after_a_probe_validated_it(
    owner, monkeypatch, tmp_path
):
    events = []
    _real_preflight_over(monkeypatch, tmp_path, [BUSY, BUSY, HEALTHY], events)

    _start_real_preflight_and_join()

    assert (events, _search_status(), owner.ready.published) == (["busy", "busy", "validated", "open"], "success", 1)


def test_given_the_index_busy_until_the_limit_when_the_owner_starts_then_it_never_opens_nor_quarantines(
    owner, monkeypatch, tmp_path
):
    events = []
    _real_preflight_over(monkeypatch, tmp_path, [BUSY], events)
    monkeypatch.setattr(_index_access, "WAIT_SECONDS", 0)

    _start_real_preflight_and_join()

    answer = json.loads(server.search_knowledge(**SEARCH))
    index_left = ((tmp_path / "chroma_db").exists(), (tmp_path / "backups").exists(), _index_access.is_held())
    assert (events, answer["status"], "restart this client" in answer["message"], index_left) == (
        ["busy"],
        "error",
        True,
        (True, False, False),
    )
