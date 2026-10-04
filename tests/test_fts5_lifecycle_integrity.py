"""Migration completeness, cancellation, and orchestrator source failures."""

import threading
from unittest.mock import Mock

import pytest

from mcp_server import server
from mcp_server.fts5_index import Fts5LexicalIndex


@pytest.fixture
def index(tmp_path):
    value = Fts5LexicalIndex(tmp_path / "fts5.db", tmp_path / "state.json")
    yield value
    value.close()


def test_short_source_cannot_publish_complete(index):
    thread = index.start_migration_background(lambda: iter([("a", "body", "a.txt", "general")]), 2)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert index.state.read()["status"] == "failed"
    assert index.is_ready() is False


def test_second_chroma_page_failure_marks_failed(index, isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    collection = Mock()
    collection.count.return_value = 501
    collection.get.side_effect = [
        {"ids": [str(i) for i in range(500)], "documents": ["body"] * 500, "metadatas": [{}] * 500},
        RuntimeError("page unavailable"),
    ]
    monkeypatch.setattr(orch, "collection", collection)
    thread = index.start_migration_background(orch._iter_chroma_chunks_for_fts5, 501)
    thread.join(timeout=5)
    assert not thread.is_alive()
    payload = index.state.read()
    assert payload["status"] == "failed"
    assert "page unavailable" in payload["error"]
    assert index.is_ready() is False


def test_close_cancels_worker_waiting_for_mutation_lock(index, isolated_orchestrator):
    orch = isolated_orchestrator
    with orch._index_lock:
        thread = index.start_migration_background(lambda: orch._locked_fts5_source(index), 1)
        index.close()
        assert not thread.is_alive()
    assert index._conn is None
    assert index.state.read()["status"] != "complete"


def test_overlapping_migrations_share_one_worker(index):
    entered, release = threading.Event(), threading.Event()

    def rows():
        entered.set()
        assert release.wait(timeout=5)
        yield ("one", "body", "one.txt", "general")

    first = index.start_migration_background(rows, 1)
    try:
        assert entered.wait(timeout=5)
        assert index.start_migration_background(rows, 1) is first
    finally:
        release.set()
        first.join(timeout=5)
    assert index.count() == 1


@pytest.mark.parametrize("count", ["garbage", -1, {}, True])
def test_bad_resume_marker_replays_safely(index, isolated_orchestrator, monkeypatch, count):
    orch = isolated_orchestrator
    monkeypatch.setattr(server.config, "fts5_enabled", True)
    orch.fts5_index = index
    index.state.write({"status": "in_progress", "docs_indexed": count})
    orch._maybe_start_fts5_migration()
    assert index.state.read()["status"] == "complete"
    assert index.count() == orch.collection.count() == 0


def test_same_size_orphan_index_is_cleared_on_forced_reset(index, isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    monkeypatch.setattr(server.config, "fts5_enabled", True)
    orch.fts5_index = index
    index.add_document("orphan", "stale text", "old.txt", "general")
    assert orch._fts5_marker_matches_reality() is False
    orch._maybe_start_fts5_migration()
    assert index.count() == 0 and index.is_ready()
