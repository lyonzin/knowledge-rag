"""Atomic staging, bounded background work, and watcher wakeup regression tests."""

import threading
from concurrent.futures import Future
from unittest.mock import Mock

import pytest

from mcp_server import server


def seed(orch):
    path = server.config.documents_dir / "one.txt"
    path.write_text("production body", encoding="utf-8")
    assert orch.index_all()["errors"] == 0
    orch._ensure_bm25_index()
    return path


def test_populate_leaves_all_production_state_visible(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    seed(orch)
    original = (orch.collection, orch.bm25_index, orch._indexed_docs, orch._source_to_docid)
    durable = orch._metadata_file.read_bytes()
    staging = orch._create_staging_collection(1)
    entered, release = threading.Event(), threading.Event()
    parse = orch.parser.parse_file

    def blocked_parse(path):
        entered.set()
        assert release.wait(timeout=5)
        return parse(path)

    monkeypatch.setattr(orch.parser, "parse_file", blocked_parse)
    error = []

    def populate():
        try:
            orch._populate_staging(staging)
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=populate)
    thread.start()
    try:
        assert entered.wait(timeout=5)
        assert (orch.collection, orch.bm25_index, orch._indexed_docs, orch._source_to_docid) == original
        assert orch.bm25_index.search("production")
        assert orch._metadata_file.read_bytes() == durable
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive() and not error
    assert orch._metadata_file.read_bytes() == durable
    assert orch.bm25_index is original[1]


@pytest.mark.parametrize("failure_point", ["_save_metadata", "_rebuild_bm25_post_swap"])
def test_failed_swap_commit_restores_real_collection(isolated_orchestrator, monkeypatch, failure_point):
    orch = isolated_orchestrator
    path = seed(orch)
    before = orch.collection.get(include=["documents"])
    metadata = orch._metadata_file.read_bytes()
    path.write_text("replacement body with changed vectors", encoding="utf-8")
    monkeypatch.setattr(orch, failure_point, Mock(side_effect=OSError("post swap failure")))
    with pytest.raises(OSError, match="post swap failure"):
        orch.nuclear_rebuild()
    assert orch.collection.get(include=["documents"]) == before
    assert orch.chroma_client.get_collection(server.config.collection_name).get(include=["documents"]) == before
    assert orch._metadata_file.read_bytes() == metadata


def test_successful_swap_publishes_metadata_before_deleting_old(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    path = seed(orch)
    path.write_text("fresh replacement text", encoding="utf-8")
    delete = orch.chroma_client.delete_collection

    def observed_delete(name):
        if "__old_" in name:
            assert orch._load_metadata() == orch._indexed_docs
            assert orch.collection.get(include=["documents"])["documents"] == ["fresh replacement text"]
        return delete(name)

    monkeypatch.setattr(orch.chroma_client, "delete_collection", observed_delete)
    stats = orch.nuclear_rebuild()
    assert stats["indexed"] == 1
    assert not any("__old_" in collection.name for collection in orch.chroma_client.list_collections())


def test_parallel_batches_bound_unconsumed_futures(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    outstanding = 0
    peak = 0

    class ObservedFuture(Future):
        def result(self, *args, **kwargs):
            nonlocal outstanding
            outstanding -= 1
            return super().result(*args, **kwargs)

    class Executor:
        def __init__(self, max_workers):
            assert max_workers == 3

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, *args, **kwargs):
            nonlocal outstanding, peak
            outstanding += 1
            peak = max(peak, outstanding)
            future = ObservedFuture()
            future.set_result(None)
            return future

    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", Executor)
    orch._add_chunks_parallel(list(range(100)), ["text"] * 100, [{}] * 100, 1, 3)
    assert peak == 3
    assert outstanding == 0


@pytest.mark.parametrize("busy", [False, True])
def test_watcher_rearms_events_during_active_callback(monkeypatch, busy):
    orchestrator = Mock()
    watcher = server.DocumentWatcher(lambda: orchestrator)
    timer = Mock()
    monkeypatch.setattr(server.threading, "Timer", Mock(return_value=timer))
    watcher._pending_paths.add("before.txt")

    def index(**kwargs):
        watcher._pending_paths.add("during.txt")
        return {"skipped_reason": "reindex_already_running"} if busy else {}

    orchestrator.index_all.side_effect = index
    watcher._do_reindex()
    assert "during.txt" in watcher._pending_paths
    assert ("before.txt" in watcher._pending_paths) is busy
    timer.start.assert_called_once()


def test_public_smart_reindex_routes_force_true(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    reindex = Mock(return_value={})
    monkeypatch.setattr(orch, "reindex_all", reindex)

    class InlineThread:
        def __init__(self, target, args, daemon):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    orch.start_reindex_background("smart_reindex")
    reindex.assert_called_once_with(resume_state=None, force=True)
