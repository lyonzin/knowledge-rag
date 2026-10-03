"""Published BM25 generations remain coherent across clear/rebuild operations."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from mcp_server.server import BM25Index, KnowledgeOrchestrator


def _index():
    index = BM25Index()
    index.add_documents(["old-a", "old-b", "old-c"], ["alpha report", "beta review", "gamma notes"])
    index.build_index()
    return index


def test_inflight_query_keeps_ids_and_scores_after_clear_rebuild(monkeypatch):
    index = _index()
    expected = index.search("alpha")
    assert expected and expected[0][0] == "old-a"
    reading = threading.Event()
    proceed = threading.Event()
    original_expand = index.expand_query

    def paused_expand(query):
        reading.set()
        assert proceed.wait(5)
        return original_expand(query)

    monkeypatch.setattr(index, "expand_query", paused_expand)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(index.search, "alpha")
        assert reading.wait(5)
        index.clear()
        index.add_documents(["new-id"], ["unrelated text"])
        index.build_index()
        proceed.set()
        assert future.result(5) == expected
    monkeypatch.setattr(index, "expand_query", original_expand)
    assert index.search("alpha") == []


def test_queries_keep_previous_generation_until_build_publishes(monkeypatch):
    index = _index()
    expected = index.search("alpha")
    index.add_documents(["new-a", "new-b"], ["alpha extra report", "delta unrelated"])
    building = threading.Event()
    proceed = threading.Event()
    original = index._build_index_locked

    def paused_build():
        building.set()
        assert proceed.wait(5)
        original()

    monkeypatch.setattr(index, "_build_index_locked", paused_build)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(index.build_index)
        assert building.wait(5)
        assert index.search("alpha") == expected
        proceed.set()
        future.result(5)
    assert {chunk_id for chunk_id, _ in index.search("alpha")} == {"old-a", "new-a"}


def test_vocabulary_is_shared_within_index_and_released_on_clear():
    index = BM25Index()
    index.add_documents(["a", "b"], ["performance migration " * 50, "performance migration " * 50])
    first, second = index._tokenized_corpus
    assert first[0] is first[2] is second[0]
    assert len(index._token_pool) == 2
    index.clear()
    assert index._token_pool == {}
    assert index._tokenized_corpus == []
    assert index._snapshot is None


def test_failed_reload_preserves_the_published_generation(monkeypatch):
    index = _index()
    expected = index.search("alpha")
    index._reset_corpus()
    index.add_documents(["replacement"], ["unrelated text"])

    def failure():
        raise MemoryError("injected build failure")

    monkeypatch.setattr(index, "_build_index_locked", failure)
    with pytest.raises(MemoryError):
        index.build_index()
    assert index.search("alpha") == expected


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_limit_returns_no_hits(limit):
    assert _index().search("alpha", top_k=limit) == []


@pytest.mark.parametrize("has_published_index", [False, True])
def test_lazy_build_never_waits_behind_active_ingestion(has_published_index):
    orchestrator = object.__new__(KnowledgeOrchestrator)
    orchestrator._index_lock = threading.RLock()
    orchestrator._bm25_build_lock = threading.Lock()
    orchestrator._bm25_initialized = False
    orchestrator.collection = Mock()
    orchestrator.bm25_index = _index() if has_published_index else BM25Index()
    before = orchestrator.bm25_index.search("alpha")
    with ThreadPoolExecutor(max_workers=1) as pool:
        with orchestrator._index_lock:
            # The writer deliberately remains active until this call returns.
            pool.submit(orchestrator._ensure_bm25_index).result(timeout=1)
    assert orchestrator.bm25_index.search("alpha") == before
    assert not orchestrator._bm25_initialized
    orchestrator.collection.count.assert_not_called()


def test_lazy_reload_can_run_reentrantly_after_a_write():
    orchestrator = object.__new__(KnowledgeOrchestrator)
    orchestrator._index_lock = threading.RLock()
    orchestrator._bm25_build_lock = threading.Lock()
    orchestrator._bm25_initialized = False
    orchestrator.collection = Mock()
    orchestrator.collection.count.return_value = 1
    orchestrator.collection.get.return_value = {"ids": ["new"], "documents": ["alpha"]}
    orchestrator.bm25_index = _index()
    with orchestrator._index_lock:
        orchestrator._ensure_bm25_index()
    assert orchestrator._bm25_initialized
    assert orchestrator.bm25_index.corpus_ids == ["new"]
