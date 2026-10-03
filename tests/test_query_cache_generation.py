"""In-flight reads must not publish stale results after a completed mutation."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from mcp_server import server
from mcp_server.fts5_index import Fts5LexicalIndex


@pytest.mark.parametrize(("search_method", "hybrid_alpha"), [("hybrid", 1.0), ("hybrid", 0.5), ("fts5", 1.0)])
def test_completed_update_cannot_be_hidden_by_an_earlier_query_cache_write(
    isolated_orchestrator, monkeypatch, search_method, hybrid_alpha
):
    orchestrator = isolated_orchestrator
    monkeypatch.setattr(server.config, "reranker_enabled", False)
    monkeypatch.setattr(server.config, "fts5_enabled", False)
    source = server.config.documents_dir / "document.txt"
    source.write_text("old content for the query", encoding="utf-8")
    assert orchestrator.index_all()["errors"] == 0
    if search_method == "fts5":
        orchestrator.fts5_index = Fts5LexicalIndex(
            server.config.data_dir / "race.db", server.config.data_dir / "race.state"
        )
        orchestrator.fts5_index._migration_worker(orchestrator._iter_chroma_chunks_for_fts5, 1, 0, None)
        assert orchestrator.fts5_index.is_ready()
        monkeypatch.setattr(server.config, "fts5_enabled", True)
        monkeypatch.setattr(server.config, "fts5_rerank_enabled", False)
    before_put, release_put = threading.Event(), threading.Event()
    put = orchestrator.query_cache.put

    def delayed_put(*args, **kwargs):
        if not before_put.is_set():
            before_put.set()
            assert release_put.wait(timeout=5)
        return put(*args, **kwargs)

    monkeypatch.setattr(orchestrator.query_cache, "put", delayed_put)
    with ThreadPoolExecutor(max_workers=1) as executor:
        reader = executor.submit(
            orchestrator.query, "content", 1, hybrid_alpha=hybrid_alpha, search_method=search_method
        )
        try:
            assert before_put.wait(timeout=5)
            update = orchestrator.update_document_content("document.txt", "new content after a successful update")
            assert "error" not in update
            # A newer query can publish first; the delayed result must not replace it.
            fresh = orchestrator.query("content", 1, hybrid_alpha=hybrid_alpha, search_method=search_method)
            assert fresh[0]["content"] == "new content after a successful update"
        finally:
            release_put.set()
        assert reader.result(timeout=5)[0]["content"] == "old content for the query"

    monkeypatch.setattr(orchestrator.query_cache, "put", put)
    assert orchestrator.collection.get(include=["documents"])["documents"] == ["new content after a successful update"]
    assert orchestrator.query("content", 1, hybrid_alpha=hybrid_alpha, search_method=search_method) == fresh
    assert orchestrator.query_cache.stats()["hits"] == 1


def test_invalidation_during_snapshot_copy_cannot_evict_a_fresh_entry(monkeypatch):
    cache = server.QueryCache(max_size=1)
    generation = cache.generation
    old_result = [{"content": "old"}]
    before_copy, release_copy = threading.Event(), threading.Event()
    clone = cache._clone_result

    def delayed_clone(result):
        if result is old_result:
            before_copy.set()
            assert release_copy.wait(timeout=5)
        return clone(result)

    monkeypatch.setattr(cache, "_clone_result", delayed_clone)
    with ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(cache.put, "old", 1, None, 1.0, old_result, expected_generation=generation)
        try:
            assert before_copy.wait(timeout=5)
            cache.invalidate()
            cache.put("fresh", 1, None, 1.0, [{"content": "new"}], expected_generation=cache.generation)
            stats = cache.stats()
        finally:
            release_copy.set()
        writer.result(timeout=5)
    assert cache.stats() == stats
    assert cache.get("fresh", 1, None, 1.0) == [{"content": "new"}]
    assert cache.get("old", 1, None, 1.0) is None


def test_query_captures_generation_before_cache_lookup(isolated_orchestrator, monkeypatch):
    orchestrator = isolated_orchestrator
    monkeypatch.setattr(server.config, "reranker_enabled", False)
    orchestrator.collection.add(
        ids=["document_0"], documents=["current document"], embeddings=[[1.0, 1.0]], metadatas=[{"source": "one.txt"}]
    )
    get = orchestrator.query_cache.get

    def invalidate_after_lookup(*args, **kwargs):
        cached = get(*args, **kwargs)
        orchestrator.query_cache.invalidate()
        return cached

    monkeypatch.setattr(orchestrator.query_cache, "get", invalidate_after_lookup)
    assert orchestrator.query("document", 1, hybrid_alpha=1.0)[0]["content"] == "current document"
    assert orchestrator.query_cache.stats()["size"] == 0


def test_cached_read_overlapping_invalidation_does_not_restore_its_snapshot(monkeypatch):
    cache = server.QueryCache()
    cache.put("query", 1, None, 1.0, [{"content": "old"}])
    before_copy, release_copy = threading.Event(), threading.Event()
    clone = cache._clone_result

    def delayed_clone(result):
        if result[0]["content"] == "old":
            before_copy.set()
            assert release_copy.wait(timeout=5)
        return clone(result)

    monkeypatch.setattr(cache, "_clone_result", delayed_clone)
    with ThreadPoolExecutor(max_workers=1) as executor:
        reader = executor.submit(cache.get, "query", 1, None, 1.0)
        try:
            assert before_copy.wait(timeout=5)
            cache.invalidate()
            # The positional get/put API still accepts calls without a generation.
            cache.put("query", 1, None, 1.0, [{"content": "new"}], "auto")
        finally:
            release_copy.set()
        assert reader.result(timeout=5) == [{"content": "old"}]
    assert cache.get("query", 1, None, 1.0) == [{"content": "new"}]
