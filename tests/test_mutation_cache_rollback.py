"""Queries from intermediate real Chroma mutations must not outlive rollback."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from mcp_server import server


def query(orch):
    return orch.query("synthetic orchard", max_results=1, hybrid_alpha=1.0, search_method="hybrid")


@pytest.fixture
def mutation_window(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    monkeypatch.setattr(server.config, "reranker_enabled", False)
    orch.add_document_from_content("Original synthetic orchard", "one.txt", "general")
    entered, proceed, caching, finish_query = (threading.Event() for _ in range(4))
    put = orch.query_cache.put

    def fail_save():
        entered.set()
        assert proceed.wait(5), "mutation was not released"
        raise OSError("metadata unavailable")

    def delayed_put(*args, **kwargs):
        caching.set()
        assert finish_query.wait(5), "query was not released"
        return put(*args, **kwargs)

    monkeypatch.setattr(orch, "_save_metadata", fail_save)
    monkeypatch.setattr(orch.query_cache, "put", delayed_put)
    return orch, (entered, proceed, caching, finish_query)


def run_concurrent_query(orch, mutation, events, late_cache):
    entered, proceed, caching, finish_query = events
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(mutation)
        try:
            assert entered.wait(5)
            reader = pool.submit(query, orch)
            assert caching.wait(5)
            if not late_cache:
                finish_query.set()
                during = reader.result(timeout=5)
            proceed.set()
            with pytest.raises(OSError, match="unavailable"):
                writer.result(timeout=5)
        finally:
            proceed.set()
            finish_query.set()
        if late_cache:
            during = reader.result(timeout=5)
    return during


def fail_after_restore(orch, monkeypatch):
    restore = orch._restore_document_chunks

    def restore_then_raise(snapshot):
        restore(snapshot)
        raise OSError("rollback persistence unavailable")

    monkeypatch.setattr(orch, "_restore_document_chunks", restore_then_raise)


@pytest.mark.parametrize("late_cache", [False, True])
@pytest.mark.parametrize("rollback_fails", [False, True])
def test_failed_delete_drops_intermediate_cache(mutation_window, monkeypatch, late_cache, rollback_fails):
    orch, events = mutation_window
    if rollback_fails:
        fail_after_restore(orch, monkeypatch)
    generation = orch.query_cache.generation
    during = run_concurrent_query(orch, lambda: orch.remove_document_by_path("one.txt"), events, late_cache)
    assert during == []
    assert orch.collection.count() == 1
    assert orch.query_cache.generation > generation
    after = query(orch)
    assert len(after) == 1 and "Original synthetic orchard" in after[0]["content"]


@pytest.mark.parametrize("operation", ["add", "update"])
def test_failed_content_rollback_drops_intermediate_cache(mutation_window, monkeypatch, operation):
    orch, events = mutation_window
    fail_after_restore(orch, monkeypatch)

    def mutation():
        if operation == "add":
            return orch.add_document_from_content("Replacement synthetic orchard", "one.txt", "general")
        return orch.update_document_content("one.txt", "Replacement synthetic orchard")

    generation = orch.query_cache.generation
    during = run_concurrent_query(orch, mutation, events, late_cache=True)
    assert len(during) == 1 and "Replacement synthetic orchard" in during[0]["content"]
    assert orch.collection.count() == 1
    assert orch.query_cache.generation > generation
    after = query(orch)
    assert len(after) == 1 and "Original synthetic orchard" in after[0]["content"]
