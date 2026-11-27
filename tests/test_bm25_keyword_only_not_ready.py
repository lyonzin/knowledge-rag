"""A keyword-only search must not answer "no results" from a BM25 never built.

Keyword-only search (``hybrid_alpha=0``) has no channel but BM25: with no
published generation it answered with an empty list, reported as ``no_results``.
It now waits for a build already running, serves a published generation while
a rebuild runs, and otherwise raises ``Bm25NotReadyError`` with the cause as it
stands. Uses ``isolated_orchestrator``: no real model is loaded.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager, suppress
from unittest.mock import MagicMock

import pytest

from mcp_server import server as srv

_TIMEOUT_S = 10
# Longer than _TIMEOUT_S, so a held lock or build never ends by itself mid-test.
_HOLD_LIMIT_S = 30


def _cold_index(orch, monkeypatch) -> None:
    """Three documents and no BM25 generation yet, as right after a start."""
    monkeypatch.setattr(srv.config, "reranker_enabled", False)
    for name, text in (("a1.md", "initial corpus content"), ("a2.md", "other filler"), ("a3.md", "third passage")):
        orch.add_document_from_content(content=text, filepath=name, category="t")


def _published_then_rebuilding(orch, monkeypatch) -> None:
    """A published generation whose rebuild started, as reindex_all leaves it (#259)."""
    _cold_index(orch, monkeypatch)
    assert _keyword_only_filenames(orch) == ["a1.md"], "the first search publishes a generation"
    orch._bm25_initialized = False


@contextmanager
def _held_on_another_thread(*locks):
    """Hold ``locks`` on another thread, as a reindex or a document write holds the index."""
    held, release = threading.Event(), threading.Event()

    def holder():
        for lock in locks:
            lock.acquire()
        held.set()
        release.wait(_HOLD_LIMIT_S)
        for lock in reversed(locks):
            lock.release()

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    try:
        assert held.wait(_TIMEOUT_S), "the other thread must hold the locks"
        yield
    finally:
        release.set()
        thread.join(_TIMEOUT_S)


class _ContentionSignalingLock:
    """Wrap a lock; ``contended`` is set when a thread tries to enter while it is held."""

    def __init__(self, lock) -> None:
        self._lock = lock
        self.contended = threading.Event()

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            self.contended.set()
            self._lock.acquire()
        return self

    def __exit__(self, *exc_info) -> None:
        self._lock.release()


def _keyword_only_filenames(orch, text: str = "initial") -> list[str]:
    return [row["filename"] for row in orch.query(text, max_results=3, hybrid_alpha=0.0)]


def _not_ready_message(orch) -> str:
    with pytest.raises(srv.Bm25NotReadyError) as caught:
        orch.query("initial", max_results=3, hybrid_alpha=0.0)
    return str(caught.value)


def _keyword_only_tool_response(orch, monkeypatch) -> dict:
    monkeypatch.setattr(srv, "get_orchestrator", lambda: orch)
    return json.loads(srv.search_knowledge("initial", hybrid_alpha=0.0))


def test_given_a_cold_index_and_a_writer_holding_the_index_lock_when_a_keyword_only_search_arrives_then_it_reports_the_keyword_index_is_not_ready(
    isolated_orchestrator, monkeypatch
):
    _cold_index(isolated_orchestrator, monkeypatch)

    with _held_on_another_thread(isolated_orchestrator._index_lock):
        message = _not_ready_message(isolated_orchestrator)

    assert "another operation holds the index" in message, message


def test_given_a_cold_index_and_a_writer_holding_the_index_lock_when_search_knowledge_runs_keyword_only_then_the_error_says_the_keyword_index_is_not_built(
    isolated_orchestrator, monkeypatch
):
    _cold_index(isolated_orchestrator, monkeypatch)

    with _held_on_another_thread(isolated_orchestrator._index_lock):
        response = _keyword_only_tool_response(isolated_orchestrator, monkeypatch)

    assert "keyword index" in response.get("error", ""), response


def test_given_a_cold_index_and_a_writer_holding_the_index_lock_when_search_knowledge_runs_keyword_only_then_the_suggestion_points_to_hybrid_alpha(
    isolated_orchestrator, monkeypatch
):
    _cold_index(isolated_orchestrator, monkeypatch)

    with _held_on_another_thread(isolated_orchestrator._index_lock):
        response = _keyword_only_tool_response(isolated_orchestrator, monkeypatch)

    assert "hybrid_alpha" in response.get("suggestion", ""), response


def test_given_a_cold_index_and_a_collection_that_cannot_be_counted_when_search_knowledge_runs_keyword_only_then_it_returns_a_structured_error(
    isolated_orchestrator, monkeypatch
):
    _cold_index(isolated_orchestrator, monkeypatch)
    monkeypatch.setattr(
        isolated_orchestrator.collection, "count", MagicMock(side_effect=RuntimeError("database is locked"))
    )

    assert _keyword_only_tool_response(isolated_orchestrator, monkeypatch)["status"] == "error"


def test_given_a_cold_index_and_a_writer_holding_the_index_lock_when_a_hybrid_search_arrives_then_it_answers_from_the_semantic_channel(
    isolated_orchestrator, monkeypatch
):
    _cold_index(isolated_orchestrator, monkeypatch)

    with _held_on_another_thread(isolated_orchestrator._index_lock):
        found = isolated_orchestrator.query("initial", max_results=3, hybrid_alpha=0.5)

    assert found and all(row["bm25_rank"] is None for row in found), found


def test_given_an_empty_collection_when_a_keyword_only_search_arrives_then_it_returns_no_results(
    isolated_orchestrator, monkeypatch
):
    monkeypatch.setattr(srv.config, "reranker_enabled", False)

    response = _keyword_only_tool_response(isolated_orchestrator, monkeypatch)

    assert response["status"] == "no_results", response


def test_given_a_writer_held_the_index_lock_when_it_lets_go_then_the_next_keyword_only_search_builds_and_finds_the_document(
    isolated_orchestrator, monkeypatch
):
    _cold_index(isolated_orchestrator, monkeypatch)
    with _held_on_another_thread(isolated_orchestrator._index_lock), suppress(srv.Bm25NotReadyError):
        isolated_orchestrator.query("initial", max_results=3, hybrid_alpha=0.0)

    assert _keyword_only_filenames(isolated_orchestrator) == ["a1.md"]


def test_given_a_published_generation_and_a_reindex_behind_the_index_lock_when_a_keyword_only_search_arrives_then_it_serves_the_old_generation(
    isolated_orchestrator, monkeypatch
):
    _published_then_rebuilding(isolated_orchestrator, monkeypatch)

    with _held_on_another_thread(isolated_orchestrator._index_lock):
        found = _keyword_only_filenames(isolated_orchestrator, "corpus")

    assert found == ["a1.md"]


def test_given_a_rebuild_holding_the_index_and_build_locks_when_a_keyword_only_search_arrives_then_it_answers_without_waiting_for_the_build(
    isolated_orchestrator, monkeypatch
):
    orch = isolated_orchestrator
    _published_then_rebuilding(orch, monkeypatch)
    found: list = []
    search = threading.Thread(target=lambda: found.extend(_keyword_only_filenames(orch, "corpus")), daemon=True)

    with _held_on_another_thread(orch._index_lock, orch._bm25_build_lock):
        search.start()
        search.join(_TIMEOUT_S)

        assert (search.is_alive(), found) == (False, ["a1.md"]), "it must not wait for the rebuild"


def test_given_another_search_building_bm25_inline_when_a_keyword_only_search_arrives_then_it_waits_and_finds_the_document(
    isolated_orchestrator, monkeypatch
):
    orch = isolated_orchestrator
    _cold_index(orch, monkeypatch)
    build_lock = _ContentionSignalingLock(orch._bm25_build_lock)
    build, gate, building = orch.bm25_index.build_index, threading.Event(), threading.Event()

    def held_build():
        building.set()
        gate.wait(_HOLD_LIMIT_S)
        build()

    monkeypatch.setattr(orch, "_bm25_build_lock", build_lock)
    monkeypatch.setattr(orch.bm25_index, "build_index", held_build)
    hybrid = threading.Thread(target=orch.query, args=("other",), kwargs={"max_results": 3}, daemon=True)
    found: list = []
    keyword = threading.Thread(target=lambda: found.extend(_keyword_only_filenames(orch)), daemon=True)
    try:
        hybrid.start()
        assert building.wait(_TIMEOUT_S), "the hybrid search must be building BM25 inline"
        keyword.start()
        assert build_lock.contended.wait(_TIMEOUT_S), "the keyword-only search must wait for that build"
    finally:
        gate.set()
    keyword.join(_TIMEOUT_S)
    hybrid.join(_TIMEOUT_S)

    assert found == ["a1.md"]


def test_given_the_bm25_build_keeps_failing_when_a_keyword_only_search_arrives_then_the_error_carries_the_failure(
    isolated_orchestrator, monkeypatch
):
    _cold_index(isolated_orchestrator, monkeypatch)
    monkeypatch.setattr(
        isolated_orchestrator.bm25_index, "build_index", MagicMock(side_effect=MemoryError("injected OOM"))
    )
    isolated_orchestrator._ensure_bm25_index()  # the last build failed

    assert "its last build failed (MemoryError: injected OOM)" in _not_ready_message(isolated_orchestrator)


def test_given_a_build_failing_with_an_exception_without_a_message_when_a_keyword_only_search_arrives_then_the_error_names_its_type(
    isolated_orchestrator, monkeypatch
):
    _cold_index(isolated_orchestrator, monkeypatch)
    monkeypatch.setattr(isolated_orchestrator.bm25_index, "build_index", MagicMock(side_effect=MemoryError()))

    assert "MemoryError" in _not_ready_message(isolated_orchestrator)


def test_given_an_earlier_failed_build_and_a_writer_holding_the_index_lock_when_a_keyword_only_search_arrives_then_the_error_names_both(
    isolated_orchestrator, monkeypatch
):
    _cold_index(isolated_orchestrator, monkeypatch)
    monkeypatch.setattr(
        isolated_orchestrator.bm25_index, "build_index", MagicMock(side_effect=MemoryError("injected OOM"))
    )
    isolated_orchestrator._ensure_bm25_index()  # an earlier build failed

    with _held_on_another_thread(isolated_orchestrator._index_lock):
        message = _not_ready_message(isolated_orchestrator)

    assert "holds the index" in message and "injected OOM" in message, message


def test_given_a_build_that_published_nothing_and_no_writer_when_a_keyword_only_search_arrives_then_the_error_says_nothing_is_published_yet(
    isolated_orchestrator, monkeypatch
):
    _cold_index(isolated_orchestrator, monkeypatch)
    monkeypatch.setattr(isolated_orchestrator, "_rebuild_bm25_from_collection", lambda refresh=False: None)

    message = _not_ready_message(isolated_orchestrator)

    assert "has not published a generation yet" in message and "another operation" not in message, message


def test_given_a_build_publishing_while_the_collection_is_counted_when_the_keyword_index_is_checked_then_it_does_not_raise(
    isolated_orchestrator, monkeypatch
):
    orch = isolated_orchestrator
    _cold_index(orch, monkeypatch)
    count, rebuild, published = orch.collection.count, orch._rebuild_bm25_from_collection, []

    def count_while_a_build_publishes():
        if not published:
            published.append(True)
            rebuild()  # its own count comes back here and passes through
        return count()

    monkeypatch.setattr(orch.collection, "count", count_while_a_build_publishes)

    orch._raise_if_keyword_index_missing()

    assert orch.bm25_index.has_published_generation, "the count let a build publish"


def test_given_a_failure_recorded_with_an_empty_message_when_the_reason_is_described_then_it_still_says_the_last_build_failed():
    message = srv._describe_missing_keyword_index(is_index_busy=False, last_error="")

    assert "its last build failed" in message, message
