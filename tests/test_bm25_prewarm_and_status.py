"""GH #316: loud early exit in BM25 rebuild + observable status in stats.

Three regressions guard the fix:
  1. ``_rebuild_bm25_from_collection`` logs when an empty page aborts the
     build and never returns silently. Collection shrinking mid-build
     publishes the partial corpus rather than leaving the orchestrator
     with ``_bm25_initialized=False`` and nothing in logs.
  2. ``get_stats()`` exposes a ``bm25`` dict with status/chunk_count/
     last_built_at/stale/error.
  3. ``_bm25_last_built_at`` is populated after a successful publish.

GH #321: the startup prewarm is a background build like the one a search
starts, so a keyword-only search arriving mid-build waits for it instead of
reading an empty BM25, while hybrid and semantic searches still never wait.
A prewarm that cannot start, or whose build fails, gives the lock back so the
next keyword-only search builds BM25 itself.

Uses the shared ``isolated_orchestrator`` fixture (mock embeddings +
EphemeralClient), so these tests never touch the real FastEmbed model.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from mcp_server import server as srv

# A search that does not wait for BM25 answers well within this window.
_ANSWER_WINDOW_S = 0.3
_TIMEOUT_S = 10
# Longer than _TIMEOUT_S, so a held build never ends by itself mid-test.
_HOLD_LIMIT_S = 30


def _seed_chunks(orch, ids_and_text):
    """Add synthetic chunks directly to the fixture collection and corpus."""
    ids = [cid for cid, _ in ids_and_text]
    docs = [doc for _, doc in ids_and_text]
    metadatas = [{"source": f"{cid}.md"} for cid in ids]
    orch.collection.add(ids=ids, documents=docs, metadatas=metadatas)


def test_rebuild_logs_when_empty_page_with_chunks_remaining(isolated_orchestrator, capsys):
    """count says N but a page returns no ids: must log and abandon loudly."""
    orch = isolated_orchestrator
    _seed_chunks(orch, [("c1", "alpha payload"), ("c2", "beta payload")])

    real_get = orch.collection.get
    call_counter = {"n": 0}

    def flaky_get(**kwargs):
        if kwargs.get("offset", 0) == 0:
            call_counter["n"] += 1
            if call_counter["n"] == 1:
                return {"ids": [], "documents": []}
        return real_get(**kwargs)

    with patch.object(orch.collection, "get", side_effect=flaky_get):
        orch._rebuild_bm25_from_collection()

    err = capsys.readouterr().err
    assert "empty page at offset 0" in err, f"early exit must be logged, got: {err!r}"
    assert orch._bm25_initialized is False
    assert orch._bm25_last_error and "offset 0" in orch._bm25_last_error


def test_rebuild_publishes_partial_when_collection_shrank(isolated_orchestrator, capsys):
    """Watcher deleted docs mid-build: publish what we already read, logged."""
    orch = isolated_orchestrator
    _seed_chunks(orch, [("c1", "alpha payload")])

    count_calls = {"n": 0}

    def shrinking_count():
        count_calls["n"] += 1
        return 501 if count_calls["n"] == 1 else 0

    real_get = orch.collection.get

    def empty_after_offset_500(**kwargs):
        if kwargs.get("offset", 0) >= 500:
            return {"ids": [], "documents": []}
        return real_get(**kwargs)

    with (
        patch.object(orch.collection, "count", side_effect=shrinking_count),
        patch.object(orch.collection, "get", side_effect=empty_after_offset_500),
    ):
        orch._rebuild_bm25_from_collection()

    err = capsys.readouterr().err
    assert "collection shrank" in err, f"shrink must be logged, got: {err!r}"
    assert orch._bm25_initialized is True
    assert orch._bm25_last_built_at is not None


def test_get_stats_exposes_bm25_status_on_empty(isolated_orchestrator):
    orch = isolated_orchestrator
    stats = orch.get_stats()
    assert "bm25" in stats
    bm25 = stats["bm25"]
    for key in ("status", "chunk_count", "last_built_at", "stale", "error"):
        assert key in bm25, f"missing bm25.{key}"
    assert bm25["status"] == "empty"
    assert bm25["chunk_count"] == 0
    assert bm25["last_built_at"] is None


def test_get_stats_bm25_ready_after_build(isolated_orchestrator):
    orch = isolated_orchestrator
    _seed_chunks(orch, [("c1", "hello world payload")])
    orch._ensure_bm25_index()
    bm25 = orch.get_stats()["bm25"]
    assert bm25["status"] == "ready"
    assert bm25["chunk_count"] >= 1
    assert bm25["last_built_at"] is not None
    assert bm25["last_built_at"] <= time.time()
    assert bm25["stale"] is False
    assert bm25["error"] is None


def _cold_index(orch, monkeypatch, is_large: bool = True) -> None:
    """Three documents and no BM25 generation yet, as right after a start."""
    monkeypatch.setattr(srv.config, "reranker_enabled", False)
    for name, text in (("a1.md", "initial corpus content"), ("a2.md", "other filler"), ("a3.md", "third passage")):
        orch.add_document_from_content(content=text, filepath=name, category="t")
    if is_large:
        monkeypatch.setattr(orch, "_BM25_INLINE_REFRESH_CHUNKS", orch.collection.count() - 1)


def _hold_builds(orch, monkeypatch):
    """Hold every BM25 build until the returned gate opens."""
    rebuild = orch._rebuild_bm25_from_collection
    gate, building = threading.Event(), threading.Event()

    def held_rebuild(**kwargs):
        building.set()
        gate.wait(_HOLD_LIMIT_S)
        return rebuild(**kwargs)

    monkeypatch.setattr(orch, "_rebuild_bm25_from_collection", held_rebuild)
    return gate, building


def _record_background_builds(orch, monkeypatch) -> list[str]:
    """Record the thread of every background build that runs from here on."""
    names: list[str] = []
    refresh = orch._refresh_bm25_in_background

    def recording_refresh():
        names.append(threading.current_thread().name)
        refresh()

    monkeypatch.setattr(orch, "_refresh_bm25_in_background", recording_refresh)
    return names


def _start_held_prewarm(orch, monkeypatch) -> threading.Event:
    """Start the prewarm as main() does and return the gate holding its build."""
    gate, building = _hold_builds(orch, monkeypatch)
    orch._start_bm25_background_build("bm25-prewarm")
    assert building.wait(_TIMEOUT_S), "the prewarm must be building"
    return gate


def _is_refresh_lock_free(orch, wait_s: float = 0) -> bool:
    """True once no background build holds the refresh lock, waiting up to ``wait_s``."""
    if not orch._bm25_refresh_lock.acquire(timeout=wait_s):
        return False
    orch._bm25_refresh_lock.release()
    return True


def _keyword_only_filenames(orch) -> list[str]:
    return [row["filename"] for row in orch.query("initial", max_results=3, hybrid_alpha=0.0)]


@pytest.mark.parametrize("is_large", [True, False], ids=["large", "small"])
def test_given_a_cold_index_and_the_prewarm_building_when_a_keyword_only_search_arrives_then_it_waits_for_the_build(
    isolated_orchestrator, monkeypatch, is_large
):
    orch = isolated_orchestrator
    _cold_index(orch, monkeypatch, is_large)
    gate = _start_held_prewarm(orch, monkeypatch)
    found: list = []
    search = threading.Thread(target=lambda: found.extend(_keyword_only_filenames(orch)), daemon=True)
    try:
        search.start()
        search.join(_ANSWER_WINDOW_S)
    finally:
        gate.set()
    search.join(_TIMEOUT_S)

    assert found == ["a1.md"], "it must not answer from an empty BM25"


@pytest.mark.parametrize("hybrid_alpha", [0.5, 1.0], ids=["hybrid", "semantic"])
def test_given_a_large_cold_index_and_the_prewarm_building_when_a_hybrid_or_semantic_search_arrives_then_it_does_not_wait(
    isolated_orchestrator, monkeypatch, hybrid_alpha
):
    orch = isolated_orchestrator
    _cold_index(orch, monkeypatch)
    gate = _start_held_prewarm(orch, monkeypatch)
    query = {"max_results": 3, "hybrid_alpha": hybrid_alpha}
    search = threading.Thread(target=orch.query, args=("initial",), kwargs=query, daemon=True)
    try:
        search.start()
        search.join(_TIMEOUT_S)
        answered = not search.is_alive()
    finally:
        gate.set()
    search.join(_TIMEOUT_S)
    _is_refresh_lock_free(orch, _TIMEOUT_S)  # nothing may touch the collection after the test

    assert answered, "only a keyword-only search waits for the prewarm"


def test_given_a_background_build_already_on_its_way_when_the_prewarm_starts_then_no_second_build_runs(
    isolated_orchestrator, monkeypatch
):
    orch = isolated_orchestrator
    _cold_index(orch, monkeypatch)
    gate, _ = _hold_builds(orch, monkeypatch)
    ran = _record_background_builds(orch, monkeypatch)
    try:
        orch.query("initial", max_results=3)  # a hybrid search starts the background build
        orch._start_bm25_background_build("bm25-prewarm")
    finally:
        gate.set()
    assert _is_refresh_lock_free(orch, _TIMEOUT_S), "the background build must end"

    assert ran == ["bm25-refresh"]


def test_given_the_prewarm_thread_cannot_start_when_the_prewarm_runs_then_the_lock_is_free_and_the_next_keyword_search_builds(
    isolated_orchestrator, monkeypatch
):
    orch = isolated_orchestrator
    _cold_index(orch, monkeypatch)
    start = threading.Thread.start

    def start_unless_prewarm(thread):
        if thread.name == "bm25-prewarm":
            raise RuntimeError("can't start new thread")
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", start_unless_prewarm)

    orch._start_bm25_background_build("bm25-prewarm")

    assert _is_refresh_lock_free(orch), "a prewarm that could not start must not keep the lock"
    assert _keyword_only_filenames(orch) == ["a1.md"]


def test_given_the_prewarm_build_fails_once_when_a_keyword_only_search_follows_then_it_retries_and_finds_the_document(
    isolated_orchestrator, monkeypatch
):
    orch = isolated_orchestrator
    _cold_index(orch, monkeypatch)
    get = orch.collection.get
    failures = [RuntimeError("injected read failure")]

    def get_failing_once(**kwargs):
        if failures:
            raise failures.pop()
        return get(**kwargs)

    monkeypatch.setattr(orch.collection, "get", get_failing_once)
    orch._start_bm25_background_build("bm25-prewarm")
    assert _is_refresh_lock_free(orch, _TIMEOUT_S), "a failed prewarm must give the lock back"

    assert _keyword_only_filenames(orch) == ["a1.md"]
