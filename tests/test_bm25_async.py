"""Regression tests for GH #299 — a search must not wait minutes for BM25.

Document writes used to set ``_bm25_initialized = False``, so the next
hybrid query rebuilt BM25 from every chunk in Chroma (pages of 500)
synchronously — 2-4 minutes on a ~610k-chunk index, timing out MCP
clients after any change. Writes now mark the published generation
stale. A stale generation whose last build was quick is refreshed before
the search, as it always was; one whose build takes longer keeps serving
searches while a background thread refreshes it. Only a build that
published clears the stale mark, so a refresh that could not run or
failed is retried by a later query.
"""

from __future__ import annotations

import threading
import time

from mcp_server import server as srv

LARGE_INDEX_BUILD_SECONDS = 60.0


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _refresh_idle(orch) -> bool:
    if not orch._bm25_refresh_lock.acquire(blocking=False):
        return False
    orch._bm25_refresh_lock.release()
    return True


def _warm_index(orch, monkeypatch) -> None:
    """Three documents and a first hybrid query: a published BM25 generation."""
    monkeypatch.setattr(srv.config, "reranker_enabled", False)
    orch.add_document_from_content(content="initial corpus content", filepath="a1.md", category="t")
    orch.add_document_from_content(content="other filler text here", filepath="a2.md", category="t")
    orch.add_document_from_content(content="third unrelated passage", filepath="a3.md", category="t")
    orch.query("initial", max_results=3)
    assert orch._bm25_initialized
    assert not orch._bm25_stale.is_set()


def _write_new_document(orch) -> None:
    orch.add_document_from_content(content="brandnew uniquetoken quarks", filepath="b.md", category="t")
    assert orch._bm25_initialized, "the published generation must survive the write"
    assert orch._bm25_stale.is_set(), "the write must mark the generation stale"


def _keyword_hit(orch, term: str, hybrid_alpha: float = 0.5) -> bool:
    return any(r["bm25_rank"] is not None for r in orch.query(term, max_results=3, hybrid_alpha=hybrid_alpha))


class TestQuickIndexRefreshesBeforeTheSearch:
    """An index whose last build was quick behaves as it did before GH #299."""

    def test_keyword_search_right_after_a_write_finds_the_new_document(
        self, isolated_orchestrator, monkeypatch
    ) -> None:
        """given a warm index, when a document is written and a keyword-only
        search follows immediately, then the search finds the new document."""
        orch = isolated_orchestrator
        _warm_index(orch, monkeypatch)
        _write_new_document(orch)

        results = orch.query("uniquetoken", max_results=3, hybrid_alpha=0.0)

        assert [r["filename"] for r in results] == ["b.md"]
        assert not orch._bm25_stale.is_set()

    def test_hybrid_search_right_after_a_write_has_the_keyword_match(self, isolated_orchestrator, monkeypatch) -> None:
        """given a warm index, when a document is written and a hybrid search
        follows immediately, then the new document is a keyword match."""
        orch = isolated_orchestrator
        _warm_index(orch, monkeypatch)
        _write_new_document(orch)

        assert _keyword_hit(orch, "uniquetoken")

    def test_refresh_behind_a_writer_stays_due(self, isolated_orchestrator, monkeypatch) -> None:
        """given a writer holding the index lock, when a search cannot refresh,
        then the generation stays marked stale and a later search refreshes it."""
        orch = isolated_orchestrator
        _warm_index(orch, monkeypatch)
        _write_new_document(orch)
        held, release = threading.Event(), threading.Event()

        def writer():
            with orch._index_lock:
                held.set()
                release.wait(10)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            assert held.wait(5)
            assert not _keyword_hit(orch, "uniquetoken"), "the published generation is served meanwhile"
            assert orch._bm25_stale.is_set(), "a refresh that did not run must stay due"
        finally:
            release.set()
            thread.join(5)

        # A query not yet in the cache: a cached one returns before the BM25 guard.
        assert _keyword_hit(orch, "quarks")

    def test_failed_refresh_stays_due(self, isolated_orchestrator, monkeypatch) -> None:
        """given a refresh whose build fails, when it ends, then the previous
        generation is still published and still marked stale."""
        orch = isolated_orchestrator
        _warm_index(orch, monkeypatch)
        _write_new_document(orch)
        build = orch.bm25_index.build_index

        def failing_build():
            raise MemoryError("injected OOM")

        monkeypatch.setattr(orch.bm25_index, "build_index", failing_build)

        assert _keyword_hit(orch, "initial"), "the previous generation must stay published"
        assert orch._bm25_initialized
        assert orch._bm25_stale.is_set(), "a failed refresh must stay due"

        monkeypatch.setattr(orch.bm25_index, "build_index", build)
        assert _keyword_hit(orch, "quarks")

    def test_first_hybrid_query_is_cached(self, isolated_orchestrator, monkeypatch) -> None:
        """given a cold index, when the same hybrid query runs twice, then the
        second one is served from the query cache."""
        monkeypatch.setattr(srv.config, "reranker_enabled", False)
        orch = isolated_orchestrator
        orch.add_document_from_content(content="initial corpus content", filepath="a1.md", category="t")

        first = orch.query("initial", max_results=3)
        second = orch.query("initial", max_results=3)

        assert second == first
        assert orch.query_cache.stats()["hits"] == 1

    def test_full_rebuild_does_not_leave_the_new_generation_stale(
        self, isolated_orchestrator, monkeypatch, tmp_path
    ) -> None:
        """given a full rebuild through the staging collection, when it ends,
        then the published generation is fresh — no redundant refresh is due."""
        monkeypatch.setattr(srv.config, "chroma_dir", tmp_path / "chroma")
        orch = isolated_orchestrator
        _warm_index(orch, monkeypatch)

        orch.nuclear_rebuild()

        assert orch._bm25_initialized
        assert not orch._bm25_stale.is_set(), "staging writes must not mark production stale"

    def test_reindex_does_not_leave_the_rebuilt_generation_stale(
        self, isolated_orchestrator, monkeypatch, tmp_path
    ) -> None:
        """given a reindex that changes documents and then rebuilds BM25, when
        it ends, then no second rebuild is due."""
        monkeypatch.setattr(srv.config, "chroma_dir", tmp_path / "chroma")
        orch = isolated_orchestrator
        _warm_index(orch, monkeypatch)
        (srv.config.documents_dir / "a1.md").write_text("initial corpus content, edited", encoding="utf-8")

        orch.reindex_all()

        assert orch._bm25_initialized
        assert not orch._bm25_stale.is_set()


class TestLargeIndexRefreshesInTheBackground:
    """An index whose last build took longer than the inline budget."""

    def _large_stale_index(self, orch, monkeypatch) -> None:
        _warm_index(orch, monkeypatch)
        orch._bm25_build_seconds = LARGE_INDEX_BUILD_SECONDS
        _write_new_document(orch)

    def test_search_does_not_wait_for_the_refresh(self, isolated_orchestrator, monkeypatch) -> None:
        """given a stale large index and a slow refresh, when a hybrid search
        runs, then it returns at once, serving the published generation."""
        orch = isolated_orchestrator
        self._large_stale_index(orch, monkeypatch)
        release = threading.Event()
        monkeypatch.setattr(orch, "_rebuild_bm25_from_collection", lambda **kwargs: release.wait(10))

        started = time.monotonic()
        assert _keyword_hit(orch, "initial")
        elapsed = time.monotonic() - started
        release.set()

        assert elapsed < 5, f"the search must not wait for the refresh, took {elapsed:.2f}s"
        assert _wait_until(lambda: _refresh_idle(orch))

    def test_concurrent_searches_start_a_single_refresh(self, isolated_orchestrator, monkeypatch) -> None:
        """given a stale large index, when several searches run while a refresh
        is in flight, then only one refresh is started."""
        orch = isolated_orchestrator
        self._large_stale_index(orch, monkeypatch)
        calls: list[int] = []
        release = threading.Event()

        def counted_rebuild(**kwargs):
            calls.append(1)
            release.wait(10)

        monkeypatch.setattr(orch, "_rebuild_bm25_from_collection", counted_rebuild)

        for term in ("one", "two", "three", "four", "five"):
            orch.query(term, max_results=3)
        release.set()

        assert _wait_until(lambda: _refresh_idle(orch))
        assert len(calls) == 1, f"expected exactly one refresh, got {len(calls)}"

    def test_refresh_publishes_the_new_generation_and_drops_cached_results(
        self, isolated_orchestrator, monkeypatch
    ) -> None:
        """given a stale large index, when a search starts the refresh, then
        the same search, once the refresh published, has the keyword match —
        its cached result from the replaced generation is dropped."""
        orch = isolated_orchestrator
        self._large_stale_index(orch, monkeypatch)

        assert _wait_until(lambda: _keyword_hit(orch, "uniquetoken"))
        assert not orch._bm25_stale.is_set()

    def test_refresh_behind_a_writer_stays_due(self, isolated_orchestrator, monkeypatch) -> None:
        """given a writer holding the index lock, when the background refresh
        cannot run, then the generation stays marked stale and a later search
        refreshes it."""
        orch = isolated_orchestrator
        self._large_stale_index(orch, monkeypatch)
        held, release = threading.Event(), threading.Event()

        def writer():
            with orch._index_lock:
                held.set()
                release.wait(10)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            assert held.wait(5)
            orch.query("uniquetoken", max_results=3)
            assert _wait_until(lambda: _refresh_idle(orch))
            assert orch._bm25_stale.is_set(), "a refresh that did not run must stay due"
        finally:
            release.set()
            thread.join(5)

        assert _wait_until(lambda: _keyword_hit(orch, "quarks"))
