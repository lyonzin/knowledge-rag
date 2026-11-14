"""Regression tests for GH #299 — a search must not wait minutes for BM25.

Document writes used to set ``_bm25_initialized = False``, so the next
hybrid query rebuilt BM25 from every chunk in Chroma (pages of 500)
synchronously — 2-4 minutes on a ~610k-chunk index, timing out MCP
clients after any change. Writes now mark the published generation
stale. A small index is refreshed before the search, as it always was;
a large one keeps serving searches while a background thread refreshes
it — or builds it, right after a start. Only a build that published
clears the stale mark, so a refresh that could not run or failed is
retried by a later query.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from mcp_server import server as srv


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


def _cold_index(orch, monkeypatch) -> None:
    """Three documents and no BM25 generation yet, as right after a start."""
    monkeypatch.setattr(srv.config, "reranker_enabled", False)
    orch.add_document_from_content(content="initial corpus content", filepath="a1.md", category="t")
    orch.add_document_from_content(content="other filler text here", filepath="a2.md", category="t")
    orch.add_document_from_content(content="third unrelated passage", filepath="a3.md", category="t")
    assert not orch._bm25_initialized


def _warm_index(orch, monkeypatch) -> None:
    """Three documents and a first hybrid query: a published BM25 generation."""
    _cold_index(orch, monkeypatch)
    orch.query("initial", max_results=3)
    assert orch._bm25_initialized
    assert not orch._bm25_stale.is_set()


def _write_new_document(orch) -> None:
    orch.add_document_from_content(content="brandnew uniquetoken quarks", filepath="b.md", category="t")
    assert orch._bm25_initialized, "the published generation must survive the write"
    assert orch._bm25_stale.is_set(), "the write must mark the generation stale"


def _record_build_threads(orch, monkeypatch) -> list:
    """Collect the name of the thread of every BM25 build from here on.

    With three documents a background build is over before the search reads
    BM25, so the results of a search cannot tell where the build ran.
    """
    names: list = []
    rebuild = orch._rebuild_bm25_from_collection

    def recording_rebuild(**kwargs):
        names.append(threading.current_thread().name)
        return rebuild(**kwargs)

    monkeypatch.setattr(orch, "_rebuild_bm25_from_collection", recording_rebuild)
    return names


def _keyword_hit(orch, term: str, hybrid_alpha: float = 0.5) -> bool:
    return any(r["bm25_rank"] is not None for r in orch.query(term, max_results=3, hybrid_alpha=hybrid_alpha))


class TestSmallIndexRefreshesBeforeTheSearch:
    """An index within the inline limit behaves as it did before GH #299."""

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

    def test_index_exactly_at_the_limit_is_still_refreshed_before_the_search(
        self, isolated_orchestrator, monkeypatch
    ) -> None:
        """given an index holding exactly as many chunks as the inline limit,
        when a search follows a write, then the search itself refreshes BM25
        and finds the new document."""
        orch = isolated_orchestrator
        _warm_index(orch, monkeypatch)
        _write_new_document(orch)
        monkeypatch.setattr(orch, "_BM25_INLINE_REFRESH_CHUNKS", orch.collection.count())
        built_on = _record_build_threads(orch, monkeypatch)

        assert _keyword_hit(orch, "uniquetoken")
        assert built_on == [threading.current_thread().name]

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

    def test_write_that_fails_and_cannot_be_rolled_back_marks_the_generation_stale(
        self, isolated_orchestrator, monkeypatch
    ) -> None:
        """given a write that fails after changing the collection, when
        restoring the previous chunks fails as well, then the generation is
        marked stale — the collection no longer matches it."""
        orch = isolated_orchestrator
        _warm_index(orch, monkeypatch)
        edited = srv.config.documents_dir / "a1.md"
        edited.write_text("initial corpus content, rewritten and longer", encoding="utf-8")
        monkeypatch.setattr(orch, "_index_document", MagicMock(side_effect=RuntimeError("embedding failure")))
        monkeypatch.setattr(orch, "_restore_document_chunks", MagicMock(side_effect=OSError("disk full")))

        stats = orch.index_paths([str(edited)])

        assert stats["errors"] == 1, stats
        assert orch._bm25_stale.is_set()

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
    """An index holding more chunks than the inline limit."""

    def _large_stale_index(self, orch, monkeypatch) -> None:
        _warm_index(orch, monkeypatch)
        _write_new_document(orch)
        monkeypatch.setattr(orch, "_BM25_INLINE_REFRESH_CHUNKS", orch.collection.count() - 1)

    def test_search_does_not_wait_for_the_refresh(self, isolated_orchestrator, monkeypatch) -> None:
        """given a stale large index and a refresh that has not finished, when
        a hybrid search runs, then it returns, serving the published generation."""
        orch = isolated_orchestrator
        self._large_stale_index(orch, monkeypatch)
        release = threading.Event()
        monkeypatch.setattr(orch, "_rebuild_bm25_from_collection", lambda **kwargs: release.wait(30))
        answered = threading.Event()
        hits: list = []

        def search():
            hits.append(_keyword_hit(orch, "initial"))
            answered.set()

        thread = threading.Thread(target=search)
        thread.start()
        try:
            assert answered.wait(10), "the search must not wait for the refresh"
            assert hits == [True], "the published generation is served"
            assert not _refresh_idle(orch), "the refresh is still running"
        finally:
            release.set()
            thread.join(5)
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

    def test_refresh_publishes_the_new_generation(self, isolated_orchestrator, monkeypatch) -> None:
        """given a stale large index, when a search starts the refresh, then
        once it published a search has the new document's keyword match."""
        orch = isolated_orchestrator
        self._large_stale_index(orch, monkeypatch)

        orch.query("quarks", max_results=3)

        assert _wait_until(lambda: not orch._bm25_stale.is_set())
        assert _keyword_hit(orch, "uniquetoken")

    def test_published_refresh_drops_results_cached_from_the_replaced_generation(
        self, isolated_orchestrator, monkeypatch
    ) -> None:
        """given a search cached while the refresh was still running, when the
        refresh publishes, then the same search is computed again and has the
        keyword match it lacked."""
        orch = isolated_orchestrator
        self._large_stale_index(orch, monkeypatch)
        rebuild = orch._rebuild_bm25_from_collection
        gate = threading.Event()

        def gated_rebuild(**kwargs):
            gate.wait(30)
            return rebuild(**kwargs)

        monkeypatch.setattr(orch, "_rebuild_bm25_from_collection", gated_rebuild)

        assert not _keyword_hit(orch, "uniquetoken"), "served from the generation without the new document"
        assert not _keyword_hit(orch, "uniquetoken"), "and cached"
        assert orch.query_cache.stats()["hits"] == 1
        gate.set()

        assert _wait_until(lambda: _refresh_idle(orch) and not orch._bm25_stale.is_set())
        assert _keyword_hit(orch, "uniquetoken"), "the cached result must not outlive the generation it came from"

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

    def test_refresh_that_could_not_start_can_start_later(self, isolated_orchestrator, monkeypatch) -> None:
        """given the background thread cannot be started, when a later search
        runs, then a refresh can still start — the guard was not left taken."""
        orch = isolated_orchestrator
        self._large_stale_index(orch, monkeypatch)
        start = threading.Thread.start
        refused: list = []

        def start_unless_refresh(thread):
            if thread.name == "bm25-refresh":
                refused.append(thread)
                raise RuntimeError("can't start new thread")
            start(thread)

        monkeypatch.setattr(threading.Thread, "start", start_unless_refresh)

        orch.query("one", max_results=3)

        monkeypatch.setattr(threading.Thread, "start", start)
        assert len(refused) == 1
        assert _refresh_idle(orch)
        assert orch._bm25_stale.is_set()
        assert _wait_until(lambda: _keyword_hit(orch, "quarks"))


class TestLargeIndexRightAfterAStart:
    """A large index with no BM25 generation yet: the first search after a start."""

    def _large_cold_index(self, orch, monkeypatch) -> None:
        _cold_index(orch, monkeypatch)
        monkeypatch.setattr(orch, "_BM25_INLINE_REFRESH_CHUNKS", orch.collection.count() - 1)

    def test_first_hybrid_search_does_not_wait_for_the_build(self, isolated_orchestrator, monkeypatch) -> None:
        """given a large index right after a start, when a hybrid search runs,
        then it answers from the semantic channel while BM25 is built in the
        background, and once that is published the same search has its
        keyword match."""
        orch = isolated_orchestrator
        self._large_cold_index(orch, monkeypatch)
        rebuild = orch._rebuild_bm25_from_collection
        gate = threading.Event()

        def gated_rebuild(**kwargs):
            gate.wait(30)
            return rebuild(**kwargs)

        monkeypatch.setattr(orch, "_rebuild_bm25_from_collection", gated_rebuild)
        answered = threading.Event()
        found: list = []

        def search():
            found.extend(orch.query("initial", max_results=3))
            answered.set()

        thread = threading.Thread(target=search)
        thread.start()
        try:
            assert answered.wait(10), "the search must not wait for the build"
            assert found, "the semantic channel answers"
            assert all(row["bm25_rank"] is None for row in found)
            assert not orch._bm25_initialized
        finally:
            gate.set()
            thread.join(5)

        assert _wait_until(lambda: orch._bm25_initialized and _refresh_idle(orch))
        assert _keyword_hit(orch, "initial"), "once the build is published the same search has its keyword match"

    def test_first_keyword_only_search_builds_before_answering(self, isolated_orchestrator, monkeypatch) -> None:
        """given a large index right after a start, when a keyword-only search
        runs, then the search itself builds BM25 — there is no other channel
        to answer from."""
        orch = isolated_orchestrator
        self._large_cold_index(orch, monkeypatch)
        built_on = _record_build_threads(orch, monkeypatch)

        results = orch.query("initial", max_results=3, hybrid_alpha=0.0)

        assert [row["filename"] for row in results] == ["a1.md"]
        assert built_on == [threading.current_thread().name]

    def test_keyword_only_search_waits_for_a_build_already_on_its_way(self, isolated_orchestrator, monkeypatch) -> None:
        """given a hybrid search already started the background build, when a
        keyword-only search arrives before it is published, then it waits for
        the build instead of answering that nothing matches."""
        orch = isolated_orchestrator
        self._large_cold_index(orch, monkeypatch)
        rebuild = orch._rebuild_bm25_from_collection
        gate = threading.Event()

        def gated_rebuild(**kwargs):
            gate.wait(30)
            return rebuild(**kwargs)

        monkeypatch.setattr(orch, "_rebuild_bm25_from_collection", gated_rebuild)
        orch.query("initial", max_results=3)
        assert not _refresh_idle(orch), "the background build is on its way"
        answered = threading.Event()
        found: list = []

        def keyword_search():
            found.extend(orch.query("initial", max_results=3, hybrid_alpha=0.0))
            answered.set()

        thread = threading.Thread(target=keyword_search)
        thread.start()
        try:
            assert not answered.wait(0.3), "it must not answer before the build is published"
        finally:
            gate.set()
            thread.join(10)

        assert [row["filename"] for row in found] == ["a1.md"]

    def test_small_index_right_after_a_start_builds_before_answering(self, isolated_orchestrator, monkeypatch) -> None:
        """given an index within the inline limit right after a start, when a
        hybrid search runs, then the search itself builds BM25, as before."""
        orch = isolated_orchestrator
        _cold_index(orch, monkeypatch)
        built_on = _record_build_threads(orch, monkeypatch)

        assert _keyword_hit(orch, "initial")
        assert built_on == [threading.current_thread().name]

    def test_search_answered_without_keyword_matches_is_not_served_again_from_the_cache(
        self, isolated_orchestrator, monkeypatch
    ) -> None:
        """given a writer holding the index lock on a large index right after a
        start, when a hybrid search is answered from the semantic channel, then
        the same search after the writer is done has its keyword match."""
        orch = isolated_orchestrator
        self._large_cold_index(orch, monkeypatch)
        held, release = threading.Event(), threading.Event()

        def writer():
            with orch._index_lock:
                held.set()
                release.wait(30)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            assert held.wait(5)
            assert not _keyword_hit(orch, "initial"), "answered from the semantic channel"
            assert _wait_until(lambda: _refresh_idle(orch))
        finally:
            release.set()
            thread.join(5)

        assert _wait_until(lambda: _keyword_hit(orch, "initial"))

    def test_build_that_did_not_run_keeps_the_query_cache(self, isolated_orchestrator, monkeypatch) -> None:
        """given a writer holding the index lock on a large index right after a
        start, when a hybrid search cannot start the build, then a result that
        is in the cache stays there."""
        orch = isolated_orchestrator
        self._large_cold_index(orch, monkeypatch)
        orch.query("initial", max_results=3, hybrid_alpha=1.0)
        held, release = threading.Event(), threading.Event()

        def writer():
            with orch._index_lock:
                held.set()
                release.wait(30)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            assert held.wait(5)
            orch.query("corpus", max_results=3)
            assert _wait_until(lambda: _refresh_idle(orch))
            orch.query("initial", max_results=3, hybrid_alpha=1.0)
        finally:
            release.set()
            thread.join(5)

        assert orch.query_cache.stats()["hits"] == 1

    def test_collection_that_cannot_be_counted_is_reported(self, isolated_orchestrator, monkeypatch, capsys) -> None:
        """given the collection cannot be counted, when a hybrid search runs
        right after a start, then it is still answered and stderr says why BM25
        was left as it is."""
        orch = isolated_orchestrator
        _cold_index(orch, monkeypatch)
        monkeypatch.setattr(orch.collection, "count", MagicMock(side_effect=RuntimeError("database is locked")))

        results = orch.query("initial", max_results=3)

        assert results, "the semantic channel answers"
        assert "could not be counted: database is locked" in capsys.readouterr().err
