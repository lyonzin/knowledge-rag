"""Regression tests for GH #299 — queries must never block on a BM25 refresh.

Document writes used to set ``_bm25_initialized = False``, so the next
hybrid query rebuilt BM25 from every chunk in Chroma (pages of 500)
synchronously — 2-4 minutes on a ~610k-chunk index, timing out MCP
clients after any change. Now writes only mark the published generation
stale; hybrid queries keep serving it and refresh on a background
thread, atomically swapping the reference when the new build is ready.
"""

from __future__ import annotations

import time

from mcp_server import server as srv


class TestBm25AsyncRefresh:
    def _make_stale_orch(self, isolated_orchestrator, monkeypatch):
        monkeypatch.setattr(srv.config, "reranker_enabled", False)
        orch = isolated_orchestrator
        orch._bm25_initialized = True
        orch._bm25_stale.set()
        return orch

    def test_query_does_not_block_on_refresh(self, isolated_orchestrator, monkeypatch) -> None:
        """given a stale warm index and a slow refresh, when a hybrid query
        runs, then it returns immediately, serving the current generation."""
        orch = self._make_stale_orch(isolated_orchestrator, monkeypatch)

        def slow_rebuild():
            time.sleep(0.6)

        monkeypatch.setattr(orch, "_rebuild_bm25_from_collection", slow_rebuild)

        start = time.monotonic()
        orch.query("anything", max_results=3)
        elapsed = time.monotonic() - start

        assert elapsed < 0.4, f"query must not wait for the refresh, took {elapsed:.2f}s"
        assert orch._bm25_initialized, "the previous generation must stay published"

    def test_concurrent_queries_trigger_a_single_refresh(self, isolated_orchestrator, monkeypatch) -> None:
        """given a stale warm index, when several hybrid queries run, then only
        one background refresh is spawned."""
        orch = self._make_stale_orch(isolated_orchestrator, monkeypatch)
        calls: list[int] = []

        def counted_rebuild():
            calls.append(1)
            time.sleep(0.1)

        monkeypatch.setattr(orch, "_rebuild_bm25_from_collection", counted_rebuild)

        for _ in range(5):
            orch.query("x", max_results=3)
        time.sleep(0.5)

        assert len(calls) == 1, f"expected exactly one refresh, got {len(calls)}"

    def test_refresh_swaps_in_the_new_generation(self, isolated_orchestrator, monkeypatch) -> None:
        """given a document written after the last build, when hybrid queries
        run, then the next generation includes the new document's keyword hits."""
        monkeypatch.setattr(srv.config, "reranker_enabled", False)
        orch = isolated_orchestrator
        orch.add_document_from_content(content="initial corpus content", filepath="a1.md", category="t")
        orch.add_document_from_content(content="other filler text here", filepath="a2.md", category="t")
        orch.query("initial", max_results=3)  # cold build of the first generation
        assert orch._bm25_initialized

        # three docs total so the new token keeps a positive IDF (GH #299 e2e)
        orch.add_document_from_content(content="brandnew uniquetoken quarks", filepath="b.md", category="t")
        assert orch._bm25_initialized, "generation must survive the write (GH #299)"
        assert orch._bm25_stale.is_set(), "write must mark the generation stale"

        orch.query("uniquetoken", max_results=3)  # serves old generation, spawns refresh

        again: list = []
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            again = orch.query("uniquetoken", max_results=3)
            if any(r["bm25_rank"] is not None for r in again):
                break
            time.sleep(0.1)

        assert any(r["bm25_rank"] is not None for r in again), (
            "the refreshed generation must serve the new document's keyword hits"
        )
