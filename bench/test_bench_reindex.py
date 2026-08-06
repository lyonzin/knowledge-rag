"""Reindex throughput benchmarks (v4.8.0 Fase 3).

Measures the orchestration cost of ``_index_document`` under three
configurations that the new ``documents.batch_size`` +
``documents.parallel_workers`` YAML fields expose:

    reindex_default          batch_size=500, workers=1  (v4.7.1 baseline)
    reindex_batch_2000       batch_size=2000, workers=1  (fewer round-trips)
    reindex_parallel_4       batch_size=2000, workers=4  (SQLite overlap)

The collection stub sleeps a fixed 5ms per ``add()`` call to simulate
SQLite write cost without depending on a real ChromaDB. This makes the
benchmark deterministic (no disk I/O jitter) and cheap in CI while still
exercising the branch we shipped in ``mcp_server/server.py``. The
absolute numbers are not comparable to production nuclear_rebuild wall
time — they exist so a follow-up PR can spot regressions in the
dispatch path itself (e.g. accidental Pool per-batch or a full-materialize
of ``batch_slices``).

Marked ``bench_reindex`` so users can select via ``-k bench_reindex``.
"""

from __future__ import annotations

import time
from typing import List
from unittest.mock import MagicMock

import pytest

from pathlib import Path

from mcp_server.ingestion import Chunk, Document
from mcp_server.server import KnowledgeOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _SleepingCollection:
    """Stub for ChromaDB collection.add — 5ms sleep per call.

    We use time.sleep (not busy-wait) so the ThreadPoolExecutor path can
    actually overlap sleeps across workers. If we spun the CPU, the GIL
    would serialize and the parallel path would show zero gain — which
    would be an artifact of the bench, not the real system.
    """

    #: Fixed simulated latency per add() call (seconds). Chosen so a
    #: 100-doc / 10-batch run at 5ms/batch = ~50ms sequential vs
    #: ~15ms with 4 workers — big enough to be measurable, small enough
    #: to keep the whole file under 5s in CI.
    ADD_LATENCY_SEC = 0.005

    def __init__(self) -> None:
        self.add_calls: List[int] = []  # sizes of each add() batch

    def add(self, ids, documents, metadatas):
        self.add_calls.append(len(ids))
        time.sleep(self.ADD_LATENCY_SEC)


def _make_synthetic_doc(n_chunks: int = 100) -> Document:
    """Build a Document with n_chunks unique chunks (dedup passthrough)."""
    chunk_len = 260
    chunks = [
        Chunk(
            # unique content per chunk — dedup by content_hash is bypassed
            content=f"synthetic chunk {i:04d} — {'lorem ipsum ' * 20}",
            index=i,
            start_char=i * chunk_len,
            end_char=(i + 1) * chunk_len,
            metadata={},
        )
        for i in range(n_chunks)
    ]
    return Document(
        id=f"synthetic-{n_chunks}c",
        content="",  # not used by _index_document; chunks carry the payload
        source=Path("/synthetic/reindex_bench.md"),
        format=".md",
        category="bench",
        chunks=chunks,
        keywords=["bench", "reindex"],
    )


@pytest.fixture
def synthetic_doc_100():
    """Freshly built 100-chunk document per benchmark run.

    Cannot be scope='session' — _index_document mutates state via
    self.bm25_index.add_documents. We want each round trip to see the
    same input.
    """
    return _make_synthetic_doc(n_chunks=100)


def _build_orchestrator() -> KnowledgeOrchestrator:
    """Build an Orchestrator with mocked collection + BM25 index.

    Bypasses __init__ (which touches ChromaDB, FastEmbed and disk) via
    object.__new__ — same pattern used in tests/test_search.py.
    """
    orch = object.__new__(KnowledgeOrchestrator)
    orch.collection = _SleepingCollection()
    orch.bm25_index = MagicMock()
    return orch


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="reindex")
class TestReindexThroughput:
    """Compare the 3 configurations exposed by v4.8.0 Fase 3.

    Numbers to look at (pytest-benchmark output):
        Mean         — average wall time per _index_document call
        StdDev       — should stay small (deterministic sleep)
        Rounds       — pytest-benchmark's autodetection

    Expected qualitative ordering (bs=500 → 5 batches, bs=2000 → 1 batch):
        reindex_batch_2000    fastest   (1 batch × 5ms = ~5ms overhead)
        reindex_default       middle    (5 batches × 5ms = ~25ms + Executor)
        reindex_parallel_4    tied w/ default when only 1 batch remains
                              (parallel path skipped for single-batch docs)
    """

    def test_reindex_default(self, benchmark, synthetic_doc_100, monkeypatch):
        """Baseline: batch_size=500, workers=1 (v4.7.1 defaults byte-for-byte)."""
        monkeypatch.setattr("mcp_server.server.config.batch_size", 500)
        monkeypatch.setattr("mcp_server.server.config.parallel_workers", 1)

        def run():
            orch = _build_orchestrator()
            return orch._index_document(synthetic_doc_100)

        indexed, skipped = benchmark(run)
        assert indexed == 100 and skipped == 0

    def test_reindex_batch_2000(self, benchmark, synthetic_doc_100, monkeypatch):
        """Fewer round-trips: batch_size=2000 → single batch for 100 chunks."""
        monkeypatch.setattr("mcp_server.server.config.batch_size", 2000)
        monkeypatch.setattr("mcp_server.server.config.parallel_workers", 1)

        def run():
            orch = _build_orchestrator()
            return orch._index_document(synthetic_doc_100)

        indexed, skipped = benchmark(run)
        assert indexed == 100 and skipped == 0

    def test_reindex_parallel_4_workers(self, benchmark, monkeypatch):
        """Parallel path: batch_size=50, workers=4 (2 batches of 50 → overlap)."""
        # 50-chunk batches × 4 workers over a 200-chunk doc → 4 concurrent
        # add()s. Would take ~20ms sequential, ~5ms with 4 workers.
        big_doc = _make_synthetic_doc(n_chunks=200)
        monkeypatch.setattr("mcp_server.server.config.batch_size", 50)
        monkeypatch.setattr("mcp_server.server.config.parallel_workers", 4)

        def run():
            orch = _build_orchestrator()
            return orch._index_document(big_doc)

        indexed, skipped = benchmark(run)
        assert indexed == 200 and skipped == 0
