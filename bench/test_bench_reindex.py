"""Compare batch dispatch using the same 8000-chunk document in every case.

The collection sleeps 5ms per write. These are orchestration measurements,
not ONNX or real SQLite throughput: 500-chunk batches require16 writes,
2000-chunk batches require4 writes, and four workers can overlap those4.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_server.ingestion import Chunk, Document
from mcp_server.server import KnowledgeOrchestrator


class _SleepingCollection:
    """Bounded per-run record of real dispatch calls, with simulated I/O latency."""

    def __init__(self):
        self.batch_lengths = []

    def upsert(self, ids, documents, metadatas):
        self.batch_lengths.append(len(ids))
        time.sleep(0.005)

    add = upsert  # Run this same workload against pre-upsert runtime revisions.


@pytest.fixture(scope="module")
def synthetic_doc():
    """Shared immutable input large enough to exercise both batch sizes."""
    chunks = [
        Chunk(
            content=f"unique chunk {index}: " + "lorem ipsum " * 20,
            index=index,
            start_char=index * 260,
            end_char=(index + 1) * 260,
            metadata={},
        )
        for index in range(8000)
    ]
    return Document(
        id="batch-benchmark",
        content="",
        source=Path("/synthetic/reindex.md"),
        format=".md",
        category="bench",
        chunks=chunks,
        keywords=["bench"],
    )


def _measure(benchmark, document, batch_size, workers, monkeypatch):
    monkeypatch.setattr("mcp_server.server.config.batch_size", batch_size)
    monkeypatch.setattr("mcp_server.server.config.parallel_workers", workers)
    benchmark.extra_info.update({"workload_version": 2, "chunks": 8000, "batch_size": batch_size, "workers": workers})

    def run():
        orchestrator = object.__new__(KnowledgeOrchestrator)
        orchestrator.collection = _SleepingCollection()
        orchestrator.bm25_index = SimpleNamespace(add_documents=lambda *args: None)
        orchestrator.fts5_index = None
        indexed, skipped = orchestrator._index_document(document)
        return indexed, skipped, orchestrator.collection.batch_lengths

    indexed, skipped, batches = benchmark(run)
    assert indexed == 8000 and skipped == 0
    assert len(batches) == 8000 // batch_size
    assert all(length == batch_size for length in batches)


@pytest.mark.benchmark(group="reindex")
class TestReindexThroughput:
    def test_reindex_default(self, benchmark, synthetic_doc, monkeypatch):
        _measure(benchmark, synthetic_doc, 500, 1, monkeypatch)

    def test_reindex_batch_2000(self, benchmark, synthetic_doc, monkeypatch):
        _measure(benchmark, synthetic_doc, 2000, 1, monkeypatch)

    def test_reindex_parallel_4_workers(self, benchmark, synthetic_doc, monkeypatch):
        _measure(benchmark, synthetic_doc, 2000, 4, monkeypatch)
