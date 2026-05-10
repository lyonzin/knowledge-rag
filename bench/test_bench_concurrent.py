"""Concurrency throughput benchmark.

Measures total wall time to process N parallel BM25 queries from a
ThreadPoolExecutor — proxy for "what happens when 50 LLM agents hit
the MCP server at the same time".
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.fixture(scope="module")
def bm25_index_seeded():
    from mcp_server.server import BM25Index

    idx = BM25Index()
    ids = [f"doc-{i}" for i in range(2000)]
    texts = [f"document {i} kerberoast bloodhound mimikatz impacket pass-the-hash dcsync" for i in range(2000)]
    idx.add_documents(ids, texts)
    idx.build_index()
    return idx


def _run_queries(index, n_queries: int) -> int:
    queries = ["kerberoast", "mimikatz pass-the-hash", "dcsync", "bloodhound impacket"] * (n_queries // 4 + 1)
    queries = queries[:n_queries]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda q: index.search(q, top_k=5), queries))
    return sum(len(r) for r in results)


def test_bench_concurrent_10_queries(benchmark, bm25_index_seeded):
    def workload():
        return _run_queries(bm25_index_seeded, 10)

    total = benchmark(workload)
    assert total >= 0


def test_bench_concurrent_50_queries(benchmark, bm25_index_seeded):
    def workload():
        return _run_queries(bm25_index_seeded, 50)

    total = benchmark(workload)
    assert total >= 0


def test_bench_concurrent_100_queries(benchmark, bm25_index_seeded):
    def workload():
        return _run_queries(bm25_index_seeded, 100)

    total = benchmark(workload)
    assert total >= 0
