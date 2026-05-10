"""Search-pipeline microbenchmarks (BM25, RRF, query cache).

Tracked metrics for the perf gate:
    bench_bm25_query                BM25 keyword search latency
    bench_query_cache_lru_hot       cache hit path
    bench_query_cache_lru_cold      cache miss path
"""

from __future__ import annotations

import pytest


def _seed_bm25(bm25, n_docs: int = 1000) -> None:
    """Populate the BM25 index with synthetic security-flavored docs."""
    ids = [f"doc-{i}" for i in range(n_docs)]
    texts = [
        f"document {i} about kerberoast bloodhound impacket privilege escalation lateral movement"
        for i in range(n_docs)
    ]
    bm25.add_documents(ids, texts)
    bm25.build_index()


@pytest.fixture(scope="module")
def bm25_index():
    from mcp_server.server import BM25Index

    idx = BM25Index()
    _seed_bm25(idx, n_docs=1000)
    return idx


def test_bench_bm25_query_1k_corpus(benchmark, bm25_index):
    """BM25 query against a 1000-doc corpus — typical user index size."""

    def query():
        return bm25_index.search("kerberoast escalation", top_k=10)

    result = benchmark(query)
    assert isinstance(result, list)


def test_bench_query_cache_hot(benchmark):
    """Cache hit path — the latency floor for repeated queries."""
    from mcp_server.server import QueryCache

    cache = QueryCache(max_size=100, ttl_seconds=300)
    cache.put("hot-key", 5, None, 0.3, [{"content": "cached"}])

    def hit():
        return cache.get("hot-key", 5, None, 0.3)

    result = benchmark(hit)
    assert result is not None


def test_bench_query_cache_miss(benchmark):
    """Cache miss path — confirms misses do not regress over time."""
    from mcp_server.server import QueryCache

    cache = QueryCache(max_size=100, ttl_seconds=300)

    def miss():
        return cache.get("never-seen", 5, None, 0.3)

    result = benchmark(miss)
    assert result is None


def test_bench_query_expansion(benchmark):
    """Security-term query expansion path."""
    from mcp_server.server import BM25Index

    idx = BM25Index()

    def expand():
        return idx.expand_query("sqli xss privesc kerberoast")

    result = benchmark(expand)
    assert "sql injection" in result
