"""Memory pressure benchmark.

Measures peak RSS during a synthetic workload — the perf gate uses
the result to detect releases that newly leak under load.
"""

from __future__ import annotations

import gc
import os

import pytest

psutil = pytest.importorskip("psutil")


def _rss_mb() -> int:
    return int(os.popen("").read() or "0") or int(psutil.Process().memory_info().rss / (1024 * 1024))


def _real_rss_mb() -> int:
    return int(psutil.Process().memory_info().rss / (1024 * 1024))


def test_bench_orchestrator_idle_rss(benchmark, fake_embed_fn):
    """RSS after constructing 50 idle FastEmbedEmbeddings — must stay flat."""
    from mcp_server.server import FastEmbedEmbeddings

    def measure():
        gc.collect()
        before = _real_rss_mb()
        embedders = [FastEmbedEmbeddings() for _ in range(50)]
        gc.collect()
        after = _real_rss_mb()
        del embedders
        return after - before

    delta = benchmark(measure)
    # The benchmark machinery returns the measured callable's return value;
    # we still assert a soft bound so a regression shows up as a failure.
    assert delta < 50, f"50 idle embedders allocated {delta} MB"


def test_bench_query_cache_5000_entries(benchmark):
    """RSS pressure of pumping 5000 entries through the cache."""
    from mcp_server.server import QueryCache

    def measure():
        gc.collect()
        before = _real_rss_mb()
        cache = QueryCache(max_size=1000, ttl_seconds=300)
        for i in range(5000):
            cache.put(f"q-{i}", 5, None, 0.3, [{"content": "x" * 100}])
        gc.collect()
        after = _real_rss_mb()
        return after - before

    delta = benchmark(measure)
    assert delta < 80, f"5000 cache writes allocated {delta} MB"
