"""Memory and concurrency regressions for long-running metrics collection."""

import gc
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import pytest

from mcp_server.metrics import MetricsCollector


def test_histogram_retains_bounded_memory_and_valid_labelled_samples():
    collector = MetricsCollector()
    collector.register_histogram_buckets("latency", (0.1, 0.5, 1.0))
    collector.observe("latency", 0.5, '{tool="search"}')
    gc.collect()
    tracemalloc.start()
    try:
        for _ in range(100_000):
            collector.observe("latency", 0.5, '{tool="search"}')
        retained, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert retained < 64_000, (retained, peak)
    assert peak < 128_000, peak
    samples = collector.exposition().splitlines()
    assert 'latency_count{tool="search"} 100001' in samples
    assert 'latency_sum{tool="search"} 50000.500000' in samples
    assert 'latency_bucket{tool="search",le="0.1"} 0' in samples
    assert 'latency_bucket{tool="search",le="0.5"} 100001' in samples
    assert 'latency_bucket{tool="search",le="+Inf"} 100001' in samples


def test_histogram_concurrent_observation_and_scrape_preserve_totals():
    collector = MetricsCollector()
    collector.register_histogram_buckets("latency", (0.25, 0.5))

    def observe():
        for offset in range(2_000):
            collector.observe("latency", 0.5)
            if offset % 100 == 0:
                collector.exposition()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: observe(), range(8)))
    assert "latency_count 16000\n" in collector.exposition()
    assert "latency_sum 8000.000000\n" in collector.exposition()
    assert 'latency_bucket{le="0.5"} 16000\n' in collector.exposition()


def test_histogram_registration_does_not_invent_historical_bucket_counts():
    collector = MetricsCollector()
    collector.register_histogram_buckets("registered", (0.5,))
    collector.observe("registered", 0.2)
    collector.register_histogram_buckets("registered", (0.5,))
    with pytest.raises(ValueError, match="before observing"):
        collector.register_histogram_buckets("registered", (0.1,))
    collector.observe("plain", 0.2)
    with pytest.raises(ValueError, match="before observing"):
        collector.register_histogram_buckets("plain", (0.5,))
