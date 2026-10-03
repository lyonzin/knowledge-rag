"""Resident memory benchmarks; RSS bytes are exported separately from latency."""

from __future__ import annotations

import gc
import statistics
import threading

import pytest

psutil = pytest.importorskip("psutil")
MIB = 1024 * 1024


def _measure_rss(operation):
    """Sample RSS while retaining the workload result until the last sample."""
    process = psutil.Process()
    gc.collect()
    before = process.memory_info().rss
    peak = [before]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.005):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        live = operation()
        peak[0] = max(peak[0], process.memory_info().rss)
        del live
        gc.collect()
        retained = process.memory_info().rss
    finally:
        stop.set()
        sampler.join(timeout=2)
    return {
        "peak_delta_bytes": max(0, peak[0] - before),
        "retained_delta_bytes": retained - before,
        "baseline_bytes": before,
        "peak_bytes": peak[0],
    }


def _benchmark_memory(benchmark, operation, budget_bytes):
    samples = []

    def measure():
        measured = _measure_rss(operation)
        samples.append(measured)
        return measured["peak_delta_bytes"]

    benchmark.pedantic(measure, iterations=1, rounds=5)
    benchmark.extra_info.update(
        {
            "measurement": "rss_peak_delta_bytes",
            "unit": "bytes",
            "workload_version": 2,
            "rss_peak_delta_bytes": statistics.median(sample["peak_delta_bytes"] for sample in samples),
            "rss_retained_delta_bytes": statistics.median(sample["retained_delta_bytes"] for sample in samples),
            "rss_budget_bytes": budget_bytes,
            "rss_samples": samples,
            "rss_sampling_interval_ms": 5,
        }
    )
    assert max(sample["peak_delta_bytes"] for sample in samples) < budget_bytes, samples


def test_bench_orchestrator_idle_rss(benchmark, fake_embed_fn):
    """Fifty lazy embedders must fit a fixed RSS allocation budget."""
    from mcp_server.server import FastEmbedEmbeddings

    _benchmark_memory(benchmark, lambda: [FastEmbedEmbeddings() for _ in range(50)], 50 * MIB)


def test_bench_query_cache_5000_entries(benchmark):
    """Measure actual RSS of bounded cache churn, not its runtime as memory."""
    from mcp_server.server import QueryCache

    def fill_cache():
        cache = QueryCache(max_size=1000, ttl_seconds=300)
        for index in range(5000):
            cache.put(f"q-{index}", 5, None, 0.3, [{"content": f"{index}:" + "x" * 100}])
        return cache

    _benchmark_memory(benchmark, fill_cache, 80 * MIB)
