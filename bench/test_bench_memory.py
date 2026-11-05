"""Python heap benchmarks: tracemalloc peak bytes, exported separately from latency.

Workload version 2 sampled process RSS and read one 4 KiB page on CI for every
workload. Both workloads allocate under 1 MiB, and CPython's allocators serve that
from memory that earlier benchmarks and rounds already made resident, so RSS only
moves once the heap outgrows its high-water mark. tracemalloc counts every Python
allocation the operation makes, whatever the allocator keeps resident, and gives the
same bytes on every run of the same code.
"""

from __future__ import annotations

import gc
import statistics
import tracemalloc
from collections.abc import Callable

MIB = 1024 * 1024
ROUNDS = 5
WORKLOAD_VERSION = 3


def _trace_allocations(operation: Callable[[], object]) -> dict[str, int]:
    """Return the bytes the operation held at its peak and still holds after cleanup.

    The peak is a running maximum, so it covers the returned object before it is dropped.
    """
    gc.collect()
    tracemalloc.start()
    try:
        operation()
        gc.collect()
        retained, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"peak_bytes": peak, "retained_bytes": retained}


def _benchmark_memory(benchmark, operation: Callable[[], object], budget_bytes: int) -> None:
    """Export the median traced peak for the perf gate and fail above the fixed budget."""
    samples: list[dict[str, int]] = []

    def measure() -> int:
        samples.append(_trace_allocations(operation))
        return samples[-1]["peak_bytes"]

    benchmark.pedantic(measure, iterations=1, rounds=ROUNDS)
    benchmark.extra_info.update(
        {
            "measurement": "tracemalloc_peak_bytes",
            "unit": "bytes",
            "workload_version": WORKLOAD_VERSION,
            "tracemalloc_peak_bytes": statistics.median(sample["peak_bytes"] for sample in samples),
            "tracemalloc_retained_bytes": statistics.median(sample["retained_bytes"] for sample in samples),
            "memory_budget_bytes": budget_bytes,
            "tracemalloc_samples": samples,
        }
    )
    assert max(sample["peak_bytes"] for sample in samples) < budget_bytes, samples


def test_bench_orchestrator_idle_rss(benchmark, fake_embed_fn):
    """Fifty lazy embedders must fit a fixed allocation budget."""
    from mcp_server.server import FastEmbedEmbeddings

    _benchmark_memory(benchmark, lambda: [FastEmbedEmbeddings() for _ in range(50)], 50 * MIB)


def test_bench_query_cache_5000_entries(benchmark):
    """Measure the allocations of bounded cache churn, not its runtime as memory."""
    from mcp_server.server import QueryCache

    def fill_cache():
        cache = QueryCache(max_size=1000, ttl_seconds=300)
        for index in range(5000):
            cache.put(f"q-{index}", 5, None, 0.3, [{"content": f"{index}:" + "x" * 100}])
        return cache

    _benchmark_memory(benchmark, fill_cache, 80 * MIB)
