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

KIB = 1024
MIB = 1024 * KIB
ROUNDS = 5
WORKLOAD_VERSION = 3
IDLE_EMBEDDERS = 50
CACHE_PUTS = 5000
CACHE_MAX_SIZE = 1000
CACHE_TTL_SECONDS = 300
CACHE_PAYLOAD_CHARS = 100
CACHE_MAX_RESULTS = 5
CACHE_HYBRID_ALPHA = 0.3
# About 10x the peaks measured on Python 3.14 (23,752 and 838,808 bytes), leaving room for
# CI's Python 3.12. The gate compares against master; these absolute guards hold even when
# the skip-perf-gate label bypasses it.
IDLE_EMBEDDERS_BUDGET_BYTES = 256 * KIB
CACHE_BUDGET_BYTES = 8 * MIB


def _trace_peak(operation: Callable[[], object]) -> int:
    """Return the most bytes of Python allocations the operation held at once, its result included."""
    if tracemalloc.is_tracing():
        raise RuntimeError(
            "tracemalloc is already tracing (PYTHONTRACEMALLOC or -X tracemalloc): start() would keep the "
            "outer peak and stop() would end the outer trace. Run the memory benchmarks without it."
        )
    # A full collection first makes automatic collections fall at the same points in every round.
    gc.collect()
    tracemalloc.start()
    try:
        operation()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def _benchmark_memory(benchmark, operation: Callable[[], object], budget_bytes: int) -> None:
    """Export the median traced peak for the perf gate and fail any round above the absolute budget."""
    peaks: list[int] = []

    def measure() -> int:
        peaks.append(_trace_peak(operation))
        return peaks[-1]

    benchmark.pedantic(measure, iterations=1, rounds=ROUNDS)
    benchmark.extra_info.update(
        {
            "measurement": "tracemalloc_peak_bytes",
            "unit": "bytes",
            "workload_version": WORKLOAD_VERSION,
            "tracemalloc_peak_bytes": statistics.median(peaks),
            "memory_budget_bytes": budget_bytes,
            "tracemalloc_samples": peaks,
        }
    )
    assert max(peaks) < budget_bytes, peaks


def test_bench_orchestrator_idle_memory(benchmark, fake_embed_fn):
    """Lazy embedders must stay within a fixed allocation budget until first use."""
    from mcp_server.server import FastEmbedEmbeddings

    def create_embedders():
        return [FastEmbedEmbeddings() for _ in range(IDLE_EMBEDDERS)]

    _benchmark_memory(benchmark, create_embedders, IDLE_EMBEDDERS_BUDGET_BYTES)


def test_bench_query_cache_5000_entries(benchmark):
    """Measure the allocations of bounded cache churn, not its runtime as memory."""
    from mcp_server.server import QueryCache

    def fill_cache():
        cache = QueryCache(max_size=CACHE_MAX_SIZE, ttl_seconds=CACHE_TTL_SECONDS)
        for index in range(CACHE_PUTS):
            content = f"{index}:" + "x" * CACHE_PAYLOAD_CHARS
            cache.put(f"q-{index}", CACHE_MAX_RESULTS, None, CACHE_HYBRID_ALPHA, [{"content": content}])
        return cache

    _benchmark_memory(benchmark, fill_cache, CACHE_BUDGET_BYTES)
