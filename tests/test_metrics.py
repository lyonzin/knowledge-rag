"""UT-039, UT-040 — ``MetricsCollector`` extensions for the FTS5 fast-path.

Covers the two behaviors Task 03 introduces to the collector:
- string-labeled counter accumulation across multiple ``.inc()`` calls
- bucketed histogram exposition with a ``+Inf`` cumulative count
"""

from __future__ import annotations

from mcp_server.metrics import (
    FAST_PATH_HITS_TOTAL,
    FAST_PATH_LATENCY_BUCKETS,
    FAST_PATH_LATENCY_SECONDS,
    MetricsCollector,
)


def _fresh_collector():
    """Return a collector pre-registered for the FTS5 latency histogram."""
    collector = MetricsCollector()
    collector.register_histogram_buckets(FAST_PATH_LATENCY_SECONDS, FAST_PATH_LATENCY_BUCKETS)
    return collector


def test_ut039_counter_inc_accumulates_per_label():
    """UT-039: three inc() calls on the same labeled counter produce '... 3'."""
    collector = _fresh_collector()
    for _ in range(3):
        collector.inc(FAST_PATH_HITS_TOTAL, '{path="fts5"}')
    exposition = collector.exposition()
    assert f'{FAST_PATH_HITS_TOTAL}{{path="fts5"}} 3' in exposition


def test_ut040_histogram_exposition_includes_inf_bucket_count():
    """UT-040: 10 observations must produce a +Inf bucket with count 10."""
    collector = _fresh_collector()
    for _ in range(10):
        collector.observe(FAST_PATH_LATENCY_SECONDS, 0.007)
    exposition = collector.exposition()
    assert f'{FAST_PATH_LATENCY_SECONDS}_bucket{{le="+Inf"}} 10' in exposition
    # Sanity: the finite buckets accumulate cumulatively as expected.
    # 0.007s falls into every bucket >= 0.010s.
    assert f'{FAST_PATH_LATENCY_SECONDS}_bucket{{le="0.01"}} 10' in exposition
    assert f'{FAST_PATH_LATENCY_SECONDS}_bucket{{le="0.005"}} 0' in exposition
