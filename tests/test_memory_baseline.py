"""Memory leak regression tests.

Pillar 3: Memory Leak — guarantee that long-lived knowledge-rag processes
do not accumulate RSS beyond budget when servicing repeated queries.

The thresholds are deliberately loose to avoid flakes from GC scheduling
on shared CI runners. They are tight enough to catch a real leak (a
forgotten append-to-list, a cache without eviction) within a few hundred
iterations.

We use ``psutil.Process().memory_info().rss`` instead of pytest-memray
because:
- pytest-memray does not support Windows in the official pyproject
  matrix; we run on win32 too.
- We do not need allocation tracebacks here — we need a coarse RSS
  ceiling regression test.

These tests intentionally avoid loading the real ONNX model (heavy +
slow). They mock TextEmbedding / TextCrossEncoder so the assertions
target the orchestrator + cache + watcher state, not the embedding
runtime itself.
"""

from __future__ import annotations

import gc
from unittest.mock import MagicMock, patch

import pytest

psutil = pytest.importorskip("psutil", reason="psutil required for memory baseline tests")
import numpy as np  # noqa: E402  — imported after the importorskip guard

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_text_embedding():
    """Patch the real TextEmbedding with a lightweight MagicMock.

    The mock returns a fresh 384-D zero vector per input — same shape as
    the production model, but without the ~200MB ONNX runtime load.
    """
    fake = MagicMock()
    fake.embed.side_effect = lambda texts: iter([np.zeros(384, dtype=np.float32) for _ in texts])
    with patch("mcp_server.server.TextEmbedding", return_value=fake):
        yield fake


def _rss_mb() -> float:
    """Return current process resident set size in megabytes."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


# ---------------------------------------------------------------------------
# Orchestrator idle footprint
# ---------------------------------------------------------------------------


def test_lazy_load_only_loads_once_under_pressure():
    """100 sequential embed calls must construct the model exactly once.

    Regression test for the original lazy-load contract: a leaking
    initialization path could re-instantiate TextEmbedding every call,
    inflating RSS linearly with iteration count.
    """
    fake = MagicMock()
    fake.embed.side_effect = lambda texts: iter([np.zeros(384, dtype=np.float32) for _ in texts])

    with patch("mcp_server.server.TextEmbedding", return_value=fake) as mock_te:
        from mcp_server.server import FastEmbedEmbeddings

        emb = FastEmbedEmbeddings()
        assert emb._model is None

        for _ in range(100):
            emb(["sample query"])

        # 100 embed calls; TextEmbedding constructed exactly once
        assert mock_te.call_count == 1, (
            f"FastEmbedEmbeddings constructed TextEmbedding {mock_te.call_count} times across 100 calls — "
            "lazy-load is leaking re-initializations."
        )


def test_search_no_leak_after_1000_queries(fake_text_embedding):
    """N calls into the embedder must NOT accumulate RSS unboundedly.

    Loose threshold (40 MB per 1000 iters baseline) catches genuine
    leaks without false-positiving on GC jitter. The nightly soak test
    overrides ``KNOWLEDGE_RAG_SOAK_ITERATIONS`` to push the loop to
    50000 calls — same threshold scaled proportionally.
    """
    import os

    from mcp_server.server import FastEmbedEmbeddings

    iterations = int(os.environ.get("KNOWLEDGE_RAG_SOAK_ITERATIONS", "1000"))
    # 40 MB per 1000 iters is the regression budget; scale linearly.
    threshold_mb = max(40, 40 * (iterations / 1000.0))

    emb = FastEmbedEmbeddings()

    # Warm up — force first model load + populate any caches
    emb(["warmup"])
    gc.collect()
    initial_rss = _rss_mb()

    for i in range(iterations):
        emb([f"query {i}"])

    gc.collect()
    final_rss = _rss_mb()
    delta = final_rss - initial_rss

    assert delta < threshold_mb, (
        f"RSS grew {delta:.1f} MB across {iterations} embed calls "
        f"(initial={initial_rss:.1f}, final={final_rss:.1f}, "
        f"threshold={threshold_mb:.1f}). Suggests a leak — bisect the growth."
    )


def test_query_cache_bounded(fake_text_embedding):
    """The orchestrator's QueryCache must enforce its max_size.

    Pumping more entries than max_size must NOT grow the cache beyond
    that bound. A regression here means LRU eviction is broken.
    """
    from mcp_server.server import QueryCache

    cache = QueryCache(max_size=50, ttl_seconds=300)
    for i in range(500):
        cache.put(f"q-{i}", 5, None, 0.3, [{"content": f"r-{i}"}])

    stats = cache.stats()
    assert stats["size"] <= 50, f"QueryCache exceeded max_size: size={stats['size']}"


def test_orchestrator_idle_no_explosive_alloc(fake_text_embedding):
    """Just constructing the embedder must not pre-allocate large buffers.

    Threshold is loose — we just want to catch obvious regressions where
    something accidentally reads N MB at __init__.
    """
    gc.collect()
    before = _rss_mb()

    from mcp_server.server import FastEmbedEmbeddings

    embedders = [FastEmbedEmbeddings() for _ in range(20)]

    gc.collect()
    after = _rss_mb()
    delta = after - before

    # 20 lazy embedders should be near-free (no model loaded yet)
    assert delta < 30, f"20 idle FastEmbedEmbeddings instances added {delta:.1f} MB — check that __init__ remains lazy."

    # Keep references alive to defeat GC during the assertion
    assert all(e._model is None for e in embedders)
