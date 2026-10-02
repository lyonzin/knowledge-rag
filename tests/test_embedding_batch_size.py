"""Regression tests for v4.9.3 CPU-OOM fix (GH #224).

FastEmbed ``TextEmbedding.embed()`` default ``batch_size=256`` allocates
~500MB intermediate buffers under CPU execution with bge-small at
``seq_len=512``. On fragmented Windows heaps this triggered ONNXRuntime
``BAD_ALLOC`` during ingestion of minified JS or large JSON.

The fix in ``mcp_server/server.py:FastEmbedEmbeddings._embed`` dispatches
``batch_size`` based on the active ONNX provider captured in
``_load_with_providers``:

- CPU  → ``batch_size=32``   (safe on fragmented Windows heap)
- CUDA → ``batch_size=256``  (unchanged, VRAM is contiguous)

Two overrides exist with the following precedence (highest first):

1. ``KNOWLEDGE_RAG_EMBED_BATCH_SIZE`` environment variable
2. ``config.embed_batch_size`` (YAML key ``documents.embed_batch_size``)
3. Provider heuristic (32 for CPU, 256 for CUDA)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np


def _fake_text_embedding_factory():
    """Return a (fake_class, captured) pair.

    ``fake_class`` is a drop-in for ``mcp_server.server.TextEmbedding``:
    each instance exposes an ``embed(texts, batch_size=256)`` method that
    records the ``batch_size`` kwarg into the shared ``captured`` dict and
    yields zero vectors.
    """
    captured: dict[str, int] = {}

    class _Fake:
        def __init__(self, **kwargs):
            # Signature is compatible with TextEmbedding(**kwargs). We don't
            # care about the init args — only the embed() call shape matters.
            self._kwargs = kwargs

        def embed(self, texts, batch_size=256):
            captured["batch_size"] = batch_size
            return iter([np.zeros(384, dtype=np.float32) for _ in texts])

    return _Fake, captured


def _make_embedder():
    from mcp_server.server import FastEmbedEmbeddings

    return FastEmbedEmbeddings()


def test_cpu_provider_uses_small_batch_size(monkeypatch):
    """CPU inference must cap batch_size at 32 to avoid OOM on fragmented heaps."""
    monkeypatch.delenv("KNOWLEDGE_RAG_EMBED_BATCH_SIZE", raising=False)
    from mcp_server.server import config

    monkeypatch.setattr(config, "embed_batch_size", None)

    fake_cls, captured = _fake_text_embedding_factory()
    with patch("mcp_server.server.TextEmbedding", fake_cls):
        emb = _make_embedder()
        # Bypass the real GPU probe + provider routing. Just prime the
        # internal state as if _load_with_providers had chosen CPU.
        emb._model = fake_cls()
        emb._active_provider = "CPU"
        emb(["hello world"] * 10)
    assert captured["batch_size"] == 32


def test_gpu_provider_uses_large_batch_size(monkeypatch):
    """GPU inference keeps batch_size=256 to maximize VRAM throughput."""
    monkeypatch.delenv("KNOWLEDGE_RAG_EMBED_BATCH_SIZE", raising=False)
    from mcp_server.server import config

    monkeypatch.setattr(config, "embed_batch_size", None)

    fake_cls, captured = _fake_text_embedding_factory()
    with patch("mcp_server.server.TextEmbedding", fake_cls):
        emb = _make_embedder()
        emb._model = fake_cls()
        emb._active_provider = "CUDA"
        emb(["hello world"] * 10)
    assert captured["batch_size"] == 256


def test_env_override_wins_over_provider_default(monkeypatch):
    """``KNOWLEDGE_RAG_EMBED_BATCH_SIZE`` env var overrides the provider heuristic."""
    monkeypatch.setenv("KNOWLEDGE_RAG_EMBED_BATCH_SIZE", "16")
    from mcp_server.server import config

    # YAML config stays unset — env var must still win.
    monkeypatch.setattr(config, "embed_batch_size", None)

    fake_cls, captured = _fake_text_embedding_factory()
    with patch("mcp_server.server.TextEmbedding", fake_cls):
        emb = _make_embedder()
        emb._model = fake_cls()
        emb._active_provider = "CPU"
        emb(["hello"] * 5)
    assert captured["batch_size"] == 16
