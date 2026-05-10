"""Tests for lazy initialization of FastEmbedEmbeddings (v3.8.0+).

The embedding model is heavy (~200MB resident). We defer the load to the
first real call so idle MCP server processes stay cheap. This matters when
multiple stdio clients spawn parallel knowledge-rag processes.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch


def _make_embedder():
    """Build a FastEmbedEmbeddings instance without triggering ONNX load."""
    from mcp_server.server import FastEmbedEmbeddings

    return FastEmbedEmbeddings()


def test_init_does_not_load_model():
    """Constructing the embedder must not touch TextEmbedding (no model load)."""
    with patch("mcp_server.server.TextEmbedding") as mock_te:
        emb = _make_embedder()
        assert emb._model is None
        mock_te.assert_not_called()


def test_first_call_loads_model():
    """The first __call__ must materialize the model exactly once."""
    fake_model = MagicMock()
    fake_model.embed.return_value = iter([])
    with patch("mcp_server.server.TextEmbedding", return_value=fake_model) as mock_te:
        emb = _make_embedder()
        assert emb._model is None
        emb([])  # empty input short-circuits before embedding, but _load is fine to skip
        # Empty input returns early — confirm by triggering with real text
        emb(["hello"])
        assert emb._model is fake_model
        assert mock_te.call_count == 1


def test_second_call_reuses_model():
    """Subsequent calls must NOT reload the model (idempotent _load_model)."""
    fake_model = MagicMock()
    fake_model.embed.side_effect = lambda texts: iter([])
    with patch("mcp_server.server.TextEmbedding", return_value=fake_model) as mock_te:
        emb = _make_embedder()
        emb(["one"])
        emb(["two"])
        emb(["three"])
        assert mock_te.call_count == 1


def test_embed_query_triggers_load():
    """embed_query path must also trigger lazy load."""
    fake_model = MagicMock()
    fake_model.embed.return_value = iter([])
    with patch("mcp_server.server.TextEmbedding", return_value=fake_model) as mock_te:
        emb = _make_embedder()
        assert emb._model is None
        emb.embed_query("query text")
        assert mock_te.call_count == 1


def test_embed_documents_triggers_load():
    """embed_documents path must also trigger lazy load."""
    fake_model = MagicMock()
    fake_model.embed.return_value = iter([])
    with patch("mcp_server.server.TextEmbedding", return_value=fake_model) as mock_te:
        emb = _make_embedder()
        assert emb._model is None
        emb.embed_documents(["doc"])
        assert mock_te.call_count == 1


def test_concurrent_first_call_loads_once():
    """Two threads racing on the first call must trigger exactly ONE load.

    The lock-protected _load_model is single-entry, so even a slow model
    construction won't cause a duplicate init. We hold the first init briefly
    to ensure the second thread arrives during the load and observes the lock.
    """
    fake_model = MagicMock()
    fake_model.embed.return_value = iter([])
    init_started = threading.Event()
    release_init = threading.Event()

    def slow_init(**kwargs):
        init_started.set()
        # Block here so the second thread has time to race on the lock
        release_init.wait(timeout=2)
        return fake_model

    with patch("mcp_server.server.TextEmbedding", side_effect=slow_init) as mock_te:
        emb = _make_embedder()
        results = []

        def worker():
            emb(["text"])
            results.append(True)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        # Wait for thread 1 to be inside slow_init (holding the lock)
        assert init_started.wait(timeout=2), "first thread never entered slow_init"
        t2.start()
        # Give t2 a moment to hit the lock
        threading.Event().wait(0.05)
        # Release t1's init; t2 should then exit fast via the double-checked guard
        release_init.set()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Even with two threads racing, the model is constructed exactly once
        assert mock_te.call_count == 1
        assert len(results) == 2
