"""Tests for lazy initialization + loud-fail behavior of FastEmbedEmbeddings.

v3.8.0 introduced lazy loading: the ~200MB ONNX model is constructed on the
first call instead of in __init__. v3.8.1 added LOUD failures: previously
this class swallowed any exception and returned vectors of zeros, which
silently corrupted the index. Both behaviors are tested here.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedder():
    """Build a FastEmbedEmbeddings instance without triggering ONNX load."""
    from mcp_server.server import FastEmbedEmbeddings

    return FastEmbedEmbeddings()


def _fake_model_returning(dim: int):
    """A MagicMock TextEmbedding that returns one ``dim``-D vector per input."""
    fake = MagicMock()
    fake.embed.side_effect = lambda texts: iter([np.zeros(dim, dtype=np.float32) for _ in texts])
    return fake


# ---------------------------------------------------------------------------
# Lazy-load behavior (v3.8.0)
# ---------------------------------------------------------------------------


def test_init_does_not_load_model():
    """Constructing the embedder must not touch TextEmbedding (no model load)."""
    with patch("mcp_server.server.TextEmbedding") as mock_te:
        emb = _make_embedder()
        assert emb._model is None
        mock_te.assert_not_called()


def test_first_call_loads_model():
    """The first __call__ with non-empty input must materialize the model once."""
    with patch("mcp_server.server.TextEmbedding", return_value=_fake_model_returning(384)) as mock_te:
        emb = _make_embedder()
        assert emb._model is None
        emb([])  # empty short-circuits before load
        assert emb._model is None
        emb(["hello"])
        assert emb._model is not None
        assert mock_te.call_count == 1


def test_second_call_reuses_model():
    """Subsequent calls must NOT reload the model (idempotent _load_model)."""
    with patch("mcp_server.server.TextEmbedding", return_value=_fake_model_returning(384)) as mock_te:
        emb = _make_embedder()
        emb(["one"])
        emb(["two"])
        emb(["three"])
        assert mock_te.call_count == 1


def test_embed_query_triggers_load():
    """embed_query path must also trigger lazy load."""
    with patch("mcp_server.server.TextEmbedding", return_value=_fake_model_returning(384)) as mock_te:
        emb = _make_embedder()
        assert emb._model is None
        emb.embed_query("query text")
        assert mock_te.call_count == 1


def test_embed_documents_triggers_load():
    """embed_documents path must also trigger lazy load."""
    with patch("mcp_server.server.TextEmbedding", return_value=_fake_model_returning(384)) as mock_te:
        emb = _make_embedder()
        assert emb._model is None
        emb.embed_documents(["doc"])
        assert mock_te.call_count == 1


def test_concurrent_first_call_loads_once():
    """Two threads racing on the first call must trigger exactly ONE load."""
    init_started = threading.Event()
    release_init = threading.Event()
    fake_model = _fake_model_returning(384)

    def slow_init(**kwargs):
        init_started.set()
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
        assert init_started.wait(timeout=2), "first thread never entered slow_init"
        t2.start()
        threading.Event().wait(0.05)
        release_init.set()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert mock_te.call_count == 1
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Loud-fail behavior (v3.8.1)
# ---------------------------------------------------------------------------


def test_load_failure_raises_loud():
    """A model that cannot be constructed must raise EmbeddingModelLoadError.

    Regression for v3.8.0 silent corruption: previously the embedder swallowed
    load failures, returned zero vectors, and ChromaDB happily stored garbage.
    """
    from mcp_server.server import EmbeddingModelLoadError

    with patch("mcp_server.server.TextEmbedding", side_effect=FileNotFoundError("model_optimized.onnx")):
        emb = _make_embedder()
        with pytest.raises(EmbeddingModelLoadError) as ei:
            emb(["hello"])
        # Original cause preserved for diagnostics
        assert isinstance(ei.value.__cause__, FileNotFoundError)


def test_load_failure_is_sticky():
    """After a load failure, subsequent calls re-raise WITHOUT retrying the load.

    Avoids the user-visible "frozen" behavior where each query triggers a
    fresh HuggingFace download attempt that hits rate limits.
    """
    from mcp_server.server import EmbeddingModelLoadError

    with patch("mcp_server.server.TextEmbedding", side_effect=FileNotFoundError("model_optimized.onnx")) as mock_te:
        emb = _make_embedder()
        with pytest.raises(EmbeddingModelLoadError):
            emb(["a"])
        with pytest.raises(EmbeddingModelLoadError):
            emb(["b"])
        with pytest.raises(EmbeddingModelLoadError):
            emb(["c"])
        # First call attempted the load; subsequent calls did NOT retry
        assert mock_te.call_count == 1


def test_embed_runtime_failure_raises_loud():
    """Model loaded but ``embed()`` raises -> EmbeddingError (NOT silent zeros)."""
    from mcp_server.server import EmbeddingError

    fake = MagicMock()
    fake.embed.side_effect = RuntimeError("ONNX runtime crashed")
    with patch("mcp_server.server.TextEmbedding", return_value=fake):
        emb = _make_embedder()
        with pytest.raises(EmbeddingError):
            emb(["text"])


def test_embed_count_mismatch_raises():
    """Model returns wrong number of vectors -> EmbeddingError."""
    from mcp_server.server import EmbeddingError

    fake = MagicMock()
    fake.embed.side_effect = lambda texts: iter([np.zeros(384, dtype=np.float32)])  # always 1
    with patch("mcp_server.server.TextEmbedding", return_value=fake):
        emb = _make_embedder()
        with pytest.raises(EmbeddingError, match="count mismatch"):
            emb(["a", "b", "c"])


def test_embed_dim_mismatch_raises():
    """Model returns wrong dimensionality -> EmbeddingError."""
    from mcp_server.server import EmbeddingError

    fake = _fake_model_returning(128)  # config expects 384
    with patch("mcp_server.server.TextEmbedding", return_value=fake):
        emb = _make_embedder()
        with pytest.raises(EmbeddingError, match="dim mismatch"):
            emb(["text"])


def test_empty_input_does_not_trigger_load_or_raise():
    """Empty input must short-circuit before model load (cheap no-op)."""
    with patch("mcp_server.server.TextEmbedding", side_effect=FileNotFoundError("would crash")) as mock_te:
        emb = _make_embedder()
        result = emb([])
        assert result == []
        mock_te.assert_not_called()


def test_does_not_return_zero_vectors_silently():
    """Pure regression guard: under no failure path do we ever return zeros silently.

    The whole point of v3.8.1 — if anything goes wrong, the caller MUST see an
    exception. Returning ``[[0.0]*dim, ...]`` was the bug.
    """
    from mcp_server.server import EmbeddingError, EmbeddingModelLoadError

    # Path 1: load fails
    with patch("mcp_server.server.TextEmbedding", side_effect=Exception("boom")):
        emb = _make_embedder()
        with pytest.raises(EmbeddingModelLoadError):
            emb(["a"])

    # Path 2: load OK, embed fails
    fake = MagicMock()
    fake.embed.side_effect = Exception("boom mid-embed")
    with patch("mcp_server.server.TextEmbedding", return_value=fake):
        emb2 = _make_embedder()
        with pytest.raises(EmbeddingError):
            emb2(["a"])
