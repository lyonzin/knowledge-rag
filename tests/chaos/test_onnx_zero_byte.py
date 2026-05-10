"""Chaos: ONNX model file zeroed (replays the v3.8.1 bug).

If a HF download is interrupted the .onnx file ends up 0-bytes on disk.
fastembed then loads "successfully" but the resulting TextEmbedding
raises NO_SUCHFILE the moment we try to embed.

Pre-v3.8.1 this was silently swallowed and the code returned zero
vectors, corrupting ChromaDB. Post-v3.8.1 the code MUST raise
EmbeddingModelLoadError loudly and stick on _load_failed so subsequent
calls do not hammer HuggingFace.

This nightly test guards against that regression class.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.chaos


def test_onnx_zero_byte_raises_loud_no_silent_zeros():
    """Simulate the exact production failure mode from v3.8.0."""
    from mcp_server.server import EmbeddingModelLoadError, FastEmbedEmbeddings

    # fastembed raises FileNotFoundError when ONNX file is missing/zero
    with patch(
        "mcp_server.server.TextEmbedding",
        side_effect=FileNotFoundError("model_optimized.onnx"),
    ):
        emb = FastEmbedEmbeddings()
        with pytest.raises(EmbeddingModelLoadError):
            emb(["any query"])

        # Sticky failure: subsequent calls re-raise WITHOUT retrying
        with pytest.raises(EmbeddingModelLoadError):
            emb(["another query"])
        with pytest.raises(EmbeddingModelLoadError):
            emb(["yet another"])


def test_onnx_runtime_failure_does_not_return_silent_zeros():
    """If embed() crashes mid-operation, caller must see EmbeddingError."""
    from unittest.mock import MagicMock

    from mcp_server.server import EmbeddingError, FastEmbedEmbeddings

    fake = MagicMock()
    fake.embed.side_effect = RuntimeError("ONNX runtime corrupted state")

    with patch("mcp_server.server.TextEmbedding", return_value=fake):
        emb = FastEmbedEmbeddings()
        with pytest.raises(EmbeddingError):
            emb(["text"])
