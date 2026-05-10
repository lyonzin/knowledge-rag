"""Chaos: HuggingFace Hub unreachable (503 / timeout).

If HF Hub returns errors during model construction, FastEmbedEmbeddings
must raise EmbeddingModelLoadError rather than hang or fail silently.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.chaos


def test_hf_503_during_load_raises_loud():
    """HF Hub 503 mid-construction must produce a loud failure."""
    from mcp_server.server import EmbeddingModelLoadError, FastEmbedEmbeddings

    with patch(
        "mcp_server.server.TextEmbedding",
        side_effect=ConnectionError("HF Hub returned HTTP 503"),
    ):
        emb = FastEmbedEmbeddings()
        with pytest.raises(EmbeddingModelLoadError):
            emb(["query during outage"])


def test_hf_timeout_does_not_leave_partial_state():
    """A timeout during load must NOT leave _model partially set."""
    from mcp_server.server import EmbeddingModelLoadError, FastEmbedEmbeddings

    with patch(
        "mcp_server.server.TextEmbedding",
        side_effect=TimeoutError("HF Hub download timed out"),
    ):
        emb = FastEmbedEmbeddings()
        with pytest.raises(EmbeddingModelLoadError):
            emb(["query"])
        # After failure, _model must remain None (not a partial object)
        assert emb._model is None
        assert emb._load_failed is not None
