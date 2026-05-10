"""Bench-specific fixtures.

We pin shared mocks here so each benchmark file stays focused on the
metric it measures, not the embedding model boilerplate.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def fake_embed_fn():
    """Return a callable that mimics FastEmbedEmbeddings without ONNX cost.

    Bench measures orchestration cost (BM25, RRF, cache, watcher) — not
    the embedding kernel — so we replace the heavy model with a constant
    384-D zero vector. Ensures bench numbers are stable across runs and
    do not depend on the host CPU's matmul speed.
    """
    fake = MagicMock()
    fake.embed.side_effect = lambda texts: iter([np.zeros(384, dtype=np.float32) for _ in texts])
    with patch("mcp_server.server.TextEmbedding", return_value=fake):
        yield fake
