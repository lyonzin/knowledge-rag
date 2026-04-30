"""Tests for reranker fallback behavior."""

from unittest.mock import patch


def test_reranker_load_failure_returns_rrf_order(monkeypatch):
    """A missing offline reranker model must not fail search results."""
    from mcp_server.server import CrossEncoderReranker

    monkeypatch.setattr("mcp_server.server.config.reranker_enabled", True)

    docs = [
        {"document": "first", "rrf_score": 0.2},
        {"document": "second", "rrf_score": 0.1},
    ]
    reranker = CrossEncoderReranker()

    with patch("mcp_server.server.TextCrossEncoder", side_effect=RuntimeError("offline")):
        assert reranker.rerank("query", docs, top_k=1) == [docs[0]]
        assert reranker._load_failed is True


def test_reranker_does_not_retry_after_load_failure(monkeypatch):
    """Avoid repeated network attempts after a known load failure."""
    from mcp_server.server import CrossEncoderReranker

    monkeypatch.setattr("mcp_server.server.config.reranker_enabled", True)

    reranker = CrossEncoderReranker()
    reranker._load_failed = True

    with patch("mcp_server.server.TextCrossEncoder") as text_cross_encoder:
        result = reranker.rerank("query", [{"document": "first"}], top_k=1)

    assert result == [{"document": "first"}]
    text_cross_encoder.assert_not_called()
