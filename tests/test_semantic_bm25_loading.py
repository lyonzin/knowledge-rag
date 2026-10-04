"""Semantic-only searches must not hydrate the corpus into a BM25 index."""

from unittest.mock import Mock

import pytest

from mcp_server import server


@pytest.mark.parametrize("hybrid_alpha", [0.0, 0.5, 1.0])
def test_query_loads_bm25_only_when_keyword_retrieval_is_needed(isolated_orchestrator, monkeypatch, hybrid_alpha):
    orch = isolated_orchestrator
    orch.collection.add(
        ids=["doc_0"],
        documents=["original production content"],
        embeddings=[[1.0, 1.0]],
        metadatas=[{"source": "one.txt"}],
    )
    monkeypatch.setattr(server.config, "reranker_enabled", False)
    ensure = Mock(wraps=orch._ensure_bm25_index)
    monkeypatch.setattr(orch, "_ensure_bm25_index", ensure)

    results = orch.query("original", 1, hybrid_alpha=hybrid_alpha)

    assert results[0]["content"] == "original production content"
    if hybrid_alpha < 1.0:
        ensure.assert_called_once_with()
        assert orch._bm25_initialized
    else:
        ensure.assert_not_called()
        assert not orch._bm25_initialized
        assert orch.bm25_index.search("original", top_k=1) == []


def test_semantic_only_crud_does_not_accumulate_unpublished_bm25_versions(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    monkeypatch.setattr(server.config, "reranker_enabled", False)
    ensure = Mock(wraps=orch._ensure_bm25_index)
    monkeypatch.setattr(orch, "_ensure_bm25_index", ensure)
    orch.add_document_from_content("initial searchable text", "revision.md", "general")
    for revision in range(31):
        orch.update_document_content("revision.md", f"revisionmarker{revision} current searchable content")
        assert orch.query(f"revisionmarker{revision}", 1, hybrid_alpha=1.0)
    ensure.assert_not_called()
    assert orch.bm25_index.corpus == []
    assert orch.bm25_index._token_pool == {}
    assert orch.query("revisionmarker30", 1, hybrid_alpha=0)
    assert len(orch.bm25_index.corpus) == orch.collection.count()
    assert "revisionmarker0 " not in " ".join(orch.bm25_index.corpus)
