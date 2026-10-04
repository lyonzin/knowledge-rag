"""Chroma permits metadata=None for individual results in query/get responses."""

import pytest

from mcp_server import server


def add_unlabelled(orch):
    orch.collection.add(ids=["unlabelled_0"], documents=["original production content"], embeddings=[[1.0, 1.0]])


@pytest.mark.parametrize("hybrid_alpha", [0.0, 0.5, 1.0])
def test_unlabelled_chroma_record_remains_searchable(isolated_orchestrator, monkeypatch, hybrid_alpha):
    orch = isolated_orchestrator
    add_unlabelled(orch)
    monkeypatch.setattr(server.config, "reranker_enabled", False)
    results = orch.query("original", 1, hybrid_alpha=hybrid_alpha)
    assert results[0]["content"] == "original production content"
    assert results[0]["source"] == ""


def test_keyword_category_filter_keeps_labelled_hits_beside_unlabelled_hits(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    add_unlabelled(orch)
    orch.collection.add(
        ids=["labelled_0"],
        documents=["original labelled document"],
        embeddings=[[1.0, 1.0]],
        metadatas=[{"category": "general"}],
    )
    monkeypatch.setattr(server.config, "reranker_enabled", False)
    results = orch.query("original", 2, category_filter="general", hybrid_alpha=0.0)
    assert [result["content"] for result in results] == ["original labelled document"]


def test_fts5_hydration_accepts_unlabelled_chroma_record(isolated_orchestrator):
    orch = isolated_orchestrator
    add_unlabelled(orch)
    results = orch._format_fts5_results([("unlabelled_0", 1.0)], 1, None)
    assert results[0]["content"] == "original production content"
    assert results[0]["source"] == ""


def test_similarity_query_accepts_unlabelled_neighbor(isolated_orchestrator):
    orch = isolated_orchestrator
    add_unlabelled(orch)
    reference = str((server.config.documents_dir / "reference.txt").resolve())
    orch._source_to_docid[reference] = "reference"
    orch.collection.add(
        ids=["reference_0"],
        documents=["original reference document"],
        embeddings=[[1.0, 1.0]],
        metadatas=[{"doc_id": "reference", "source": reference}],
    )
    results = orch.search_similar(reference, 1)
    assert results[0]["preview"] == "original production content"
