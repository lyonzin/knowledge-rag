"""Real Chroma arrays and mutable cached results exercise retrieval boundaries."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import chromadb
import pytest

import mcp_server.server as server
from mcp_server.fts5_index import Fts5LexicalIndex


def test_cache_isolates_mutation_without_copying_immutable_document_text():
    content = "documentation " * 2_000
    original = [{"content": content, "keywords": ["alpha"]}]
    cache = server.QueryCache()
    cache.put("query", 5, None, 0.3, original)
    original[0]["keywords"].append("not cached")
    cached = cache.get("query", 5, None, 0.3)
    assert cached[0]["content"] is content
    assert cached[0]["keywords"] == ["alpha"]
    cached[0]["content"] = "snippet"
    cached[0]["keywords"].clear()
    assert cache.get("query", 5, None, 0.3) == [{"content": content, "keywords": ["alpha"]}]


def test_snippet_response_preserves_full_response_on_same_cached_query(monkeypatch):
    content = "documentation " * 2_000
    cache = server.QueryCache()
    cache.put("query", 5, None, 0.3, [{"content": content, "score": 1.0}])
    orchestrator = SimpleNamespace(query=lambda *_, **__: cache.get("query", 5, None, 0.3), query_cache=cache)
    monkeypatch.setattr(server, "get_orchestrator", lambda: orchestrator)
    snippet = json.loads(server.search_knowledge("query", snippet_mode=True))
    full = json.loads(server.search_knowledge("query", snippet_mode=False))
    assert len(snippet["results"][0]["content"]) <= 503
    assert snippet["results"][0]["content_length"] == len(content)
    assert full["results"][0]["content"] == content
    assert "content_length" not in full["results"][0]


def _index_guide_matching_only_in_its_middle_chunk(orch, monkeypatch):
    """Index three markdown sections, one chunk each; only the middle one holds ``needleterm``."""
    monkeypatch.setattr(server.config, "reranker_enabled", False)
    monkeypatch.setattr(server, "get_orchestrator", lambda: orch)
    sections = [
        "## Before\n\n" + "preface " * 80,
        "## Match\n\nneedleterm " + "detail " * 80,
        "## After\n\n" + "ending " * 80,
    ]
    (server.config.documents_dir / "guide.md").write_text("\n\n".join(sections), encoding="utf-8")
    assert orch.index_all()["errors"] == 0


def test_snippet_starts_at_the_matched_chunk_when_context_comes_before_it(isolated_orchestrator, monkeypatch):
    """Given a match in a middle chunk, the default snippet shows that chunk, not the one before it (#260)."""
    _index_guide_matching_only_in_its_middle_chunk(isolated_orchestrator, monkeypatch)
    full = json.loads(server.search_knowledge("needleterm", hybrid_alpha=0.0, snippet_mode=False))["results"][0]
    snippet = json.loads(server.search_knowledge("needleterm", hybrid_alpha=0.0))["results"][0]  # QueryCache hit
    assert snippet["content"].startswith("## Match\n\nneedleterm")
    assert snippet["content_length"] == len(full["content"])


def test_fts5_snippet_starts_at_the_matched_chunk(isolated_orchestrator, monkeypatch, tmp_path):
    """Given the opt-in FTS5 index, its snippet also starts at the matched chunk (#260)."""
    orch = isolated_orchestrator
    monkeypatch.setattr(server.config, "fts5_enabled", True)
    orch.fts5_index = Fts5LexicalIndex(tmp_path / "fts5.db", tmp_path / "fts5.state")
    orch.fts5_index._write_state("complete", 0, 0, "", None, None)
    _index_guide_matching_only_in_its_middle_chunk(orch, monkeypatch)  # chunks sync into FTS5 as they index
    hit = json.loads(server.search_knowledge("needleterm", search_method="fts5"))["results"][0]
    assert hit["search_method"] == "fts5"
    assert hit["content"].startswith("## Match\n\nneedleterm")


def test_replacing_cached_key_at_capacity_keeps_other_entry():
    cache = server.QueryCache(max_size=2)
    cache.put("a", 5, None, 0.3, ["a"])
    cache.put("b", 5, None, 0.3, ["b"])
    cache.put("b", 5, None, 0.3, ["new b"])
    assert cache.get("a", 5, None, 0.3) == ["a"]
    assert cache.get("b", 5, None, 0.3) == ["new b"]


@pytest.mark.parametrize("relative", [False, True])
def test_search_similar_accepts_real_chroma_ndarrays_and_document_relative_paths(tmp_path, monkeypatch, relative):
    documents = tmp_path / "documents"
    documents.mkdir()
    reference = documents / "reference.md"
    neighbor = documents / "neighbor.md"
    monkeypatch.setattr(server.config, "documents_dir", documents)
    client = chromadb.EphemeralClient()
    collection = client.create_collection(f"similar-{tmp_path.name}", embedding_function=None)
    collection.add(
        ids=["reference_0", "neighbor_0"],
        embeddings=[[1.0, 0.0, 0.0], [0.98, 0.01, 0.01]],
        documents=["reference content", "related content"],
        metadatas=[{"source": str(reference), "doc_id": "reference"}, {"source": str(neighbor), "doc_id": "neighbor"}],
    )
    try:
        orchestrator = object.__new__(server.KnowledgeOrchestrator)
        orchestrator.collection = collection
        orchestrator._source_to_docid = {str(reference.resolve()): "reference"}
        path = "reference.md" if relative else str(reference)
        hits = orchestrator.search_similar(path, max_results=1)
        assert len(hits) == 1
        assert hits[0]["source"] == str(neighbor)
    finally:
        client.delete_collection(collection.name)


def test_search_similar_returns_neighbors_of_a_document_with_many_similar_chunks(isolated_orchestrator):
    """Given a reference whose own chunks outrank every neighbor, the neighbors still come back (#260)."""
    orch = isolated_orchestrator
    reference = str((server.config.documents_dir / "reference.md").resolve())
    neighbor = str((server.config.documents_dir / "neighbor.md").resolve())
    orch._source_to_docid[reference] = "reference"
    chunk_count = 30  # more than the 25 candidates search_similar(max_results=5) asks Chroma for
    orch.collection.add(
        ids=[f"reference_{index}" for index in range(chunk_count)],
        documents=["reference chunk"] * chunk_count,
        embeddings=[[1.0, 1.0]] * chunk_count,
        metadatas=[{"doc_id": "reference", "source": reference}] * chunk_count,
    )
    orch.collection.add(
        ids=["neighbor_0"],
        documents=["neighbor chunk"],
        embeddings=[[0.5, 1.0]],
        metadatas=[{"doc_id": "neighbor", "source": neighbor}],
    )
    assert [hit["source"] for hit in orch.search_similar(reference, max_results=5)] == [neighbor]


def test_adjacent_chunk_expansion_deduplicates_real_chroma_ids(tmp_path):
    source = str((tmp_path / "reference.md").resolve())
    client = chromadb.EphemeralClient()
    collection = client.create_collection(f"adjacent-{tmp_path.name}", embedding_function=None)
    collection.add(ids=["doc_1", "doc_3"], embeddings=[[1.0, 0.0], [0.0, 1.0]], documents=["middle", "last"])
    try:
        orchestrator = object.__new__(server.KnowledgeOrchestrator)
        orchestrator.collection = collection
        orchestrator._source_to_docid = {source: "doc"}
        hits = [{"source": source, "chunk_index": index, "content": f"chunk {index}"} for index in (0, 2)]
        expanded = orchestrator._expand_with_adjacent_chunks(hits)
        assert expanded[0]["content"] == "chunk 0\n\nmiddle"
        assert expanded[1]["content"] == "middle\n\nchunk 2\n\nlast"
    finally:
        client.delete_collection(collection.name)


def test_keyword_query_fetches_all_missing_candidates_in_one_database_read(monkeypatch):
    monkeypatch.setattr(server.config, "reranker_enabled", False)
    monkeypatch.setattr(server.config, "fts5_enabled", False)
    orchestrator = object.__new__(server.KnowledgeOrchestrator)
    orchestrator.query_cache = server.QueryCache()
    orchestrator._ensure_bm25_index = lambda: None
    orchestrator._route_by_keywords = lambda _: None
    orchestrator._source_to_docid = {}
    orchestrator.bm25_index = Mock()
    orchestrator.bm25_index.search.return_value = [(f"chunk_{index}", 10 - index) for index in range(6)]
    orchestrator.collection = Mock()

    def fetch(*, ids, include):
        return {"ids": list(reversed(ids)), "documents": list(reversed(ids)), "metadatas": [{}] * len(ids)}

    orchestrator.collection.get.side_effect = fetch
    hits = orchestrator.query("documentation", max_results=2, hybrid_alpha=0.0)
    assert [hit["content"] for hit in hits] == ["chunk_0", "chunk_1"]
    orchestrator.collection.get.assert_called_once()
    assert len(orchestrator.collection.get.call_args.kwargs["ids"]) == 6
