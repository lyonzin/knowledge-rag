"""Ranking candidates and evaluation inputs must reflect actual retrieval quality."""

import json
from unittest.mock import Mock

import pytest

import mcp_server.server as server


@pytest.mark.parametrize(
    "case",
    [
        {},
        {"query": "test"},
        {"query": "", "expected_filepath": "a.md"},
        {"query": "test", "expected_filepath": " "},
        "bad",
        None,
        {"query": 42, "expected_filepath": "a.md"},
    ],
)
def test_invalid_evaluation_cases_are_rejected_before_search(case, monkeypatch):
    get_orchestrator = Mock()
    monkeypatch.setattr(server, "get_orchestrator", get_orchestrator)
    response = json.loads(server.evaluate_retrieval(json.dumps([case])))
    assert response["status"] == "error"
    get_orchestrator.assert_not_called()
    orchestrator = object.__new__(server.KnowledgeOrchestrator)
    orchestrator.query = Mock()
    with pytest.raises(ValueError):
        orchestrator.evaluate_retrieval([case])
    orchestrator.query.assert_not_called()


def test_valid_evaluation_cases_keep_reciprocal_rank_contract():
    orchestrator = object.__new__(server.KnowledgeOrchestrator)
    orchestrator.query = Mock(return_value=[{"source": "other.md"}, {"source": "/docs/expected.md"}])
    metrics = orchestrator.evaluate_retrieval([{"query": "query", "expected_filepath": "expected.md"}])
    assert metrics["mrr_at_5"] == 0.5
    assert metrics["recall_at_5"] == 1.0


@pytest.mark.parametrize(
    "source, expected, matched",
    [
        (r"C:\docs\security\a.md", "security/a.md", True),
        ("/docs/security/a.md", r"security\a.md", True),
        (r"C:\docs\security\a.md", r"C:\docs\security\a.md", True),
        (r"C:\docs\security\a.md", "c:/DOCS/security/A.md", True),
        ("/docs/security/a.md", "a.md", True),
        ("/docs/security/nota.md", "a.md", False),
        ("/other/docs/security/a.md", "/docs/security/a.md", False),
        ("/docs/notsecurity/a.md", "security/a.md", False),
    ],
)
def test_evaluation_matches_path_components_not_substrings(source, expected, matched):
    orchestrator = object.__new__(server.KnowledgeOrchestrator)
    orchestrator.query = Mock(return_value=[{"source": source}])
    metrics = orchestrator.evaluate_retrieval([{"query": "query", "expected_filepath": expected}])
    assert metrics["recall_at_5"] == float(matched)


def test_enabled_fts5_reranker_can_promote_candidate_beyond_result_limit(monkeypatch):
    monkeypatch.setattr(server.config, "fts5_rerank_enabled", True)
    orchestrator = object.__new__(server.KnowledgeOrchestrator)
    orchestrator.fts5_index = Mock()
    orchestrator.fts5_index.search.return_value = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
    orchestrator.collection = Mock()
    orchestrator.collection.get.return_value = {
        "ids": ["a", "b", "c"],
        "documents": ["ordinary", "other", "best answer"],
        "metadatas": [{}, {}, {}],
    }
    orchestrator._source_to_docid = {}
    orchestrator.reranker = Mock()
    orchestrator.reranker.rerank.side_effect = lambda query, documents, top_k: list(reversed(documents))[:top_k]
    hits = orchestrator._run_fts5_search("query", 1, None, skip_min_hits=True)
    assert hits[0]["content"] == "best answer"
    assert len(hits) == 1
    assert len(orchestrator.reranker.rerank.call_args.args[1]) == 3


def test_evaluation_matches_windows_paths_but_not_similar_names():
    """Exercise the public evaluation method, including rank calculation."""
    from unittest.mock import Mock

    from mcp_server.server import KnowledgeOrchestrator

    orchestrator = object.__new__(KnowledgeOrchestrator)
    orchestrator.query = Mock(
        return_value=[
            {"source": "C:\\knowledge\\security\\notfoo.md"},
            {"source": "C:\\knowledge\\security\\foo.md"},
        ]
    )
    result = orchestrator.evaluate_retrieval(
        [
            {"query": "security guidance", "expected_filepath": "security/foo.md"},
            {"query": "other guidance", "expected_filepath": "other/foo.md"},
        ]
    )
    assert result["mrr_at_5"] == 0.25
    assert result["recall_at_5"] == 0.5
    assert [entry["found_at_rank"] for entry in result["per_query"]] == [2, None]
