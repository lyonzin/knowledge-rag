"""Tests for MCP tool input validation and error handling.

Tests tool wrapper functions WITHOUT requiring ChromaDB or embeddings.
Validates input sanitization and error responses.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


def _mock_orchestrator():
    """Create a mock orchestrator that returns predictable results."""
    mock = MagicMock()
    mock.query.return_value = [
        {
            "content": "test",
            "source": "test.md",
            "filename": "test.md",
            "category": "general",
            "chunk_index": 0,
            "score": 1.0,
            "raw_rrf_score": 0.016,
            "reranker_score": None,
            "semantic_rank": 1,
            "bm25_rank": 1,
            "search_method": "hybrid",
            "keywords": ["test"],
            "routed_by": "none",
        }
    ]
    mock.query_cache.stats.return_value = {"hit_rate": "0%"}
    mock.list_categories.return_value = {"general": 5}
    mock.list_documents.return_value = [{"id": "abc", "source": "test.md"}]
    mock.get_stats.return_value = {"total_documents": 5, "total_chunks": 50}
    mock.get_document.return_value = {"content": "doc content", "source": "test.md"}
    return mock


@pytest.fixture
def mock_orch():
    mock = _mock_orchestrator()
    with patch("mcp_server.server.get_orchestrator", return_value=mock):
        yield mock


class TestSearchKnowledge:
    def test_empty_query_error(self, mock_orch):
        from mcp_server.server import search_knowledge

        r = json.loads(search_knowledge(""))
        assert r["status"] == "error"

    def test_whitespace_query_error(self, mock_orch):
        from mcp_server.server import search_knowledge

        r = json.loads(search_knowledge("   "))
        assert r["status"] == "error"

    def test_invalid_category_error(self, mock_orch):
        from mcp_server.server import search_knowledge

        r = json.loads(search_knowledge("test", category="NONEXISTENT"))
        assert r["status"] == "error"

    def test_valid_query_success(self, mock_orch):
        from mcp_server.server import search_knowledge

        r = json.loads(search_knowledge("test query", snippet_mode=False))
        assert r["status"] == "success"
        assert r["result_count"] == 1

    def test_min_score_filters_low_results(self, mock_orch):
        from mcp_server.server import search_knowledge

        mock_orch.query.return_value = [
            {
                "content": "high",
                "source": "a.md",
                "filename": "a.md",
                "category": "general",
                "chunk_index": 0,
                "score": 0.9,
                "raw_rrf_score": 0.02,
                "reranker_score": None,
                "semantic_rank": 1,
                "bm25_rank": 1,
                "search_method": "hybrid",
                "keywords": ["test"],
                "routed_by": "none",
            },
            {
                "content": "low",
                "source": "b.md",
                "filename": "b.md",
                "category": "general",
                "chunk_index": 0,
                "score": 0.1,
                "raw_rrf_score": 0.001,
                "reranker_score": None,
                "semantic_rank": None,
                "bm25_rank": 5,
                "search_method": "keyword",
                "keywords": ["test"],
                "routed_by": "none",
            },
        ]
        r = json.loads(search_knowledge("test", min_score=0.5, snippet_mode=False))
        assert r["result_count"] == 1
        assert r["filtered_by_score"] == 1
        assert r["results"][0]["content"] == "high"

    def test_min_score_zero_returns_all(self, mock_orch):
        from mcp_server.server import search_knowledge

        mock_orch.query.return_value = [
            {
                "content": "a",
                "source": "a.md",
                "filename": "a.md",
                "category": "general",
                "chunk_index": 0,
                "score": 0.9,
                "raw_rrf_score": 0.02,
                "reranker_score": None,
                "semantic_rank": 1,
                "bm25_rank": 1,
                "search_method": "hybrid",
                "keywords": [],
                "routed_by": "none",
            },
            {
                "content": "b",
                "source": "b.md",
                "filename": "b.md",
                "category": "general",
                "chunk_index": 0,
                "score": 0.0,
                "raw_rrf_score": 0.001,
                "reranker_score": None,
                "semantic_rank": None,
                "bm25_rank": 5,
                "search_method": "keyword",
                "keywords": [],
                "routed_by": "none",
            },
        ]
        r = json.loads(search_knowledge("test", min_score=0.0, snippet_mode=False))
        assert r["result_count"] == 2
        assert r["filtered_by_score"] == 0

    def test_snippet_mode_truncates_long_content(self, mock_orch):
        from mcp_server.server import search_knowledge

        long_content = "A" * 300 + ". " + "B" * 300 + ". " + "C" * 300
        mock_orch.query.return_value = [
            {
                "content": long_content,
                "source": "a.md",
                "filename": "a.md",
                "category": "general",
                "chunk_index": 0,
                "score": 1.0,
                "raw_rrf_score": 0.02,
                "reranker_score": None,
                "semantic_rank": 1,
                "bm25_rank": 1,
                "search_method": "hybrid",
                "keywords": [],
                "routed_by": "none",
            },
        ]
        r = json.loads(search_knowledge("test", snippet_mode=True))
        result = r["results"][0]
        assert len(result["content"]) <= 510
        assert result["content_length"] == len(long_content)

    def test_snippet_mode_preserves_short_content(self, mock_orch):
        from mcp_server.server import search_knowledge

        short = "Short content here."
        mock_orch.query.return_value = [
            {
                "content": short,
                "source": "a.md",
                "filename": "a.md",
                "category": "general",
                "chunk_index": 0,
                "score": 1.0,
                "raw_rrf_score": 0.02,
                "reranker_score": None,
                "semantic_rank": 1,
                "bm25_rank": 1,
                "search_method": "hybrid",
                "keywords": [],
                "routed_by": "none",
            },
        ]
        r = json.loads(search_knowledge("test", snippet_mode=True))
        assert r["results"][0]["content"] == short
        assert r["results"][0]["content_length"] == len(short)

    def test_snippet_mode_false_returns_full(self, mock_orch):
        from mcp_server.server import search_knowledge

        long_content = "X" * 1000
        mock_orch.query.return_value = [
            {
                "content": long_content,
                "source": "a.md",
                "filename": "a.md",
                "category": "general",
                "chunk_index": 0,
                "score": 1.0,
                "raw_rrf_score": 0.02,
                "reranker_score": None,
                "semantic_rank": 1,
                "bm25_rank": 1,
                "search_method": "hybrid",
                "keywords": [],
                "routed_by": "none",
            },
        ]
        r = json.loads(search_knowledge("test", snippet_mode=False))
        assert r["results"][0]["content"] == long_content
        assert "content_length" not in r["results"][0]


class TestAddDocument:
    def test_empty_content_error(self, mock_orch):
        from mcp_server.server import add_document

        r = json.loads(add_document("", "test.md", "general"))
        assert r["status"] == "error"

    def test_empty_filepath_error(self, mock_orch):
        from mcp_server.server import add_document

        r = json.loads(add_document("content", "", "general"))
        assert r["status"] == "error"


class TestUpdateDocument:
    def test_empty_content_error(self, mock_orch):
        from mcp_server.server import update_document

        r = json.loads(update_document("somefile.md", ""))
        assert r["status"] == "error"

    def test_missing_filepath_error(self, mock_orch):
        from mcp_server.server import update_document

        r = json.loads(update_document("", "content"))
        assert r["status"] == "error"


class TestRemoveDocument:
    def test_empty_filepath_error(self, mock_orch):
        from mcp_server.server import remove_document

        r = json.loads(remove_document(""))
        assert r["status"] == "error"


class TestAddFromUrl:
    def test_empty_url_error(self, mock_orch):
        from mcp_server.server import add_from_url

        r = json.loads(add_from_url(""))
        assert r["status"] == "error"

    def test_file_scheme_blocked(self, mock_orch):
        from mcp_server.server import add_from_url

        mock_orch.add_from_url.return_value = {"error": "Only http:// and https:// URLs are supported"}
        r = json.loads(add_from_url("file:///etc/passwd"))
        assert r["status"] == "error"


class TestSearchSimilar:
    def test_empty_filepath_error(self, mock_orch):
        from mcp_server.server import search_similar

        r = json.loads(search_similar(""))
        assert r["status"] == "error"


class TestListDocuments:
    def test_pages_category_matches_in_source_order_and_reports_their_total(self, monkeypatch):
        from mcp_server import server

        orchestrator = object.__new__(server.KnowledgeOrchestrator)
        orchestrator._indexed_docs = {  # index order differs from source order, as after re-indexing
            f"doc{index}": {"source": f"doc{index}.md", "category": "security" if index % 2 else "general"}
            for index in reversed(range(7))
        }
        monkeypatch.setattr(server, "get_orchestrator", lambda: orchestrator)
        r = json.loads(server.list_documents("security", limit=2, offset=1))
        assert [doc["id"] for doc in r["documents"]] == ["doc3", "doc5"]
        assert (r["total"], r["offset"], r["limit"], r["count"]) == (3, 1, 2, 2)

    @pytest.mark.parametrize(("limit", "offset", "expected"), [(0, -3, (1, 0)), (5000, 0, (200, 0))])
    def test_out_of_range_paging_is_clamped(self, mock_orch, limit, offset, expected):
        from mcp_server.server import list_documents

        r = json.loads(list_documents(limit=limit, offset=offset))
        assert (r["limit"], r["offset"]) == expected


class TestEvaluateRetrieval:
    def test_invalid_json_error(self, mock_orch):
        from mcp_server.server import evaluate_retrieval

        r = json.loads(evaluate_retrieval("not json"))
        assert r["status"] == "error"

    def test_empty_array_error(self, mock_orch):
        from mcp_server.server import evaluate_retrieval

        r = json.loads(evaluate_retrieval("[]"))
        assert r["status"] == "error"
