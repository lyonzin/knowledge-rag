"""Anti-regression tests for PR #98.

Pins two contracts so future refactors cannot silently revert them:

1. `search_knowledge(category="general")` must be accepted by input validation
   even when the user customizes `config.category_mappings` to drop the
   default `"general": "general"` entry — the parser hardcodes `"general"`
   as a fallback in `_detect_category`, so the validator must always
   tolerate it.

2. The hybrid-search pipeline must skip BM25-only hits whose `chunk_id`
   Chroma can no longer resolve (stale after reindex / removal). The
   previous fallback inserted an entry with empty `document`/`metadata`
   into the reranker; the guard added in PR #98 drops it instead.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_server.server import KnowledgeOrchestrator, search_knowledge

# ---------------------------------------------------------------------------
# Bug 1 — general category accepted under custom config
# ---------------------------------------------------------------------------


def _mock_orchestrator_returning_one_result():
    mock = MagicMock()
    mock.query.return_value = [
        {
            "content": "sample",
            "source": "sample.md",
            "filename": "sample.md",
            "category": "general",
            "chunk_index": 0,
            "score": 1.0,
            "raw_rrf_score": 0.016,
            "reranker_score": None,
            "semantic_rank": 1,
            "bm25_rank": 1,
            "search_method": "hybrid",
            "keywords": ["sample"],
            "routed_by": "none",
        }
    ]
    mock.query_cache.stats.return_value = {"hit_rate": "0%"}
    return mock


class TestGeneralCategoryAcceptedWithCustomConfig:
    """category='general' must be accepted even if user config omits it."""

    def test_general_accepted_when_missing_from_category_mappings(self, monkeypatch):
        # Simulate a user who customized config.yaml and dropped the
        # default "general": "general" mapping. Parser still emits
        # "general" as fallback (ingestion.py _detect_category), so
        # validation must not reject it.
        monkeypatch.setattr(
            "mcp_server.server.config.category_mappings",
            {"security/redteam": "redteam", "logscale": "logscale"},
        )
        monkeypatch.setattr(
            "mcp_server.server.config.keyword_routes",
            {"redteam": ["pentest"], "logscale": ["lql"]},
        )

        mock = _mock_orchestrator_returning_one_result()
        with patch("mcp_server.server.get_orchestrator", return_value=mock):
            r = json.loads(search_knowledge("test", category="general", snippet_mode=False))

        assert r["status"] != "error", (
            f"general category must be accepted even with custom config, got: {r.get('message', r)}"
        )

    def test_truly_invalid_category_still_rejected(self, monkeypatch):
        # Defensive companion: the guard must not blanket-accept everything.
        monkeypatch.setattr(
            "mcp_server.server.config.category_mappings",
            {"security/redteam": "redteam"},
        )
        monkeypatch.setattr(
            "mcp_server.server.config.keyword_routes",
            {"redteam": ["pentest"]},
        )

        mock = _mock_orchestrator_returning_one_result()
        with patch("mcp_server.server.get_orchestrator", return_value=mock):
            r = json.loads(search_knowledge("test", category="ZZZ_NOT_A_CATEGORY"))

        assert r["status"] == "error"
        assert "Invalid category" in r["message"]


# ---------------------------------------------------------------------------
# Bug 2 — BM25 stale chunk_id must be skipped, not emitted as empty result
# ---------------------------------------------------------------------------


class TestBM25StaleChunkIdSkipped:
    """BM25 returns chunk_id that Chroma no longer resolves -> skip, not empty."""

    @pytest.fixture
    def orch(self, monkeypatch):
        # Build a KnowledgeOrchestrator without running __init__ so no Chroma,
        # no embedding model, no filesystem touched. Only the attributes the
        # query() pipeline reads are populated.
        o = KnowledgeOrchestrator.__new__(KnowledgeOrchestrator)

        o.collection = MagicMock()
        # Semantic search returns no hits — forces BM25-only path through
        # the branch that calls collection.get() per chunk_id.
        o.collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        def fake_get(ids, include):
            # Note: collection.get() returns flat lists (one entry per id),
            # NOT nested like collection.query(). The server code at
            # server.py:1545-1551 indexes with [0] — matches Chroma 1.4+ shape.
            if ids == ["stale_chunk_id"]:
                # Real Chroma 1.4+ behavior for unknown ids: empty lists,
                # no exception.
                return {"documents": [], "metadatas": []}
            return {
                "documents": ["valid content"],
                "metadatas": [
                    {
                        "source": "valid.md",
                        "filename": "valid.md",
                        "category": "general",
                        "chunk_index": 0,
                        "keywords": "",
                    }
                ],
            }

        o.collection.get.side_effect = fake_get

        o.bm25_index = MagicMock()
        o.bm25_index.search.return_value = [
            ("stale_chunk_id", 5.0),
            ("valid_chunk_id", 3.0),
        ]

        o.reranker = MagicMock()
        # Pass-through reranker so we can observe what the pipeline actually
        # forwarded to it.
        o.reranker.rerank.side_effect = lambda q, docs, top_k: docs[:top_k]

        o.query_cache = MagicMock()
        o.query_cache.get.return_value = None

        o._bm25_initialized = True
        o._ensure_bm25_index = MagicMock()
        o._route_by_keywords = MagicMock(return_value=None)
        o._source_to_docid = {}
        # Skip adjacent-chunk expansion (it would hit collection.get again
        # with a different id pattern and clutter the assertion).
        o._expand_with_adjacent_chunks = lambda results, window=1: results
        # Skip MMR diversification — not relevant to this guard.
        o._apply_mmr = lambda results, k, lambda_param: results[:k]

        # Reranker enabled toggles whether reranker.rerank is even called;
        # leave defaults and just trust the pass-through above.
        return o

    def test_stale_bm25_id_does_not_emit_empty_result(self, orch):
        # hybrid_alpha=0.0 disables the semantic branch entirely, forcing
        # every BM25 hit through the collection.get() path where the guard
        # lives.
        results = orch.query("test", max_results=5, hybrid_alpha=0.0)

        assert len(results) == 1, f"expected only valid chunk to survive, got {results}"
        assert results[0]["content"] == "valid content"
        assert results[0]["source"] == "valid.md"
        # Hard contract: no empty-content entries may ever leak through.
        assert all(r["content"] for r in results), "stale BM25 chunk_id must be skipped, not emitted with empty content"

    def test_reranker_never_sees_empty_document(self, orch):
        orch.query("test", max_results=5, hybrid_alpha=0.0)

        # Inspect what was actually handed to the reranker. Even if a future
        # refactor changes the output filter, this assertion catches empty
        # docs leaking into the reranker pipeline.
        for call in orch.reranker.rerank.call_args_list:
            rerank_input = call.args[1] if len(call.args) >= 2 else call.kwargs.get("documents", [])
            for entry in rerank_input:
                assert entry["document"], f"reranker received an entry with empty document: {entry}"
