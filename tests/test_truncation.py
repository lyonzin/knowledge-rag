"""
╭─╴ KNOWLEDGE RAG · TRUNCATION TESTS ╶───────────────── v 4.6.0 ─╮
│                                                                │
│   Coverage for budget-aware result truncation.                 │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    ├─ Since   ·  v4.6.0
    └─ Date    ·  2026-07-27

Covers Q1.6: oversized responses must be cut deliberately, never silently. A
truncated payload always says so, always keeps the rank-1 result, and always
cuts at a result boundary — a half-serialized result would be worse than none.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_server.server import (
    DOCUMENT_CHAR_BUDGET,
    RESULT_CHAR_BUDGET,
    _truncate_document_safe,
    _truncate_results_safe,
    _truncation_note,
)


def _result(name: str, size: int) -> dict:
    """Build a search-result-shaped dict with a payload of roughly ``size`` chars."""
    return {
        "content": "x" * size,
        "source": f"{name}.md",
        "filename": f"{name}.md",
        "category": "general",
        "chunk_index": 0,
        "score": 1.0,
        "search_method": "hybrid",
    }


def _mock_orchestrator(results):
    """Mock orchestrator returning ``results`` from ``query()``."""
    mock = MagicMock()
    mock.query.return_value = results
    mock.query_cache.stats.return_value = {"hit_rate": "0%"}
    return mock


# ── Helper behaviour ──


class TestTruncateResultsSafe:
    def test_within_budget_is_untouched(self):
        """A small list passes through with truncated=False."""
        results = [_result("a", 10), _result("b", 10)]
        kept, was_truncated, total = _truncate_results_safe(results, char_budget=100_000)
        assert kept == results
        assert was_truncated is False
        assert total == 2

    def test_empty_list(self):
        """Zero results never reports truncation."""
        kept, was_truncated, total = _truncate_results_safe([], char_budget=10)
        assert kept == []
        assert was_truncated is False
        assert total == 0

    def test_over_budget_drops_tail(self):
        """Results beyond the budget are dropped and the cut is reported."""
        results = [_result(f"doc{i}", 1000) for i in range(10)]
        kept, was_truncated, total = _truncate_results_safe(results, char_budget=3500)
        assert was_truncated is True
        assert total == 10
        assert 0 < len(kept) < 10

    def test_seed_protection_keeps_oversized_top_result(self):
        """The rank-1 result survives even when it alone blows the budget.

        A giant answer is recoverable; a missing best answer is not.
        """
        results = [_result("huge", 50_000), _result("small", 10)]
        kept, was_truncated, total = _truncate_results_safe(results, char_budget=100)
        assert len(kept) == 1
        assert kept[0]["source"] == "huge.md"
        assert was_truncated is True
        assert total == 2

    def test_single_oversized_result_is_not_flagged(self):
        """One result that exceeds the budget is kept, and nothing was dropped."""
        results = [_result("huge", 50_000)]
        kept, was_truncated, total = _truncate_results_safe(results, char_budget=100)
        assert kept == results
        assert was_truncated is False
        assert total == 1

    def test_cut_is_on_result_boundary(self):
        """Every kept result is a complete, unmodified object."""
        results = [_result(f"doc{i}", 2000) for i in range(8)]
        kept, was_truncated, _ = _truncate_results_safe(results, char_budget=5000)
        assert was_truncated is True
        for i, item in enumerate(kept):
            assert item is results[i]  # identical object, not a slice
            assert len(item["content"]) == 2000
            assert set(item) == set(results[i])

    def test_kept_results_keep_original_order(self):
        """Truncation is a prefix — ranking order is preserved."""
        results = [_result(f"doc{i}", 500) for i in range(10)]
        kept, _, _ = _truncate_results_safe(results, char_budget=2000)
        assert [r["source"] for r in kept] == [f"doc{i}.md" for i in range(len(kept))]

    def test_non_positive_budget_disables_truncation(self):
        """A budget of 0 opts out entirely."""
        results = [_result(f"doc{i}", 5000) for i in range(5)]
        kept, was_truncated, total = _truncate_results_safe(results, char_budget=0)
        assert kept == results
        assert was_truncated is False
        assert total == 5

    def test_non_serializable_result_does_not_raise(self):
        """A result carrying something JSON cannot encode still gets a size."""
        results = [{"content": "ok"}, {"content": object()}]
        kept, _, total = _truncate_results_safe(results, char_budget=100_000)
        assert total == 2
        assert len(kept) == 2

    def test_default_budget_is_documented_constant(self):
        """The default budget is the module constant, not a magic number."""
        results = [_result("a", 10)]
        assert _truncate_results_safe(results)[0] == results
        assert RESULT_CHAR_BUDGET > 0


class TestTruncationNote:
    def test_note_carries_the_marker_and_both_counts(self):
        """The note is the signal the calling model reads — it must be explicit."""
        note = _truncation_note(3, 20)
        assert "[!] TRUNCATED" in note
        assert "3" in note and "20" in note


# ── search_knowledge payload ──


class TestSearchKnowledgeTruncation:
    def test_untruncated_payload_reports_false(self):
        """The ``truncated`` field is always present, so clients can rely on it."""
        from mcp_server.server import search_knowledge

        with patch("mcp_server.server.get_orchestrator", return_value=_mock_orchestrator([_result("a", 50)])):
            payload = json.loads(search_knowledge("test", snippet_mode=False))
        assert payload["status"] == "success"
        assert payload["truncated"] is False
        assert "note" not in payload
        assert "shown" not in payload

    def test_truncated_payload_carries_warning_and_counts(self):
        """An oversized result set is reported, not silently shortened."""
        from mcp_server.server import search_knowledge

        big = [_result(f"doc{i}", 60_000) for i in range(20)]
        with patch("mcp_server.server.get_orchestrator", return_value=_mock_orchestrator(big)):
            payload = json.loads(search_knowledge("test", max_results=20, snippet_mode=False))

        assert payload["truncated"] is True
        assert "[!] TRUNCATED" in payload["note"]
        assert payload["total_matched"] == 20
        assert payload["shown"] == len(payload["results"]) < 20
        assert payload["result_count"] == payload["shown"]

    def test_top_result_survives_truncation(self):
        """Seed protection holds end-to-end through the tool wrapper."""
        from mcp_server.server import search_knowledge

        big = [_result("winner", 200_000)] + [_result(f"doc{i}", 60_000) for i in range(10)]
        with patch("mcp_server.server.get_orchestrator", return_value=_mock_orchestrator(big)):
            payload = json.loads(search_knowledge("test", max_results=20, snippet_mode=False))

        assert payload["results"][0]["source"] == "winner.md"
        assert payload["truncated"] is True

    def test_snippet_mode_usually_avoids_truncation(self):
        """Snippets shrink results enough that the budget is not hit."""
        from mcp_server.server import search_knowledge

        big = [_result(f"doc{i}", 60_000) for i in range(20)]
        with patch("mcp_server.server.get_orchestrator", return_value=_mock_orchestrator(big)):
            payload = json.loads(search_knowledge("test", max_results=20, snippet_mode=True))

        assert payload["truncated"] is False
        assert payload["result_count"] == 20

    def test_filtered_by_score_is_unaffected_by_truncation(self):
        """``filtered_by_score`` counts score drops only, never truncation drops."""
        from mcp_server.server import search_knowledge

        big = [_result(f"doc{i}", 60_000) for i in range(20)]
        with patch("mcp_server.server.get_orchestrator", return_value=_mock_orchestrator(big)):
            payload = json.loads(search_knowledge("test", max_results=20, snippet_mode=False))

        assert payload["filtered_by_score"] == 0


# ── get_document payload ──


class TestGetDocumentTruncation:
    def test_small_document_is_not_truncated(self):
        """Ordinary documents pass through untouched."""
        doc = {"content": "short content", "source": "a.md"}
        trimmed, was_truncated = _truncate_document_safe(doc)
        assert trimmed is doc
        assert was_truncated is False

    def test_oversized_document_is_cut_at_a_line_boundary(self):
        """The cut lands on a newline so the model never sees a split line."""
        content = "\n".join(f"line {i} " + "y" * 100 for i in range(5000))
        doc = {"content": content, "source": "big.md"}
        trimmed, was_truncated = _truncate_document_safe(doc, char_budget=10_000)

        assert was_truncated is True
        assert trimmed["content_length"] == len(content)
        assert trimmed["shown_chars"] == len(trimmed["content"])
        assert len(trimmed["content"]) <= 10_000
        assert not trimmed["content"].endswith("\n")
        assert content.startswith(trimmed["content"])

    def test_original_document_is_not_mutated(self):
        """Truncation returns a copy — the orchestrator's dict stays intact."""
        content = "z" * 50_000
        doc = {"content": content, "source": "big.md"}
        trimmed, _ = _truncate_document_safe(doc, char_budget=1000)
        assert doc["content"] == content
        assert trimmed is not doc

    def test_non_string_content_is_ignored(self):
        """A malformed payload must not raise."""
        doc = {"content": None, "source": "a.md"}
        trimmed, was_truncated = _truncate_document_safe(doc, char_budget=10)
        assert trimmed is doc
        assert was_truncated is False

    def test_tool_reports_truncation(self):
        """``get_document`` surfaces the flag and the note."""
        from mcp_server.server import get_document

        mock = MagicMock()
        mock.get_document.return_value = {
            "content": "\n".join("w" * 200 for _ in range(5000)),
            "source": "big.md",
        }
        with patch("mcp_server.server.get_orchestrator", return_value=mock):
            payload = json.loads(get_document("big.md"))

        assert payload["truncated"] is True
        assert "[!] TRUNCATED" in payload["note"]
        assert payload["document"]["shown_chars"] < payload["document"]["content_length"]

    def test_tool_reports_no_truncation_for_small_docs(self):
        """The flag is present and False for ordinary documents."""
        from mcp_server.server import get_document

        mock = MagicMock()
        mock.get_document.return_value = {"content": "tiny", "source": "a.md"}
        with patch("mcp_server.server.get_orchestrator", return_value=mock):
            payload = json.loads(get_document("a.md"))

        assert payload["truncated"] is False
        assert "note" not in payload

    def test_document_budget_constant_is_positive(self):
        """The document budget is a real, tunable constant."""
        assert DOCUMENT_CHAR_BUDGET > 0


@pytest.mark.parametrize("budget", [1, 10, 500, 5000])
def test_seed_is_never_lost_at_any_budget(budget):
    """Whatever the budget, at least the best result comes back."""
    results = [_result(f"doc{i}", 400) for i in range(6)]
    kept, _, _ = _truncate_results_safe(results, char_budget=budget)
    assert len(kept) >= 1
    assert kept[0] is results[0]
