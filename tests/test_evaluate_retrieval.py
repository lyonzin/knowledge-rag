"""
╭─╴ KNOWLEDGE RAG · EVAL TESTS ╶─────────────────────── v 4.6.0 ─╮
│                                                                │
│   Coverage for retrieval quality metrics (MRR/Recall/P@5).     │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    ├─ Since   ·  v4.6.0
    └─ Date    ·  2026-07-27

Covers Q1.8: the ``evaluate_retrieval`` docstring promised Precision@5 while the
code only computed MRR@5 and Recall@5. These tests pin the real formula,
including the denominator choice when fewer than five results come back.

Denominator decision
--------------------
``Precision@5 = hits_in_top_5 / min(5, len(results))``.

With a normally-sized index the denominator is 5, matching the textbook
definition and the ADR (``P@5 = hits_at_5 / 5``). When the index can only return
three chunks, dividing by 5 would cap the score at 0.6 for reasons unrelated to
ranking quality, so the denominator follows what was actually returned. Each
per-query entry reports its own ``precision_denominator`` so the figure stays
auditable either way.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mcp_server.server import KnowledgeOrchestrator


def _fake_orchestrator(results_by_query):
    """Build an object exposing the real evaluate_retrieval over canned results.

    Args:
        results_by_query: Mapping of query string -> list of result dicts.

    Returns:
        SimpleNamespace: Bound ``evaluate_retrieval`` / ``_expected_paths`` with a
        stubbed ``query`` — the metric maths is exercised, the search is not.
    """
    holder = SimpleNamespace()
    holder.query = lambda q, max_results=5: results_by_query.get(q, [])[:max_results]
    holder._expected_paths = KnowledgeOrchestrator._expected_paths
    holder.evaluate_retrieval = KnowledgeOrchestrator.evaluate_retrieval.__get__(holder)
    return holder


def _hits(*sources):
    """Shorthand: turn source paths into minimal result dicts."""
    return [{"source": s} for s in sources]


# ── Precision@5 ──


class TestPrecisionAtFive:
    def test_three_of_five_relevant(self):
        """3 relevant out of 5 returned -> 0.6."""
        results = _hits(
            "docs/a.md",
            "docs/noise1.md",
            "docs/b.md",
            "docs/noise2.md",
            "docs/c.md",
        )
        orch = _fake_orchestrator({"q": results})
        report = orch.evaluate_retrieval(
            [{"query": "q", "expected_filepaths": ["docs/a.md", "docs/b.md", "docs/c.md"]}]
        )

        assert report["precision_at_5"] == 0.6
        assert report["per_query"][0]["hits_at_5"] == 3
        assert report["per_query"][0]["precision_denominator"] == 5

    def test_zero_of_five_relevant(self):
        """No relevant document in the top-5 -> 0.0."""
        results = _hits(*[f"docs/noise{i}.md" for i in range(5)])
        orch = _fake_orchestrator({"q": results})
        report = orch.evaluate_retrieval([{"query": "q", "expected_filepath": "docs/target.md"}])

        assert report["precision_at_5"] == 0.0
        assert report["per_query"][0]["hits_at_5"] == 0
        assert report["per_query"][0]["found_at_rank"] is None

    def test_five_of_five_relevant(self):
        """A perfect page -> 1.0."""
        results = _hits(*[f"docs/hit{i}.md" for i in range(5)])
        orch = _fake_orchestrator({"q": results})
        report = orch.evaluate_retrieval([{"query": "q", "expected_filepaths": [f"docs/hit{i}.md" for i in range(5)]}])

        assert report["precision_at_5"] == 1.0

    def test_single_expected_path_caps_at_one_fifth(self):
        """With one ground-truth document, 0.2 is the ceiling — by definition."""
        results = _hits("docs/target.md", *[f"docs/noise{i}.md" for i in range(4)])
        orch = _fake_orchestrator({"q": results})
        report = orch.evaluate_retrieval([{"query": "q", "expected_filepath": "docs/target.md"}])

        assert report["precision_at_5"] == 0.2
        assert report["per_query"][0]["hits_at_5"] == 1


class TestPrecisionDenominator:
    def test_fewer_than_five_results_divides_by_length(self):
        """Only 3 results returned, 2 relevant -> 2/3, not 2/5.

        A three-chunk index must not be scored as if two slots were wrong.
        """
        results = _hits("docs/a.md", "docs/noise.md", "docs/b.md")
        orch = _fake_orchestrator({"q": results})
        report = orch.evaluate_retrieval([{"query": "q", "expected_filepaths": ["docs/a.md", "docs/b.md"]}])

        assert report["per_query"][0]["precision_denominator"] == 3
        assert report["precision_at_5"] == pytest.approx(2 / 3, abs=1e-4)

    def test_no_results_scores_zero_without_dividing_by_zero(self):
        """An empty result list yields 0.0, never a ZeroDivisionError."""
        orch = _fake_orchestrator({"q": []})
        report = orch.evaluate_retrieval([{"query": "q", "expected_filepath": "docs/a.md"}])

        assert report["precision_at_5"] == 0.0
        assert report["per_query"][0]["precision_denominator"] == 0
        assert report["per_query"][0]["top_result"] == "none"

    def test_five_or_more_results_divides_by_five(self):
        """At full page size the denominator matches the ADR formula exactly."""
        results = _hits(*[f"docs/hit{i}.md" for i in range(5)])
        orch = _fake_orchestrator({"q": results})
        report = orch.evaluate_retrieval([{"query": "q", "expected_filepaths": ["docs/hit0.md"]}])

        assert report["per_query"][0]["precision_denominator"] == 5
        assert report["precision_at_5"] == 0.2


# ── Metrics coexist ──


class TestAllMetrics:
    def test_mrr_recall_precision_reported_together(self):
        """All three headline metrics are present in the payload."""
        results = _hits("docs/noise.md", "docs/a.md", "docs/b.md", "docs/x.md", "docs/y.md")
        orch = _fake_orchestrator({"q": results})
        report = orch.evaluate_retrieval([{"query": "q", "expected_filepaths": ["docs/a.md", "docs/b.md"]}])

        assert report["mrr_at_5"] == 0.5  # first hit at rank 2
        assert report["recall_at_5"] == 1.0
        assert report["precision_at_5"] == 0.4
        assert report["total_queries"] == 1

    def test_metrics_average_across_queries(self):
        """Aggregates are means over the test cases, not sums."""
        orch = _fake_orchestrator(
            {
                "good": _hits("docs/a.md", "docs/b.md", "n1.md", "n2.md", "n3.md"),
                "bad": _hits("n1.md", "n2.md", "n3.md", "n4.md", "n5.md"),
            }
        )
        report = orch.evaluate_retrieval(
            [
                {"query": "good", "expected_filepaths": ["docs/a.md", "docs/b.md"]},
                {"query": "bad", "expected_filepath": "docs/z.md"},
            ]
        )

        assert report["total_queries"] == 2
        assert report["precision_at_5"] == 0.2  # (0.4 + 0.0) / 2
        assert report["recall_at_5"] == 0.5
        assert report["mrr_at_5"] == 0.5

    def test_only_top_five_are_scored(self):
        """A relevant hit at rank 6 counts for nothing at k=5."""
        orch = _fake_orchestrator({"q": _hits(*[f"n{i}.md" for i in range(5)], "docs/a.md")})
        report = orch.evaluate_retrieval([{"query": "q", "expected_filepath": "docs/a.md"}])

        assert report["precision_at_5"] == 0.0
        assert report["recall_at_5"] == 0.0


# ── Ground-truth parsing ──


class TestExpectedPaths:
    def test_single_key(self):
        """Legacy single-path key still works."""
        assert KnowledgeOrchestrator._expected_paths({"expected_filepath": "a.md"}) == ["a.md"]

    def test_plural_key(self):
        """Plural key accepts a list."""
        assert KnowledgeOrchestrator._expected_paths({"expected_filepaths": ["a.md", "b.md"]}) == ["a.md", "b.md"]

    def test_both_keys_merge_without_duplicates(self):
        """Mixing both keys unions them, order preserved."""
        case = {"expected_filepath": "a.md", "expected_filepaths": ["a.md", "b.md"]}
        assert KnowledgeOrchestrator._expected_paths(case) == ["a.md", "b.md"]

    def test_missing_ground_truth(self):
        """A case with no ground truth yields no paths instead of raising."""
        assert KnowledgeOrchestrator._expected_paths({"query": "q"}) == []

    def test_non_string_entries_ignored(self):
        """Malformed JSON input must not crash the evaluation."""
        case = {"expected_filepaths": ["a.md", 42, None, ""]}
        assert KnowledgeOrchestrator._expected_paths(case) == ["a.md"]

    def test_case_without_ground_truth_scores_zero(self):
        """No expected path means nothing can match."""
        orch = _fake_orchestrator({"q": _hits("a.md", "b.md")})
        report = orch.evaluate_retrieval([{"query": "q"}])

        assert report["precision_at_5"] == 0.0
        assert report["recall_at_5"] == 0.0


# ── Backwards compatibility ──


class TestBackwardsCompatibility:
    def test_legacy_payload_fields_survive(self):
        """Existing consumers keep reading the same per-query keys."""
        orch = _fake_orchestrator({"q": _hits("docs/a.md")})
        entry = orch.evaluate_retrieval([{"query": "q", "expected_filepath": "docs/a.md"}])["per_query"][0]

        for key in ("query", "expected", "found_at_rank", "reciprocal_rank", "top_result"):
            assert key in entry
        assert entry["expected"] == "docs/a.md"

    def test_tool_wrapper_reports_precision(self):
        """The MCP tool surfaces precision_at_5 in its JSON payload."""
        from mcp_server.server import evaluate_retrieval

        orch = _fake_orchestrator({"q": _hits("docs/a.md", "n1.md", "n2.md", "n3.md", "n4.md")})
        with patch("mcp_server.server.get_orchestrator", return_value=orch):
            payload = json.loads(evaluate_retrieval(json.dumps([{"query": "q", "expected_filepath": "docs/a.md"}])))

        assert payload["status"] == "success"
        assert payload["precision_at_5"] == 0.2
        assert payload["mrr_at_5"] == 1.0
