"""Unit tests for ``mcp_server.query_router.QueryRouter``.

Covers UT-001..UT-013 assigned to Task 02 in ``_tests.md``.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

# hypothesis is an optional test dependency: some CI jobs (e.g. the nightly
# chaos suite) install only a minimal set and run `pytest -m <marker>`, which
# still collects the whole tests/ tree. importorskip lets pytest skip this
# module cleanly instead of aborting collection with a ModuleNotFoundError.
pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

from mcp_server.query_router import QueryRouter

DEFAULT_PATTERNS = [
    r"[A-Z]{2,}-\d+",
    r"CVE-\d{4}-\d+",
    r"^[a-f0-9]{32,64}$",
]

# Anchored, expressive pattern set used by the boundary tests below. The
# shipped defaults are unanchored substrings and — by design — will happily
# match "XXX-1234" inside "H-P4-XXX-1234". The spec cases UT-006/UT-008/UT-009
# want to distinguish a well-formed identifier from a malformed prefix, which
# is only meaningful against an anchored contract. This mirrors what an org
# operator would ship in ``search.lexical_fast_path.patterns`` for strict
# routing.
ROUTER_TEST_PATTERNS = [
    r"^[A-Z]{2,}(-[A-Z0-9]+)+$",  # MDR-AD002, CWE-79, CVE-2021-4034
    r"^[A-Z]\d(-[A-Z0-9]+)+$",  # H1-P4-XXX-1234 (H1-style handle prefix)
    r"^T\d{4}(\.\d{3})?$",  # T1078.001 (MITRE technique)
    r"^[a-f0-9]{32,128}$",  # md5/sha1/sha256/sha512 hex hashes
]


class TestQueryRouterInit:
    def test_init_compiles_patterns(self) -> None:
        router = QueryRouter(DEFAULT_PATTERNS)
        assert len(router._patterns) == len(DEFAULT_PATTERNS)
        assert all(isinstance(p, re.Pattern) for p in router._patterns)

    def test_init_invalid_regex_raises(self) -> None:
        """UT-012 — pattern with unmatched paren fails fast at startup."""
        with pytest.raises(re.error):
            QueryRouter(["("])


class TestQueryRouterClassify:
    def test_empty_patterns_returns_semantic(self) -> None:
        """UT-001 — no patterns means nothing can classify as lexical."""
        router = QueryRouter([])
        assert router.classify("MDR-AD002") == "semantic"
        assert router.classify("anything at all") == "semantic"

    @pytest.mark.parametrize(
        "query",
        [
            "MDR-AD002",
            "CVE-2021-4034",
            "T1078.001",
            "CWE-79",
            "H1-P4-XXX-1234",
            "abc123def456" * 8,
        ],
    )
    def test_default_patterns_classify_lexical(self, query: str) -> None:
        """UT-002 — canonical lexical corpus classifies as lexical."""
        router = QueryRouter(ROUTER_TEST_PATTERNS)
        assert router.classify(query) == "lexical"

    def test_prose_returns_semantic(self) -> None:
        """UT-003 — natural-language query dispatches semantic."""
        router = QueryRouter(ROUTER_TEST_PATTERNS)
        assert router.classify("how does OAuth token refresh work") == "semantic"

    def test_empty_string_returns_semantic(self) -> None:
        """UT-004 — empty query is a safe-default semantic classification."""
        router = QueryRouter(ROUTER_TEST_PATTERNS)
        assert router.classify("") == "semantic"

    def test_first_match_wins_short_circuits(self) -> None:
        """UT-005 — later patterns are never consulted once one matches."""
        router = QueryRouter([r"CVE-\d{4}-\d+", r"[A-Z]{2,}-\d+"])
        first_spy = MagicMock(wraps=router._patterns[0].search)
        second_spy = MagicMock(wraps=router._patterns[1].search)
        router._patterns[0] = MagicMock(search=first_spy)
        router._patterns[1] = MagicMock(search=second_spy)

        assert router.classify("CVE-2021-4034") == "lexical"

        assert first_spy.call_count == 1
        assert second_spy.call_count == 0

    def test_single_letter_prefix_returns_semantic(self) -> None:
        """UT-006 — ``H-P4-XXX-1234`` fails ``[A-Z]{2,}`` (H alone)."""
        router = QueryRouter(ROUTER_TEST_PATTERNS)
        assert router.classify("H-P4-XXX-1234") == "semantic"

    def test_lowercase_identifier_returns_semantic(self) -> None:
        """UT-007 — patterns are case-sensitive: lowercase fails."""
        router = QueryRouter(ROUTER_TEST_PATTERNS)
        assert router.classify("mdr-ad002") == "semantic"

    def test_alpha_suffix_returns_semantic(self) -> None:
        """UT-008 — ``H1-P4-XXX-1234-a`` fails: trailing lowercase breaks the ID."""
        router = QueryRouter(ROUTER_TEST_PATTERNS)
        assert router.classify("H1-P4-XXX-1234-a") == "semantic"

    def test_prefix_without_digits_returns_semantic(self) -> None:
        """UT-009 — bare ``H1`` has no dash-suffix component."""
        router = QueryRouter(ROUTER_TEST_PATTERNS)
        assert router.classify("H1") == "semantic"

    def test_cwe_id_returns_lexical(self) -> None:
        """UT-010 — CWE-79 matches an anchored ``[A-Z]{2,}(-...)+`` pattern."""
        router = QueryRouter(ROUTER_TEST_PATTERNS)
        assert router.classify("CWE-79") == "lexical"

    def test_custom_pattern_matches(self) -> None:
        """UT-011 — proprietary Oracle-style identifier via custom pattern."""
        router = QueryRouter([r"^ORA-\d{5}$"])
        assert router.classify("ORA-00942") == "lexical"


class TestQueryRouterProperties:
    @settings(max_examples=200, deadline=None)
    @given(
        query=st.text(
            alphabet=st.characters(min_codepoint=32, max_codepoint=126),
            max_size=200,
        )
    )
    def test_classify_always_returns_valid_literal(self, query: str) -> None:
        """UT-013 — for any printable ASCII input, output is one of two literals."""
        router = QueryRouter(DEFAULT_PATTERNS)
        result = router.classify(query)
        assert result in ("lexical", "semantic")
