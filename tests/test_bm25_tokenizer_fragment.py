"""Regression tests for BM25 fragment query support.

Bug #140: hyphenated composite tokens (e.g. `RULE-A001`, `CVE-2024-1234`) were
indexed as single tokens, causing fragment queries to silently return
NO_RESULTS. Fix emits both the composite AND sub-parts (len >= 2) so
fragments match while IDF preserves exact-match ranking.

All examples in this file use synthetic identifiers (`RULE-*`, `PROJECT-*`)
or well-known public identifiers (`CVE-*`, `MS17-*`, `ADR-*`, `T1078.002`).
No corporate / customer-specific taxonomy appears in test fixtures.
"""

from mcp_server.server import BM25Index

# ── Unit: tokenizer behavior ──


class TestTokenizerFragments:
    def setup_method(self):
        self.bm25 = BM25Index()

    def test_composite_token_preserved(self):
        """Composite tokens must still be indexed as-is (retrocompat)."""
        tokens = self.bm25._tokenize("RULE-A002")
        assert "rule-a002" in tokens

    def test_sub_tokens_emitted_on_hyphen(self):
        """Sub-parts of hyphenated tokens must be emitted."""
        tokens = self.bm25._tokenize("RULE-A002")
        assert "rule" in tokens
        assert "a002" in tokens

    def test_cve_pattern_emits_all_parts(self):
        """CVE-YYYY-NNNN emits composite + 3 sub-parts."""
        tokens = self.bm25._tokenize("CVE-2024-1234")
        assert "cve-2024-1234" in tokens
        assert "cve" in tokens
        assert "2024" in tokens
        assert "1234" in tokens

    def test_adr_pattern(self):
        """ADR-NNNN also expands (Architecture Decision Record naming)."""
        tokens = self.bm25._tokenize("ADR-0003")
        assert "adr-0003" in tokens
        assert "adr" in tokens
        assert "0003" in tokens

    def test_ms_cve_alias(self):
        """MS17-010 pattern (Microsoft bulletin codes — public)."""
        tokens = self.bm25._tokenize("MS17-010")
        assert "ms17-010" in tokens
        assert "ms17" in tokens
        assert "010" in tokens

    def test_multi_hyphen_composite(self):
        """PROJECT-A-EXAMPLE-001 emits composite + 3 sub-parts (single-char dropped)."""
        tokens = self.bm25._tokenize("PROJECT-A-EXAMPLE-001")
        assert "project-a-example-001" in tokens
        assert "project" in tokens
        # Single-char parts ("A") must be dropped
        assert tokens.count("a") == 0
        assert "example" in tokens
        assert "001" in tokens

    def test_single_char_parts_dropped(self):
        """Sub-parts of length 1 must be dropped to avoid index noise.

        Uses a code with digits (so expansion is triggered) whose sub-parts
        include single chars like 'a' that must be filtered out.
        """
        tokens = self.bm25._tokenize("a-b-hello-42-world")
        assert "a-b-hello-42-world" in tokens
        assert "hello" in tokens
        assert "world" in tokens
        assert "42" in tokens
        # Single-char parts must be dropped
        assert tokens.count("a") == 0
        assert tokens.count("b") == 0

    def test_non_hyphenated_unchanged(self):
        """Tokens without hyphens must not expand."""
        assert self.bm25._tokenize("elasticsearch") == ["elasticsearch"]

    def test_dot_separator_still_splits(self):
        """Dot behavior unchanged — still splits tokens."""
        tokens = self.bm25._tokenize("T1078.002")
        assert "t1078" in tokens
        assert "002" in tokens

    def test_natural_language_hyphen_not_expanded(self):
        """Hyphenated natural-language phrases (no digits) must NOT expand.

        Words like "state-of-the-art" or "end-to-end" should stay as single
        composite tokens — expanding them floods the index with stop-word-like
        sub-parts and hurts query throughput without helping typical recall.
        """
        tokens = self.bm25._tokenize("state-of-the-art detection engine")
        assert "state-of-the-art" in tokens
        # Sub-parts must NOT be emitted (they'd match too broadly)
        assert "state" not in tokens
        assert "art" not in tokens
        assert "the" not in tokens

    def test_natural_language_multi_hyphen_not_expanded(self):
        """Multi-hyphen natural phrase like 'end-to-end' stays composite."""
        tokens = self.bm25._tokenize("end-to-end pipeline")
        assert "end-to-end" in tokens
        assert "end" not in tokens

    def test_alphanumeric_code_still_expands(self):
        """Regression sentinel: codes with digits keep expanding after the
        digit-required heuristic (would silently break the fix otherwise)."""
        for code, must_include in [
            ("RULE-A002", ["rule", "a002"]),
            ("CVE-2024-1234", ["cve", "2024", "1234"]),
            ("MS17-010", ["ms17", "010"]),
            ("PROJECT-B005", ["project", "b005"]),
        ]:
            tokens = self.bm25._tokenize(code)
            for part in must_include:
                assert part in tokens, f"{code} did not emit sub-token '{part}'"

    def test_empty_and_whitespace(self):
        """Empty / whitespace-only inputs return empty list."""
        assert self.bm25._tokenize("") == []
        assert self.bm25._tokenize("   ") == []

    def test_lowercase_normalization(self):
        """Tokenizer lowercases (existing contract)."""
        tokens = self.bm25._tokenize("RULE-A002")
        assert "RULE-A002" not in tokens
        assert "rule-a002" in tokens


# ── Integration: fragment query → correct BM25 match ──


class TestFragmentQueryMatching:
    """End-to-end: index → search fragment → return doc."""

    def setup_method(self):
        self.bm25 = BM25Index()
        self.bm25.add_documents(
            chunk_ids=["doc1"],
            texts=["RULE-A002 detects example configuration change events"],
        )
        self.bm25.build_index()

    def test_fragment_a002_matches(self):
        """Fragment 'a002' must return the RULE-A002 doc."""
        results = self.bm25.search("a002", top_k=5)
        assert len(results) > 0
        assert results[0][0] == "doc1"

    def test_composite_rule_a002_still_matches(self):
        """Composite query continues to match (retrocompat)."""
        results = self.bm25.search("rule-a002", top_k=5)
        assert len(results) > 0
        assert results[0][0] == "doc1"

    def test_uppercase_query_matches(self):
        """Case-insensitive — uppercase fragment works."""
        results = self.bm25.search("A002", top_k=5)
        assert len(results) > 0
        assert results[0][0] == "doc1"

    def test_composite_ranks_higher_than_fragment(self):
        """Exact composite match must score >= fragment match with realistic IDF.

        Requires a multi-doc corpus so IDF is not degenerate. In a corpus where
        the composite "rule-a002" is rare (1 doc) but "a002" appears as part
        of many composites (rule-a002, rule-a001, rule-a003...), the composite
        query still ranks the exact-match doc highest — via composite token
        appearing exclusively in that doc.
        """
        bm25 = BM25Index()
        bm25.add_documents(
            chunk_ids=[f"doc{i}" for i in range(10)],
            texts=[
                "RULE-A002 detects example event type 2",  # target
                "RULE-A001 - example event type 1",
                "RULE-A003 - example event type 3",
                "RULE-A010 - example event type 10",
                "RULE-A019 - example event type 19",
                "generic log correlation document",
                "detection engineering procedures",
                "watchlist of high-value entities",
                "ADR-0003 about plugin architecture",
                "reference material on log parsing",
            ],
        )
        bm25.build_index()

        composite_results = bm25.search("rule-a002", top_k=3)
        fragment_results = bm25.search("a002", top_k=3)

        assert composite_results[0][0] == "doc0", "composite must find target doc first"
        assert fragment_results[0][0] == "doc0", "fragment must find target doc first"
        assert composite_results[0][1] >= fragment_results[0][1], (
            f"composite score ({composite_results[0][1]}) must be >= "
            f"fragment score ({fragment_results[0][1]}) — composite is rarer, higher IDF"
        )


class TestMultiDocFragmentDisambiguation:
    """Fragment queries across multiple docs return correct ranking."""

    def setup_method(self):
        self.bm25 = BM25Index()
        self.bm25.add_documents(
            chunk_ids=["r002", "r010", "cve24"],
            texts=[
                "RULE-A002 - example detection rule number two",
                "RULE-A010 - example detection rule number ten",
                "CVE-2024-1234 - dummy vulnerability reference",
            ],
        )
        self.bm25.build_index()

    def test_a002_returns_correct_doc(self):
        """Fragment A002 must return r002 doc, not r010."""
        results = self.bm25.search("a002", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "r002"

    def test_a010_returns_correct_doc(self):
        """Fragment A010 must return r010 doc, not r002."""
        results = self.bm25.search("a010", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "r010"

    def test_cve_2024_finds_cve_doc(self):
        """Fragment 2024 must return CVE doc (only one containing that year)."""
        results = self.bm25.search("2024", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "cve24"


class TestReportedQueryPatterns:
    """Explicit tests for query patterns reported in issue #140.

    All fixture identifiers are synthetic (RULE-*, PROJECT-*) or public
    (CVE-*, MS17-*). Every one of these MUST return the correct doc without
    the user needing to know the tokenizer's internal representation.

    Patterns covered:
    1. Full composite: RULE-A019
    2. Multi-hyphen composite: PROJECT-Custom001-xxxx
    3. Prefix-only family: RULE (all RULE-* docs)
    4. Fragment: A019
    5. Short-code fragment: B005 (matches RULE-B005)
    """

    def setup_method(self):
        self.bm25 = BM25Index()
        self.bm25.add_documents(
            chunk_ids=["r019", "custom", "r002", "b005", "unrelated"],
            texts=[
                "RULE-A019 - example detection rule variant nineteen",
                "PROJECT-Custom001-xxxx - custom internal detection rule",
                "RULE-A002 - example detection rule variant two",
                "RULE-B005 - example detection rule variant five (family B)",
                "generic document about logs and observability",
            ],
        )
        self.bm25.build_index()

    def test_query_full_composite_rule_a019(self):
        """Pattern 1: full composite `RULE-A019` must return r019 doc first."""
        results = self.bm25.search("RULE-A019", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "r019"

    def test_query_multi_hyphen_composite_project_custom001_xxxx(self):
        """Pattern 2: multi-hyphen composite `PROJECT-Custom001-xxxx` matches custom rule."""
        results = self.bm25.search("PROJECT-Custom001-xxxx", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "custom"

    def test_query_prefix_only_rule_returns_rule_family(self):
        """Pattern 3: prefix-only `RULE` must return all 4 RULE-* docs, not 'unrelated'."""
        results = self.bm25.search("RULE", top_k=5)
        top_ids = {r[0] for r in results[:4]}
        assert top_ids == {"r019", "custom", "r002", "b005"}, f"Expected all 4 RULE-* docs, got {top_ids}"
        # The 'unrelated' doc must NOT rank in the RULE family
        if len(results) >= 5:
            assert results[4][0] == "unrelated"

    def test_query_fragment_a019_returns_correct_doc(self):
        """Pattern 4: fragment `A019` must return r019 doc (not r002)."""
        results = self.bm25.search("A019", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "r019"

    def test_query_short_code_fragment_b005(self):
        """Pattern 5: short-code fragment `B005` must return RULE-B005 doc."""
        results = self.bm25.search("B005", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "b005"

    def test_query_short_code_fragment_lowercase(self):
        """Fragment queries are case-insensitive (`b005` == `B005`)."""
        results = self.bm25.search("b005", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "b005"

    def test_all_reported_patterns_hit_expected_target(self):
        """Regression sentinel: all listed patterns from #140 return non-empty."""
        for query in ["RULE-A019", "PROJECT-Custom001-xxxx", "RULE", "A019", "B005"]:
            results = self.bm25.search(query, top_k=3)
            assert len(results) > 0, f"Query '{query}' returned empty — regression!"
