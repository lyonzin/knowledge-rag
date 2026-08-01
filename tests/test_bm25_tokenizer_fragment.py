"""Regression tests for BM25 fragment query support.

Bug #140: hyphenated composite tokens ("MDR-AD002", "CVE-2024-1234") were
indexed as single tokens, causing fragment queries to silently return
NO_RESULTS. Fix emits both the composite AND sub-parts (len >= 2) so
fragments match while IDF preserves exact-match ranking.
"""

from mcp_server.server import BM25Index


# ── Unit: tokenizer behavior ──


class TestTokenizerFragments:
    def setup_method(self):
        self.bm25 = BM25Index()

    def test_composite_token_preserved(self):
        """Composite tokens must still be indexed as-is (retrocompat)."""
        tokens = self.bm25._tokenize("MDR-AD002")
        assert "mdr-ad002" in tokens

    def test_sub_tokens_emitted_on_hyphen(self):
        """Sub-parts of hyphenated tokens must be emitted."""
        tokens = self.bm25._tokenize("MDR-AD002")
        assert "mdr" in tokens
        assert "ad002" in tokens

    def test_cve_pattern_emits_all_parts(self):
        """CVE-YYYY-NNNN emits composite + 3 sub-parts."""
        tokens = self.bm25._tokenize("CVE-2024-1234")
        assert "cve-2024-1234" in tokens
        assert "cve" in tokens
        assert "2024" in tokens
        assert "1234" in tokens

    def test_adr_pattern(self):
        """ADR-NNNN also expands (used in project's ADR references)."""
        tokens = self.bm25._tokenize("ADR-0003")
        assert "adr-0003" in tokens
        assert "adr" in tokens
        assert "0003" in tokens

    def test_ms_cve_alias(self):
        """MS17-010 pattern (MS bulletin codes)."""
        tokens = self.bm25._tokenize("MS17-010")
        assert "ms17-010" in tokens
        assert "ms17" in tokens
        assert "010" in tokens

    def test_multi_hyphen_composite(self):
        """MDR-CS-OFHOURS-001 emits composite + 4 sub-parts."""
        tokens = self.bm25._tokenize("MDR-CS-OFHOURS-001")
        assert "mdr-cs-ofhours-001" in tokens
        assert "mdr" in tokens
        assert "cs" in tokens
        assert "ofhours" in tokens
        assert "001" in tokens

    def test_single_char_parts_dropped(self):
        """Sub-parts of length 1 must be dropped to avoid index noise."""
        tokens = self.bm25._tokenize("a-b-c-hello-world")
        assert "a-b-c-hello-world" in tokens
        assert "hello" in tokens
        assert "world" in tokens
        assert tokens.count("a") == 0
        assert tokens.count("b") == 0
        assert tokens.count("c") == 0

    def test_non_hyphenated_unchanged(self):
        """Tokens without hyphens must not expand."""
        assert self.bm25._tokenize("crowdstrike") == ["crowdstrike"]

    def test_dot_separator_still_splits(self):
        """Dot behavior unchanged — still splits tokens."""
        tokens = self.bm25._tokenize("T1078.002")
        assert "t1078" in tokens
        assert "002" in tokens

    def test_empty_and_whitespace(self):
        """Empty / whitespace-only inputs return empty list."""
        assert self.bm25._tokenize("") == []
        assert self.bm25._tokenize("   ") == []

    def test_lowercase_normalization(self):
        """Tokenizer lowercases (existing contract)."""
        tokens = self.bm25._tokenize("MDR-AD002")
        assert "MDR-AD002" not in tokens
        assert "mdr-ad002" in tokens


# ── Integration: fragment query → correct BM25 match ──


class TestFragmentQueryMatching:
    """End-to-end: index → search fragment → return doc."""

    def setup_method(self):
        self.bm25 = BM25Index()
        self.bm25.add_documents(
            chunk_ids=["doc1"],
            texts=["MDR-AD002 detecta alteracao de senha nao autorizada"],
        )
        self.bm25.build_index()

    def test_fragment_ad002_matches(self):
        """Fragment 'ad002' must return the MDR-AD002 doc."""
        results = self.bm25.search("ad002", top_k=5)
        assert len(results) > 0
        assert results[0][0] == "doc1"

    def test_composite_mdr_ad002_still_matches(self):
        """Composite query continues to match (retrocompat)."""
        results = self.bm25.search("mdr-ad002", top_k=5)
        assert len(results) > 0
        assert results[0][0] == "doc1"

    def test_uppercase_query_matches(self):
        """Case-insensitive — uppercase fragment works."""
        results = self.bm25.search("AD002", top_k=5)
        assert len(results) > 0
        assert results[0][0] == "doc1"

    def test_composite_ranks_higher_than_fragment(self):
        """Exact composite match must score >= fragment match with realistic IDF.

        Requires a multi-doc corpus so IDF is not degenerate. In a corpus where
        the composite "mdr-ad002" is rare (1 doc) but "ad002" appears as part
        of many composites (mdr-ad002, mdr-ad001, mdr-ad003...), the composite
        query still ranks the exact-match doc highest — via composite token
        appearing exclusively in that doc.
        """
        bm25 = BM25Index()
        bm25.add_documents(
            chunk_ids=[f"doc{i}" for i in range(10)],
            texts=[
                "MDR-AD002 detecta alteracao de senha",  # target
                "MDR-AD001 - criacao de usuario nao autorizada",
                "MDR-AD003 - exclusao de usuario",
                "MDR-AD010 - login fora de horario",
                "MDR-AD019 - ataque pass-the-hash",
                "documento generico de logs de auditoria",
                "procedimentos de deteccao geral",
                "watchlist de eventos de logon",
                "ADR-0003 sobre plugins hibridos",
                "referencia de eventos active directory",
            ],
        )
        bm25.build_index()

        composite_results = bm25.search("mdr-ad002", top_k=3)
        fragment_results = bm25.search("ad002", top_k=3)

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
            chunk_ids=["mdr002", "mdr010", "cve24"],
            texts=[
                "MDR-AD002 - Alteracao de senha de usuario nao autorizada",
                "MDR-AD010 - Login fora de horario comercial watchlist",
                "CVE-2024-1234 - Vulnerabilidade critica de teste",
            ],
        )
        self.bm25.build_index()

    def test_ad002_returns_correct_doc(self):
        """Fragment AD002 must return mdr002 doc, not mdr010."""
        results = self.bm25.search("ad002", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "mdr002"

    def test_ad010_returns_correct_doc(self):
        """Fragment AD010 must return mdr010 doc, not mdr002."""
        results = self.bm25.search("ad010", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "mdr010"

    def test_cve_2024_finds_cve_doc(self):
        """Fragment 2024 must return CVE doc (only one containing that year)."""
        results = self.bm25.search("2024", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "cve24"


class TestReporterListedQueryPatterns:
    """Explicit tests for query patterns Ailton called out in issue #140.

    Every one of these MUST return the correct doc, without the user needing
    to know the tokenizer's internal representation.

    Patterns covered:
    1. Full composite: MDR-AD019
    2. Multi-hyphen composite: MDR-Custom001-xxxx
    3. Prefix-only family: MDR (all MDR-* docs)
    4. Fragment: AD019
    5. Short-code fragment: CS005 (matches MDR-CS005)
    """

    def setup_method(self):
        self.bm25 = BM25Index()
        self.bm25.add_documents(
            chunk_ids=["ad019", "custom", "ad002", "cs005", "unrelated"],
            texts=[
                "MDR-AD019 - Ataque Pass-the-hash WinRM detection",
                "MDR-Custom001-xxxx - custom internal detection rule",
                "MDR-AD002 - Alteracao de senha de usuario",
                "MDR-CS005 - Cloud storage exfiltration detection",
                "documento generico sobre logs de auditoria e observabilidade",
            ],
        )
        self.bm25.build_index()

    def test_query_full_composite_mdr_ad019(self):
        """Pattern 1: full composite `MDR-AD019` must return ad019 doc first."""
        results = self.bm25.search("MDR-AD019", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "ad019"

    def test_query_multi_hyphen_composite_mdr_custom001_xxxx(self):
        """Pattern 2: multi-hyphen composite `MDR-Custom001-xxxx` matches custom rule."""
        results = self.bm25.search("MDR-Custom001-xxxx", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "custom"

    def test_query_prefix_only_mdr_returns_mdr_family(self):
        """Pattern 3: prefix-only `MDR` must return all 4 MDR-* docs, not 'unrelated'."""
        results = self.bm25.search("MDR", top_k=5)
        top_ids = {r[0] for r in results[:4]}
        assert top_ids == {"ad019", "custom", "ad002", "cs005"}, (
            f"Expected all 4 MDR-* docs, got {top_ids}"
        )
        # The 'unrelated' doc must NOT rank in the MDR family
        if len(results) >= 5:
            assert results[4][0] == "unrelated"

    def test_query_fragment_ad019_returns_correct_doc(self):
        """Pattern 4: fragment `AD019` must return ad019 doc (not ad002)."""
        results = self.bm25.search("AD019", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "ad019"

    def test_query_short_code_fragment_cs005(self):
        """Pattern 5: short-code fragment `CS005` must return MDR-CS005 doc."""
        results = self.bm25.search("CS005", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "cs005"

    def test_query_short_code_fragment_lowercase(self):
        """Fragment queries are case-insensitive (`cs005` == `CS005`)."""
        results = self.bm25.search("cs005", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "cs005"

    def test_all_reported_patterns_hit_expected_target(self):
        """Regression sentinel: all listed patterns from #140 return non-empty."""
        for query in ["MDR-AD019", "MDR-Custom001-xxxx", "MDR", "AD019", "CS005"]:
            results = self.bm25.search(query, top_k=3)
            assert len(results) > 0, f"Query '{query}' returned empty — regression!"
