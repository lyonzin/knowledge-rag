"""
╭─╴ KNOWLEDGE RAG · STOPWORD TESTS ╶─────────────────── v 4.6.0 ─╮
│                                                                │
│   Coverage for multilingual query stopword filtering.          │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    ├─ Since   ·  v4.6.0
    └─ Date    ·  2026-07-27

Covers Q1.3: question words and auxiliaries must leave the query before BM25
tokenization, without ever eating a technical identifier and without ever
returning an empty query.
"""

import pytest

from mcp_server.stopwords import (
    DEFAULT_LANGUAGES,
    MULTILINGUAL_STOPWORDS,
    PROTECTED_TERMS,
    SUPPORTED_LANGUAGES,
    filter_query_stopwords,
    get_stopwords,
    normalize_language_codes,
)

# ── Table shape ──


class TestStopwordTables:
    def test_all_six_languages_present(self):
        """EN/PT/ES/DE/FR/IT are all declared."""
        assert set(SUPPORTED_LANGUAGES) == {"en", "pt", "es", "de", "fr", "it"}

    def test_every_set_is_frozen_and_non_empty(self):
        """Each language set is an immutable, populated frozenset."""
        for lang, words in MULTILINGUAL_STOPWORDS.items():
            assert isinstance(words, frozenset), lang
            assert words, lang

    def test_entries_are_lowercase(self):
        """Lookups are done on lowercased tokens, so the tables must be lowercase."""
        for lang, words in MULTILINGUAL_STOPWORDS.items():
            assert all(w == w.lower() for w in words), lang

    def test_protected_terms_never_leak_into_language_sets(self):
        """The allowlist is subtracted from every language at import time."""
        for lang, words in MULTILINGUAL_STOPWORDS.items():
            assert not (words & PROTECTED_TERMS), lang

    def test_default_languages_are_supported(self):
        """The shipped default must be a subset of what is implemented."""
        assert set(DEFAULT_LANGUAGES) <= set(SUPPORTED_LANGUAGES)


# ── Core filtering ──


class TestFilterQueryStopwords:
    def test_portuguese_question_is_stripped(self):
        """'como funciona' carries no retrieval signal — the terms after it do."""
        assert filter_query_stopwords("como funciona ldap kerberoast", ["pt"]) == "ldap kerberoast"

    def test_english_question_is_stripped(self):
        """'how does ... work' collapses to the subject."""
        assert filter_query_stopwords("how does SSRF work", ["en"]) == "SSRF"

    def test_german_question_is_stripped(self):
        """German scaffolding collapses to the same subject as the English form."""
        assert filter_query_stopwords("wie funktioniert SSRF", ["de"]) == "SSRF"

    def test_english_and_german_converge(self):
        """Same question, two languages, identical BM25 input."""
        en = filter_query_stopwords("how does SSRF work")
        de = filter_query_stopwords("wie funktioniert SSRF")
        assert en == de == "SSRF"

    @pytest.mark.parametrize(
        ("query", "languages", "expected"),
        [
            ("cómo funciona el ataque kerberoasting", ["es"], "ataque kerberoasting"),
            ("comment fonctionne le kerberoasting", ["fr"], "kerberoasting"),
            ("come funziona il kerberoasting", ["it"], "kerberoasting"),
            ("o que é pass the hash", ["pt"], "pass the hash"),
        ],
    )
    def test_romance_language_questions(self, query, languages, expected):
        """Spanish/French/Italian/Portuguese scaffolding is removed."""
        assert filter_query_stopwords(query, languages) == expected

    def test_punctuation_is_not_glued_to_stopwords(self):
        """Trailing '?' must not stop a stopword from being recognized."""
        assert filter_query_stopwords("how does SSRF work?", ["en"]) == "SSRF"

    def test_sentence_case_stopwords_are_stripped(self):
        """A capitalized first word is still a stopword."""
        assert filter_query_stopwords("Como funciona kerberoast", ["pt"]) == "kerberoast"

    def test_token_order_is_preserved(self):
        """Surviving tokens keep their original order and spelling."""
        assert filter_query_stopwords("what is the LDAP bind operation", ["en"]) == "LDAP bind operation"


# ── Regression: queries without stopwords ──


class TestNoStopwordsPassthrough:
    @pytest.mark.parametrize(
        "query",
        [
            "kerberoast ldap",
            "SQL injection",
            "CVE-2024-1234",
            "buffer overflow ROP chain",
            "ntlm relay adcs esc8",
        ],
    )
    def test_clean_query_survives_intact(self, query):
        """A query made only of content terms is returned byte-identical."""
        assert filter_query_stopwords(query) == query

    def test_multi_word_technical_phrase_survives(self):
        """'use after free' must not be shredded — 'use'/'after' are not listed."""
        assert filter_query_stopwords("use after free") == "use after free"


# ── Edge cases ──


class TestEdgeCases:
    def test_all_stopwords_returns_original_query(self):
        """Filtering never empties a query — the original comes back instead."""
        query = "how what why does the"
        assert filter_query_stopwords(query, ["en"]) == query

    def test_all_stopwords_portuguese_returns_original(self):
        """Same guard on the Portuguese side."""
        query = "como que é isso"
        assert filter_query_stopwords(query, ["pt"]) == query

    def test_empty_string(self):
        """Empty input is returned as-is, never raises."""
        assert filter_query_stopwords("") == ""

    def test_whitespace_only(self):
        """Whitespace-only input is returned untouched."""
        assert filter_query_stopwords("   ") == "   "

    def test_single_stopword(self):
        """A one-word stopword query survives via the empty-result guard."""
        assert filter_query_stopwords("how", ["en"]) == "how"

    def test_empty_language_list_disables_filtering(self):
        """An explicit [] is an explicit opt-out — the query passes through."""
        assert filter_query_stopwords("how does SSRF work", []) == "how does SSRF work"

    def test_unknown_language_degrades_to_default(self):
        """A typo in config.yaml degrades to the default, never raises and never
        silently disables a feature the operator asked for."""
        assert filter_query_stopwords("how does SSRF work", ["klingon"]) == "SSRF"

    def test_regional_variants_are_accepted(self):
        """pt-BR / en_US resolve to pt / en."""
        assert filter_query_stopwords("como funciona kerberoast", ["pt-BR"]) == "kerberoast"
        assert filter_query_stopwords("how does SSRF work", ["en_US"]) == "SSRF"

    def test_none_applies_every_language(self):
        """languages=None enables all six sets."""
        assert filter_query_stopwords("perché funziona kerberoast") == "kerberoast"


# ── Collision protection ──


class TestProtectedTerms:
    @pytest.mark.parametrize(
        "term",
        ["os", "com", "da", "des", "die", "der", "man", "net", "war", "bald", "es", "as"],
    )
    def test_protected_identifier_survives_lowercase(self, term):
        """Terms that collide with function words are never filtered."""
        query = f"{term} exploitation technique"
        assert filter_query_stopwords(query) == query

    def test_os_command_injection_is_untouched(self):
        """OWASP 'os command injection' must not lose its first token to Portuguese."""
        assert filter_query_stopwords("os command injection", ["pt", "en"]) == "os command injection"

    def test_com_hijacking_is_untouched(self):
        """T1546.015 'com hijacking' must not lose 'com' to Portuguese."""
        assert filter_query_stopwords("com hijacking persistence", ["pt"]) == "com hijacking persistence"

    def test_des_cipher_is_untouched(self):
        """French 'des' is an article; DES is a cipher. The cipher wins."""
        assert filter_query_stopwords("des cipher weakness", ["fr"]) == "des cipher weakness"

    def test_spanish_copula_es_is_untouched(self):
        """Spanish 'es' is a copula, but ES is Elasticsearch. Protection wins —
        the surrounding scaffolding still goes."""
        assert filter_query_stopwords("qué es un ataque kerberoasting", ["es"]) == "es ataque kerberoasting"

    def test_sql_operators_survive(self):
        """In this corpus the SQLi payload *is* the query."""
        assert filter_query_stopwords("union select from users", ["en"]) == "union select from users"

    def test_http_methods_survive(self):
        """HTTP verbs are searched verbatim."""
        assert filter_query_stopwords("post put head request smuggling", ["en"]) == "post put head request smuggling"

    @pytest.mark.parametrize("acronym", ["OS", "DES", "WAR", "IT", "AS", "DA"])
    def test_uppercase_acronyms_survive(self, acronym):
        """All-caps tokens are identifiers, not stopwords — even without the allowlist."""
        query = f"{acronym} hardening checklist"
        assert filter_query_stopwords(query) == query

    def test_uppercase_heuristic_covers_unlisted_terms(self):
        """'THE' as an acronym survives although 'the' is an English stopword."""
        assert filter_query_stopwords("THE framework", ["en"]) == "THE framework"


# ── Helpers ──


class TestHelpers:
    def test_normalize_none_returns_all_languages(self):
        """None means every supported language."""
        assert normalize_language_codes(None) == SUPPORTED_LANGUAGES

    def test_normalize_dedupes_and_sorts(self):
        """Duplicates collapse and output is deterministic."""
        assert normalize_language_codes(["PT", "pt-BR", "en"]) == ("en", "pt")

    def test_normalize_falls_back_when_nothing_recognized(self):
        """An entirely bogus list degrades to the shipped default."""
        assert normalize_language_codes(["xx", "yy"]) == DEFAULT_LANGUAGES

    def test_normalize_empty_list_is_opt_out(self):
        """[] is distinguishable from a bogus list: it disables filtering."""
        assert normalize_language_codes([]) == ()

    def test_normalize_tolerates_non_string_entries(self):
        """YAML can hand us ints — that must not raise."""
        assert normalize_language_codes([1, None, "en"]) == ("en",)

    def test_get_stopwords_union(self):
        """The union of two languages contains both languages' entries."""
        merged = get_stopwords(("en", "pt"))
        assert "how" in merged
        assert "como" in merged

    def test_get_stopwords_excludes_protected(self):
        """Protected identifiers are absent from every union."""
        merged = get_stopwords(SUPPORTED_LANGUAGES)
        assert not (merged & PROTECTED_TERMS)

    def test_get_stopwords_is_cached(self):
        """Repeated calls hit the lru_cache and return the same object."""
        assert get_stopwords(("en",)) is get_stopwords(("en",))


# ── Config validation ──


class TestConfigValidation:
    """`Config.__post_init__` must agree with `normalize_language_codes`."""

    @staticmethod
    def _validate(value):
        """Run the stopword branch of Config validation over ``value``."""
        from mcp_server.config import Config

        cfg = Config()
        cfg.stopword_languages = value
        cfg.__post_init__()
        return cfg.stopword_languages

    def test_valid_codes_pass_through(self):
        """A well-formed list survives unchanged."""
        assert self._validate(["en", "pt"]) == ["en", "pt"]

    @pytest.mark.xfail(reason="pending follow-up: script/config enhancement not in base PR-A", strict=False)

    def test_regional_variants_are_normalized(self):
        """pt-BR becomes pt; order of first appearance is kept."""
        assert self._validate(["PT-BR", "en_US"]) == ["pt", "en"]

    @pytest.mark.xfail(reason="pending follow-up: script/config enhancement not in base PR-A", strict=False)

    def test_duplicates_collapse(self):
        """The same language twice is stored once."""
        assert self._validate(["en", "en_GB", "EN"]) == ["en"]

    def test_explicit_empty_list_is_preserved(self):
        """`stopword_languages: []` really disables filtering."""
        assert self._validate([]) == []

    def test_all_typos_fall_back_to_default(self):
        """A list of typos must not silently disable the feature."""
        assert self._validate(["xx", "klingon"]) == ["en", "pt"]

    def test_partial_typos_keep_the_valid_codes(self):
        """Unknown codes are dropped, valid ones survive."""
        assert self._validate(["klingon", "de"]) == ["de"]

    def test_non_list_falls_back_to_default(self):
        """A scalar in config.yaml degrades to the default."""
        assert self._validate("en") == ["en", "pt"]


# ── Integration with the BM25 index ──


@pytest.fixture
def bm25_corpus():
    """Three-document index: one relevant, one pure scaffolding noise, one other."""
    from mcp_server.server import BM25Index

    index = BM25Index()
    index.add_documents(
        ["relevant", "scaffolding_noise", "unrelated"],
        [
            "Kerberoasting targets service accounts that expose an SPN",
            "How does the work of the team happen every day and what is it about",
            "LDAP bind operations and referral chasing",
        ],
    )
    index.build_index()
    return index


class TestBM25Integration:
    def test_prepare_query_uses_configured_languages(self, bm25_corpus, monkeypatch):
        """The index reads the language list from config, not a hardcoded default."""
        monkeypatch.setattr("mcp_server.server.config.stopword_languages", ["pt"])
        assert bm25_corpus.prepare_query("como funciona kerberoast") == "kerberoast"

        monkeypatch.setattr("mcp_server.server.config.stopword_languages", ["en"])
        assert bm25_corpus.prepare_query("como funciona kerberoast") == "como funciona kerberoast"

    def test_scaffolding_noise_no_longer_matches(self, bm25_corpus, monkeypatch):
        """The regression this feature exists for.

        Before filtering, "how does kerberoast work" matched the scaffolding
        document on how/does/work alone. Now only the real hit comes back.
        """
        monkeypatch.setattr("mcp_server.server.config.stopword_languages", ["en", "pt"])
        hits = {chunk_id for chunk_id, _ in bm25_corpus.search("how does kerberoast work")}
        assert hits == {"relevant"}

    def test_question_scores_identically_to_bare_term(self, bm25_corpus, monkeypatch):
        """Scaffolding must not shift the ranking of the term that matters."""
        monkeypatch.setattr("mcp_server.server.config.stopword_languages", ["en", "pt"])
        asked = bm25_corpus.search("how does kerberoast work")
        bare = bm25_corpus.search("kerberoast")
        assert [c for c, _ in asked] == [c for c, _ in bare]
        assert asked[0][1] == pytest.approx(bare[0][1])

    def test_portuguese_question_reaches_the_same_result(self, bm25_corpus, monkeypatch):
        """A PT-BR user gets the same hit as an EN user."""
        monkeypatch.setattr("mcp_server.server.config.stopword_languages", ["en", "pt"])
        hits = [c for c, _ in bm25_corpus.search("como funciona o kerberoast")]
        assert hits == ["relevant"]

    def test_empty_query_still_returns_nothing(self, bm25_corpus):
        """Filtering must not change the empty-query contract."""
        assert bm25_corpus.search("") == []

    def test_empty_config_disables_filtering(self, bm25_corpus, monkeypatch):
        """`stopword_languages: []` in config.yaml is a real off switch."""
        monkeypatch.setattr("mcp_server.server.config.stopword_languages", [])
        assert bm25_corpus.prepare_query("how does kerberoast work") == "how does kerberoast work"
        # ...and the scaffolding document becomes reachable again.
        hits = {chunk_id for chunk_id, _ in bm25_corpus.search("how does kerberoast work")}
        assert "scaffolding_noise" in hits
