"""Regression coverage for #145 — BM25 tokenizer must not drop non-ASCII scripts.

Before this fix, ``BM25Index._tokenize`` and ``_metadata_path_score`` used
the character class ``[a-z0-9]``. Every code point outside Basic Latin
(Cyrillic, Greek, CJK, Arabic — and, silently, Portuguese/Spanish/French
letters with diacritics) matched nothing, so:

* Cyrillic / Greek / CJK docs produced an empty token list — BM25 half of
  hybrid search never fired for them. Retrieval degraded to semantic-only
  without any warning.
* Latin-with-diacritics text (e.g. Portuguese) was **fragmented** on the
  accent — ``"informação técnica"`` tokenized as
  ``['informa', 'o', 't', 'cnica']``, so ``bm25_rank`` matched garbage
  fragments instead of the intended words.

Fix: use ``[^\\W_]+(?:-[^\\W_]+)*`` — Unicode-aware character class that
keeps composite hyphenated tokens (``CVE-2024-1234``, ``rule-a002``,
``pass-the-hash``) intact while also matching letters from any script.

This module covers ONLY the Unicode aspect. ASCII behaviour is exercised
by ``tests/test_bm25_tokenizer_fragment.py`` (28 tests), which must stay
green — those tests are the anti-regression guard for the fix here.
"""

from __future__ import annotations

import pytest

from mcp_server.server import BM25Index, _metadata_path_score


@pytest.fixture
def idx() -> BM25Index:
    return BM25Index()


# ---------------------------------------------------------------------------
# _tokenize — every script produces at least one token from a non-empty word
# ---------------------------------------------------------------------------


class TestTokenizeUnicode:
    def test_cyrillic_produces_tokens(self, idx: BM25Index) -> None:
        tokens = idx._tokenize("кассовый разрыв")
        assert tokens == ["кассовый", "разрыв"]

    def test_cjk_produces_a_single_token(self, idx: BM25Index) -> None:
        # CJK scripts lack word separators, so we accept a single token —
        # what matters is that BM25 gets *something* rather than an empty
        # list. Word segmentation is a separate concern (jieba, mecab …).
        tokens = idx._tokenize("中文文档")
        assert tokens == ["中文文档"]

    def test_greek_produces_tokens(self, idx: BM25Index) -> None:
        assert idx._tokenize("ασφάλεια δεδομένων") == ["ασφάλεια", "δεδομένων"]

    def test_arabic_produces_tokens(self, idx: BM25Index) -> None:
        assert idx._tokenize("أمن المعلومات") == ["أمن", "المعلومات"]

    def test_portuguese_with_diacritics_is_not_fragmented(self, idx: BM25Index) -> None:
        # This is the regression that most Portuguese docs hit silently:
        # accented words were split on every diacritic (ç, ã, é …) and BM25
        # scored the garbage fragments instead of the real word.
        assert idx._tokenize("informação técnica máxima") == [
            "informação",
            "técnica",
            "máxima",
        ]

    def test_mixed_script_preserves_each_word(self, idx: BM25Index) -> None:
        # A common shape in real docs: English identifiers embedded in a
        # non-English sentence.
        assert idx._tokenize("análise de CVE-2024-1234 no ambiente") == [
            "análise",
            "de",
            "cve-2024-1234",
            # sub-token expansion still fires on hyphenated codes with digits
            "cve",
            "2024",
            "1234",
            "no",
            "ambiente",
        ]

    def test_diacritics_do_not_break_composite_hyphenation(self, idx: BM25Index) -> None:
        # The hyphenated-composite path must keep working when the composite
        # itself contains Unicode letters (rare, but possible in trademark
        # naming, e.g. Japanese product codes).
        assert idx._tokenize("préfix-suffixe") == ["préfix-suffixe"]

    def test_underscore_still_separates_tokens(self, idx: BM25Index) -> None:
        # The old regex excluded underscore by omission; the new class
        # excludes it explicitly (``[^\W_]``). Keep parity so identifiers
        # like ``foo_bar_baz`` still tokenise as three words rather than one.
        assert idx._tokenize("foo_bar_baz") == ["foo", "bar", "baz"]


# ---------------------------------------------------------------------------
# _metadata_path_score — the two other call-sites of the same regex
# ---------------------------------------------------------------------------


class TestMetadataPathScoreUnicode:
    """Path/filename metadata scoring must also honour Unicode filepaths."""

    def test_cyrillic_query_matches_cyrillic_filename(self) -> None:
        meta = {"source": "/docs/ru/руководство.md", "filename": "руководство.md"}
        assert _metadata_path_score("руководство", meta) > 0.0

    def test_portuguese_accented_query_matches_accented_filename(self) -> None:
        meta = {"source": "/docs/pt-br/configuração.md", "filename": "configuração.md"}
        assert _metadata_path_score("configuração", meta) > 0.0

    def test_ascii_query_still_scores_zero_against_pure_unicode_path(self) -> None:
        # Sanity: ASCII query without any tokens present in a Cyrillic-only
        # path scores zero. Confirms the fix does not add spurious matches.
        meta = {"source": "/docs/ru/руководство.md", "filename": "руководство.md"}
        assert _metadata_path_score("english query terms", meta) == 0.0


# ---------------------------------------------------------------------------
# End-to-end — non-Latin docs actually retrievable via hybrid BM25 fusion
# ---------------------------------------------------------------------------


class TestBM25EndToEndUnicode:
    def test_cyrillic_corpus_returns_matching_doc(self, idx: BM25Index) -> None:
        # Before the fix this returned zero results because the corpus
        # tokenised to an empty list — the inverted index had nothing to
        # look up.
        docs = [
            ("doc-ru-1", "кассовый разрыв — оперативный отчёт"),
            ("doc-ru-2", "план миграции базы данных"),
            ("doc-en", "the quick brown fox jumps over the lazy dog"),
        ]
        idx.add_documents(
            chunk_ids=[cid for cid, _ in docs],
            texts=[txt for _, txt in docs],
        )
        idx.build_index()

        results = idx.search("кассовый", top_k=3)
        matched_ids = [chunk_id for chunk_id, _score in results]
        assert "doc-ru-1" in matched_ids
        # Unrelated english doc must not match a purely-cyrillic query.
        assert "doc-en" not in matched_ids

    def test_portuguese_corpus_returns_matching_doc(self, idx: BM25Index) -> None:
        docs = [
            ("doc-pt-1", "análise técnica da configuração do servidor"),
            ("doc-pt-2", "guia de instalação e manutenção"),
            ("doc-en", "server configuration best practices"),
        ]
        idx.add_documents(
            chunk_ids=[cid for cid, _ in docs],
            texts=[txt for _, txt in docs],
        )
        idx.build_index()

        results = idx.search("configuração", top_k=3)
        matched_ids = [chunk_id for chunk_id, _score in results]
        assert "doc-pt-1" in matched_ids
        # Even though doc-en talks about "configuration", it's a different
        # token in the index — BM25 does not fold diacritics silently.
        assert "doc-en" not in matched_ids
