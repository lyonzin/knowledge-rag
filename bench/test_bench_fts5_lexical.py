"""FTS5 lexical fast-path microbenchmarks (Fase 5, ADR-004 gate).

Tracked metrics for the v4.9.0 default-flip gate:
    bench_fts5_lexical_cold_p95           BENCH-001 target p95 <= 10ms
    bench_fts5_lexical_hot_p95            BENCH-002 target p95 <=  2ms

Regression sentinels (compared against the pre-Fase-5 baseline of
``bench/test_bench_search.py`` via ``scripts/check_perf_regression.py
--threshold 0.02``):
    BENCH-003 — hybrid path with feature ON (<= 2% regression, gate G2)
    BENCH-004 — hybrid path with feature OFF (zero-cost early-return)

BENCH-003 and BENCH-004 are executed by re-running the existing
``bench/test_bench_search.py`` suite with the FTS5 feature toggled via
env override (see ``bench_v4_9_0_gate.md`` for the CI procedure). They
are documented here — not duplicated — so the baseline metric IDs stay
stable between comparisons.

The ``corpus_3865_docs`` fixture from PRD ``_tests.md`` was not shipped
as a shared fixture in earlier fases; this file builds an equivalent
3865-row corpus in-process against a temporary SQLite FTS5 index. That
keeps the bench self-contained and lets CI cells run it without a
FastEmbed/ChromaDB download.
"""

from __future__ import annotations

import pytest

from mcp_server.fts5_index import Fts5LexicalIndex

_CORPUS_SIZE = 3865
_LEXICAL_QUERIES = ["CVE-2021-4034", "MS17-010", "T1078.001", "CWE-79"]


def _seed_fts5_corpus(index: Fts5LexicalIndex) -> None:
    """Populate ``index`` with 3865 synthetic docs carrying identifier tokens.

    Every 100th chunk carries one of the four lexical query identifiers so
    each benchmark query hits ~10 rows — enough to exercise the FTS5 bm25()
    ranker without dominating latency with row-materialisation cost.
    """
    for i in range(_CORPUS_SIZE):
        marker = _LEXICAL_QUERIES[i % len(_LEXICAL_QUERIES)] if i % 100 == 0 else ""
        content = f"document {i} kerberoast bloodhound impacket privilege escalation lateral movement notes on {marker}"
        index.add_document(
            chunk_id=f"chunk-{i}",
            content=content,
            filename=f"doc-{i}.md",
            category="benchmarks",
        )


@pytest.fixture(scope="module")
def fts5_index(tmp_path_factory):
    """Fresh FTS5 index seeded with 3865 rows, closed on teardown."""
    tmp = tmp_path_factory.mktemp("fts5_bench")
    idx = Fts5LexicalIndex(
        db_path=tmp / "fts5_index.db",
        state_path=tmp / "fts5_migration.state",
    )
    _seed_fts5_corpus(idx)
    yield idx
    idx.close()


@pytest.mark.parametrize("query", _LEXICAL_QUERIES)
def test_bench_fts5_lexical_cold(benchmark, fts5_index, query):
    """BENCH-001 — cold FTS5 lookup on a 3865-doc corpus (rerank OFF default).

    Target: p95 <= 10ms per ADR-004 gate G1 (measured post-run by
    ``scripts/check_perf_regression.py`` against the hybrid baseline).
    """

    def run():
        return fts5_index.search(query, top_k=10)

    result = benchmark(run)
    assert isinstance(result, list)


@pytest.mark.parametrize("query", _LEXICAL_QUERIES)
def test_bench_fts5_lexical_hot(benchmark, fts5_index, query):
    """BENCH-002 — hot FTS5 lookup (SQLite page-cache warmed by BENCH-001).

    Target: p95 <= 2ms. Ordering matters — this test relies on the
    ``fts5_index`` module-scoped fixture having been queried already by
    BENCH-001, so pytest's default file-order collection preserves the
    warm-cache semantics.
    """

    def run():
        return fts5_index.search(query, top_k=10)

    result = benchmark(run)
    assert isinstance(result, list)
