# FTS5 Lexical Fast-Path

**Status:** Opt-in in v4.8.2. Default flip to `enabled: true` is gated on the
CI perf-gate procedure documented in ADR-009 and reserved for v4.9.0.

## Overview

The lexical fast-path is a dedicated SQLite FTS5 index optimised for exact
identifier queries — CVE IDs, MITRE ATT&CK codes, VRT/CWE codes, file hashes,
bug-bounty report IDs, error strings. When a query looks lexical, the
`KnowledgeOrchestrator` dispatches to the FTS5 index and skips the full
hybrid pipeline (BM25 + semantic + RRF + reranker), returning results in
under ~10ms cold / ~2ms hot on a 3865-doc corpus.

Non-lexical queries continue to flow through the existing hybrid pipeline
byte-for-byte — the fast-path only intercepts what its regex router
classifies as lexical.

## Quick Start

1. Add the block to your `config.yaml`:

```yaml
search:
  lexical_fast_path:
    enabled: true
```

2. Restart the server. On first start with the flag on, a background
   daemon thread rebuilds the FTS5 index from your existing ChromaDB
   corpus. Progress is checkpointed every 100 rows to
   `data/fts5_migration.state`; queries during migration transparently
   fall back to the hybrid pipeline.

3. Check readiness:

```bash
curl -s http://127.0.0.1:9179/metrics | grep knowledge_rag_fast_path
```

`knowledge_rag_fast_path_migration_docs_indexed` reaching
`_docs_total` marks the fast-path as live.

## Configuration Reference

Full field docs live in `config.example.yaml` under
`search.lexical_fast_path`. The five knobs:

| Field            | Default | What it controls                                              |
| ---------------- | ------- | ------------------------------------------------------------- |
| `enabled`        | `false` | Master toggle. Off → zero runtime cost, no FTS5 index open.   |
| `min_hits`       | `3`     | Minimum FTS5 hits to skip hybrid fallback (recall safety).    |
| `rerank_enabled` | `false` | Layer cross-encoder rerank on FTS5 hits. ADR-003 keeps off.   |
| `patterns`       | *(see)* | First-match-wins regex list that classifies "lexical".        |

Default `patterns`:

```yaml
patterns:
  - "[A-Z]{2,}-\\d+"    # H1-P4-XXX, MDR-AD002, CWE-79, MS17-010
  - "CVE-\\d{4}-\\d+"   # canonical CVE identifiers
  - "^[a-f0-9]{32,64}$" # md5/sha1/sha256 file hashes
```

Add project-specific taxonomies at the end of the list — ordering matters
(PRD OQ-2), first match wins.

## How It Works

Query flow when the fast-path is enabled:

1. `QueryRouter.classify(query)` runs the regex list. Returns `"lexical"` on
   first hit, else `"semantic"` (empty query defaults to `"semantic"`).
2. `"semantic"` → hybrid pipeline unchanged.
3. `"lexical"` → `Fts5LexicalIndex.search(query, top_k)` (SQLite FTS5 `MATCH`
   + `bm25()` ordering). Returns `[(chunk_id, score)]`.
4. If result count `< min_hits` → fall back to hybrid, increment
   `knowledge_rag_fast_path_fallback_total{reason="low_hits"}`.
5. Else → optional rerank pass (only if `rerank_enabled: true`), then
   adjacent-chunk expansion identical to the hybrid path.

The `search_knowledge` MCP tool exposes an override:
`search_method: Literal["auto", "hybrid", "fts5"] = "auto"`.

- `"auto"` (default) — router decides.
- `"hybrid"` — force full hybrid pipeline, ignore router.
- `"fts5"` — force fast-path even for prosa queries. When the index is not
  ready, returns a structured JSON error envelope with a `suggestion` field
  pointing back at `"auto"`.

See ADR-006 for the API surface diff (LEI 1 compatible — additive tail
kwarg, all existing calls unchanged).

## Troubleshooting

**1. All queries fall back to hybrid, fast-path never fires.**
Check `knowledge_rag_fast_path_fallback_total{reason="disabled"}`. Non-zero
means `enabled: false` — verify the config path and restart. If
`{reason="migration_pending"}` is climbing, the background rebuild is still
running; see `_migration_docs_indexed / _docs_total` gauges for progress.

**2. Lexical query returns `NO_RESULTS` but hybrid finds the chunk.**
Router probably misclassified. Confirm with an explicit
`search_method="fts5"` call — if that returns hits, the pattern list is
right and `min_hits` may be too aggressive; lower to `1`. If the forced
`"fts5"` call also empty, the FTS5 index does not contain that chunk — run
`python scripts/build_fts5_index.py --data-dir <path> --force` to rebuild
from ChromaDB.

**3. Latency higher than the 10ms budget.**
Check `knowledge_rag_fast_path_latency_seconds_bucket`. Latency above the
`le="0.010"` bucket generally means `rerank_enabled: true` — the
cross-encoder adds 40-80ms per query (ADR-003). Set `rerank_enabled:
false` and re-measure.

**4. `Fts5NotReadyError` in logs after restart.**
Index migration failed. Inspect `data/fts5_migration.state` (JSON marker
file). `state: "failed"` includes the exception. Remedy: run the standalone
rebuild helper:

```bash
python scripts/build_fts5_index.py --data-dir ./data --force --foreground
```

**5. Storage doubled after enabling.**
Expected during migration — the FTS5 index is a full copy of the corpus in
a dedicated SQLite database. After migration completes, expect
`fts5_index.db` to sit at roughly 30-60% of ChromaDB's on-disk size on a
typical mixed corpus.

## When to Use vs. Not

**Enable when:**
- Your corpus is dominated by exact identifiers (CVEs, CWEs, MITRE codes,
  hashes, bug-bounty report IDs, error codes).
- You measure hybrid-path latency and see the semantic branch adding
  cost you do not need for identifier lookups.
- You want deterministic ranking on identifier queries — BM25 alone is
  reproducible; RRF + semantic can drift as embedding models change.

**Leave off when:**
- Your corpus is prose-heavy and identifier queries are rare (< 5% of
  traffic).
- You depend on the cross-encoder reranker for lexical queries too
  (recall over latency) — the fast-path skips rerank by default.
- You cannot afford the one-time migration cost (5-30 minutes on 3865
  docs, longer on larger corpora).

## References

- ADR-001 — storage layout (`data/fts5_index.db`, PRAGMAs)
- ADR-002 — regex router (first-match-wins, unanchored)
- ADR-003 — rerank OFF default (empirical basis)
- ADR-004 — v4.9.0 default-flip conditional gate
- ADR-005 — tokenizer (`unicode61 remove_diacritics 2 tokenchars '-_.'`)
- ADR-006 — `search_method` MCP tool surface diff
- ADR-008 — CRUD sync + lazy migration (Fase 4)
- ADR-009 — bench gate deferred to CI
- `docs/runbooks/fts5_migration.md` — operator runbook
- `.compozy/tasks/fts5-lexical-fast-path/bench_v4_9_0_gate.md` — gate
  procedure and result template
