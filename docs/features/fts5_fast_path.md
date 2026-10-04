# FTS5 lexical fast path

FTS5 is an optional SQLite index for lexical queries such as identifiers and
error codes. The current default is `enabled: false`; earlier plans to flip the
default were conditional and do not describe current behavior.

The index avoids embedding inference for a successful lexical lookup, but total
request latency also includes ChromaDB hydration, optional reranking, adjacent
context and serialization. Measure those stages on the intended corpus rather
than treating historical benchmark timings as a service guarantee.

## Enable and verify

Add this block to the effective configuration and restart the server:

```yaml
search:
  lexical_fast_path:
    enabled: true
```

The server builds the derived index from existing ChromaDB chunks in a background
thread when migration is required. The configured data directory contains
`fts5_index.db` and `fts5_migration.state`. The migration counters and marker
fields named `docs_indexed` and `docs_total` count **chunks**.

During migration, `search_method="auto"` can fall back to hybrid retrieval.
Explicit `search_method="fts5"` requires a ready index and otherwise returns a
structured error. Test readiness with a known indexed identifier, then verify
its source. A completed marker or matching progress gauges alone are not a
complete integrity check; compare chunk counts and representative queries.

If metrics are enabled, inspect the configured metrics endpoint. Its bind and
access controls are separate from MCP HTTP authentication; see
[configuration](../CONFIGURATION.md#server).

## Configuration

| Field under `search.lexical_fast_path` | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Open and maintain the optional lexical index |
| `min_hits` | `3` | Minimum raw FTS5 hits before accepting the auto path |
| `rerank_enabled` | `false` | Rerank the hydrated candidate pool with the cross-encoder |
| `patterns` | Below | Regexes that classify auto queries as lexical |

```yaml
patterns:
  - "[A-Z]{2,}-\\d+"
  - "CVE-\\d{4}-\\d+"
  - "^[a-f0-9]{32,64}$"
```

Patterns use case-sensitive Python regex matching and are tested in order.
The first match classifies the query as lexical; it does not assign a category.
These expressions do not cover every product's identifier syntax. For example,
`MDR-AD002` needs an additional pattern such as `"MDR-[A-Z]+[0-9]+"`.
Validate custom patterns against both positive and negative examples.

## Query behavior

1. `auto` uses the regex router. Semantic queries go directly to hybrid search.
2. Lexical queries retrieve FTS5 candidates ordered by SQLite BM25.
3. Fewer raw hits than `min_hits` cause an auto fallback. ChromaDB hydrates
   candidates and applies an explicit category filter; an empty useful result
   also triggers fallback.
4. Optional reranking operates on the candidate pool before final truncation.
   The default skips this model cost.
5. Adjacent chunks add context to the selected results.

The `search_knowledge` override has three values:

| Value | Behavior |
| --- | --- |
| `auto` | Router chooses; unavailable or insufficient lexical results fall back |
| `hybrid` | Bypass FTS5 and use the hybrid pipeline |
| `fts5` | Bypass router and `min_hits`; report unavailability explicitly |

Category filtering happens after the bounded FTS5 candidate retrieval. It is not
a folder prefilter, and sparse categories can require a different query or the
semantic path. `hybrid_alpha` does not affect FTS5 ranking.

## Metrics and troubleshooting

The metric prefix is `knowledge_rag_fast_path_`:

| Metric | Interpretation |
| --- | --- |
| `hits_total{path="fts5"}` | Successful uncached FTS5 path executions |
| `fallback_total{reason="low_hits"}` | Insufficient or no useful lexical hits |
| `fallback_total{reason="disabled"}` | Router selected lexical but the index handle was not ready |
| `fallback_total{reason="error"}` | FTS5 execution failed and auto used hybrid |
| `errors_total{error_class="..."}` | Execution errors classified by exception name |
| `latency_seconds_bucket` | SQLite lexical search duration buckets |
| `migration_docs_indexed`, `migration_docs_total` | Migration progress in chunks |

When the feature is disabled, dispatch short-circuits; absence of fallback
metrics does not prove the feature is enabled. Cached results also bypass these
execution counters.

The lexical latency histogram times the SQLite search itself. It excludes
hydration, reranking and response handling, so it cannot establish end-to-end
request p95. Compare client timings as well.

If an explicit FTS5 query misses content found by hybrid search, check the
indexed text, tokenization, selected category and migration state. A miss does
not by itself prove router misclassification or index corruption. Do not lower
`min_hits` solely to make the fast path appear more frequently; check relevance.

## Recovery and lifecycle

Incremental indexing and document mutations synchronize the derived index.
Replaying a chunk ID updates that chunk instead of adding a duplicate. A full
rebuild refreshes FTS5 from the published ChromaDB state.

For a failed or inconsistent migration, stop the server before a standalone
forced rebuild, then run from a source checkout:

```bash
python scripts/build_fts5_index.py --data-dir /absolute/path/to/data --force --foreground
```

Use the directory containing the existing `chroma_db/`, not the documents
directory. `--force` rebuilds the derived FTS5 data; preserve the ChromaDB source.
Check the exit status and verify known queries after restart. Detailed recovery,
marker inspection and count checks are in the
[migration runbook](../runbooks/fts5_migration.md).

FTS5 adds its own database and temporary migration/WAL space. The ratio to
ChromaDB size depends on text, metadata and vector storage; no fixed percentage
is guaranteed.

## Choosing whether to enable it

Enable FTS5 when lexical identifiers are common and measured retrieval quality
remains acceptable. Keep the default while evaluating prose-heavy or unfamiliar
corpora. If cross-encoder reranking materially improves the relevant queries,
test `rerank_enabled` and account for its model and latency costs.

Historical design records are in the repository's FTS5 task documentation.
For current behavior, use this guide, [the API](../API.md), the effective
configuration and the tests for the version being deployed.
