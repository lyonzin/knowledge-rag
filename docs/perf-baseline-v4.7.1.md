# Performance Baseline — v4.7.1

**Captured:** 2026-08-05 (Phase 0 of v4.8.0 planning cycle).
**Purpose:** Reference numbers to compare against for future optimizations
(Phase 3 batch-parallel reindex is the primary consumer).

Nothing here changes runtime behaviour. This document exists so a later
change can be graded objectively: "we spent X hours on Y and moved
throughput from A to B on the same hardware".

---

## Hardware / environment

| Field | Value |
|---|---|
| Host | Windows 11 Pro 26200 (x64) |
| CPU | Local desktop (see host inventory) |
| GPU | Available but ONNX Runtime falls back to CPU (`CUDAExecutionProvider` missing — needs `pip install onnxruntime-gpu`) |
| Python | 3.14.6 |
| knowledge-rag | 4.7.1 (commit at branch base) |
| Embedding model | `BAAI/bge-small-en-v1.5` (384D) via `Qdrant/bge-small-en-v1.5-onnx-Q` |
| Chunker | Markdown-aware, split on `##` headings |

---

## Microbenchmarks (`pytest bench/ --benchmark-only`)

All 12 tests passed. Raw JSON stored at `_src/perf-baseline-v4.7.1.json`.

Numbers in microseconds unless noted. `OPS` is operations/second.

| Test | Median | OPS |
|---|---:|---:|
| `bench_query_cache_miss` | 1.3 us | 754,825 |
| `bench_query_cache_hot` | 1.5 us | 655,434 |
| `bench_query_expansion` | 2.5 us | 385,392 |
| `bench_chunk_markdown_small` (5 KB) | 440 us | 2,227 |
| `bench_parse_json_nested` | 781 us | 1,257 |
| `bench_bm25_query_1k_corpus` | 1.26 ms | 782 |
| `bench_chunk_markdown_large` (100 KB) | 6.16 ms | 161 |
| `bench_concurrent_10_queries` | 25.8 ms | 38 |
| `bench_orchestrator_idle_rss` | 146 ms | 7 |
| `bench_concurrent_50_queries` | 125 ms | 8 |
| `bench_query_cache_5000_entries` | 160 ms | 6 |
| `bench_concurrent_100_queries` | 252 ms | 4 |

Observations:
* Query cache lookup is 3-order-of-magnitude cheaper than a real BM25 hit
  (1.5 us vs 1.26 ms) — the cache is doing its job on repeated queries.
* Chunking a 100 KB markdown takes ~6 ms — indexing a 10 MB corpus
  spends ~600 ms in chunking alone, i.e. chunking is a real cost, not a
  rounding error.
* Concurrent-N scales roughly linearly (10x -> 5x-10x wall time), which
  is the expected shape when the reranker is the serial bottleneck.

---

## Full reindex — `nuclear_rebuild()` on 100 synthetic docs

Corpus: 100 Markdown files, ~2.2 KB each, produced by the harness at
`scratchpad/reindex_baseline.py`. Each file splits into 5 chunks under
the default settings (500 chunks total).

Run with model cache warm (embedding weights already on disk):

| Metric | Value |
|---|---:|
| Docs indexed | 100 |
| Chunks added | 500 |
| Wall clock | 27.0 s |
| Docs/second | 3.7 |
| Chunks/second | 18.5 |

Cold-start (first run of the day, ONNX weights fetched from HF):

| Metric | Value |
|---|---:|
| Wall clock | 41.5 s |
| Docs/second | 2.4 |
| Chunks/second | 12.0 |

Both runs used CPU-only inference — the GPU baseline is a separate
number Phase 3 will need to capture once the CUDA provider is
reinstalled.

---

## What Phase 3 will compare against

The batch-parallel reindex work should move `chunks_per_second` on the
warm run. Anything below 25 chunks/s is a regression; the review target
is 40-60 chunks/s on the same hardware. Cold-start is not a Phase 3
target — model download dominates and lives outside our code.
