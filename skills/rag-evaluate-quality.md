---
name: rag-evaluate-quality
description: Periodically measure the retrieval quality of the knowledge base using evaluate_retrieval (MRR@5, Recall@5, Precision@5) plus get_index_stats for health metrics. Run weekly, after significant reindex activity, or when the user reports declining answer quality. Prevents silent index rot and grounds "should we tune X" decisions in real numbers.
metadata:
  type: rag-workflow
  kind: maintenance
  target: any-mcp-client
---

# rag-evaluate-quality — measure, do not guess

## When to use this skill

Trigger this skill:

- **Weekly cadence** — set a recurring reminder (Monday morning, Friday afternoon)
- **After significant reindex activity** — new content added, models changed, presets swapped
- **When answer quality feels off** — the user reports "search is worse than it used to be"
- **After a version upgrade** — `pip install -U knowledge-rag` bump, worth confirming no regression
- **Before proposing tuning changes** — if you are about to suggest `hybrid_alpha=0.5` or `min_score=0.3`, measure first

**Do NOT run:**

- Every session (waste of cycles; retrieval quality is stable session-to-session)
- On brand-new empty corpora (nothing to evaluate)

---

## What this skill commits to

The agent produces a **numeric quality report**, not a vibe check. Outputs:

- **Index health** — chunks, cache hit rate, embedding model, dimensions
- **Retrieval metrics** — MRR@5, Recall@5, Precision@5 across a small set of representative queries
- **Interpretation** — what the numbers mean, whether they moved vs last run, what to do about it

Numbers get logged so the trend is visible over time.

---

## Steps

1. **Snapshot index health:**
   ```
   get_index_stats()
   ```
   Capture:
   - `documents_count`, `chunks_count`
   - `cache_hit_rate` (higher after warmup = healthy)
   - `embedding_model`, `embedding_dim`

2. **Prepare or reuse an evaluation set.** knowledge-rag's `evaluate_retrieval` tool needs test queries + expected documents (ground truth). Two options:

   - **Reuse a canonical set** — if the project already has `tests/evaluation-queries.json` or similar, load it.
   - **Build a quick set inline** — 5-10 queries that a domain expert (or the user) knows the "correct" answer document for.

   Format (per the API contract):
   ```json
   [
     {"query": "authentication design", "expected_docs": ["docs/adr/0018-auth.md"]},
     {"query": "retry policy", "expected_docs": ["docs/adr/0031-retries.md"]},
     ...
   ]
   ```

3. **Run the evaluation:**
   ```
   evaluate_retrieval(test_queries=<the-json-array>)
   ```
   Response includes:
   - `mrr_at_5` — Mean Reciprocal Rank (0-1; higher is better; >0.7 is good)
   - `recall_at_5` — fraction of expected docs found in top-5 (0-1; >0.8 is good)
   - `precision_at_5` — fraction of top-5 that are relevant (0-1; >0.4 is good)
   - Per-query breakdown

4. **Interpret honestly:**

   | Metric | Value | Interpretation |
   |---|---|---|
   | MRR@5 | < 0.5 | Poor — expected docs rarely in top result |
   | MRR@5 | 0.5–0.7 | OK — often top-3 but not top-1 |
   | MRR@5 | > 0.7 | Good — expected doc usually first |
   | Recall@5 | < 0.5 | Poor — expected docs missing from top-5 half the time |
   | Recall@5 | 0.5–0.8 | OK |
   | Recall@5 | > 0.8 | Good |
   | Precision@5 | < 0.3 | Poor — top-5 mostly irrelevant |
   | Precision@5 | 0.3–0.5 | OK |
   | Precision@5 | > 0.5 | Good |

5. **Compare to prior runs** if there is a history log (e.g. `docs/eval-history.md`). Movement of ±0.02 = noise, ±0.05 = signal.

6. **Recommend action based on findings:**

   | Symptom | Likely cause | Recommendation |
   |---|---|---|
   | MRR@5 drops after upgrade | Embedding model changed | Confirm `models.embedding.model` matches prior; consider `reindex_documents(force=True)` |
   | Recall@5 drops after adding new docs | New docs shifted the vector distribution | Reindex; check if a different `embedding.profile` fits better |
   | Precision@5 drops | Corpus grew but queries not more specific | Tune `min_score`, add query expansions |
   | All metrics low from day 1 | Model / language mismatch | English `bge-small-en-v1.5` on multilingual corpus? Switch to `multilingual` profile |
   | `cache_hit_rate` = 0% after 100+ queries | Queries not repeating (fine) OR cache misconfigured | Only worry if you expected repeats |

7. **Log the run.** Ideally to a persistent `docs/eval-history.md` in the repo (indexed alongside the corpus itself — meta-reflexive but useful):
   ```markdown
   ## 2026-08-15
   - Corpus: 3865 docs / 39241 chunks
   - MRR@5: 0.78 (▲ +0.03 vs 2026-08-08)
   - Recall@5: 0.85 (▬)
   - Precision@5: 0.52 (▬)
   - Notes: added 200 new ADRs this week; MRR up because ADRs are more targeted.
   ```

   Then chain to [`rag-index-decisions`](rag-index-decisions.md) to make the log part of the searchable corpus.

---

## Examples

**Example 1 — Weekly checkup**

User: *"Weekly RAG checkup — how are we doing?"*

Agent:
```
get_index_stats() → 3865 docs / 39241 chunks, cache 34%, bge-small-en-v1.5 384D
Loaded tests/evaluation-queries.json (12 queries)
evaluate_retrieval(test_queries=<the 12>) →
  MRR@5: 0.79, Recall@5: 0.83, Precision@5: 0.48

Compared to last week [docs/eval-history.md]:
  MRR: 0.79 vs 0.77 (▲ +0.02, noise)
  Recall: 0.83 vs 0.85 (▽ -0.02, noise)
  Precision: 0.48 vs 0.50 (▽ -0.02, noise)

Reply: "Health check green. All 3 metrics moved within noise (±0.02).
No action needed.

- Index: 39K chunks, up from 38K last week (organic growth)
- Cache: 34% hit rate (healthy for a mixed-topic session)
- Retrieval: MRR 0.79 (top result usually correct), Recall 0.83 (5-of-6
  expected docs surface in top-5)

Trend log updated: docs/eval-history.md"
```

**Example 2 — Post-reindex confirmation**

User: *"I just did nuclear_rebuild after switching to bge-large. Did anything break?"*

Agent:
```
get_index_stats() → 3865 docs / 39241 chunks, embedding: bge-large-en-v1.5 (1024D)
evaluate_retrieval(test_queries=<the 12>) →
  MRR@5: 0.83, Recall@5: 0.88, Precision@5: 0.55

Prior baseline (bge-small): MRR 0.79, Recall 0.83, Precision 0.48

Reply: "Nothing broke. Quality actually IMPROVED across all 3 metrics:
  MRR@5:       0.79 → 0.83  (▲ +0.04, real signal)
  Recall@5:    0.83 → 0.88  (▲ +0.05, real signal)
  Precision@5: 0.48 → 0.55  (▲ +0.07, strong signal)

The bge-large upgrade is worth keeping. Trade-off: index size grew ~2.7×
(384D → 1024D). Query latency likely up 20-40ms (measure with
mcp__cti__ or your own timing if it matters).

Updated docs/eval-history.md with the upgrade delta."
```

**Example 3 — User reports quality issue**

User: *"Search feels worse this week. Am I imagining it?"*

Agent:
```
get_index_stats() → same as last week
evaluate_retrieval → MRR 0.71 (was 0.79 last week, ▽ -0.08 SIGNAL)
                    Recall 0.79 (was 0.83, ▽ -0.04, borderline)
                    Precision 0.42 (was 0.48, ▽ -0.06 SIGNAL)

Per-query drill: 3 of 12 queries got the WRONG top hit.

Reply: "You are NOT imagining it. Quality dropped meaningfully:
  MRR@5:       0.79 → 0.71  (▽ -0.08, real regression)
  Precision@5: 0.48 → 0.42  (▽ -0.06)

Investigation direction:
1. `git log --since='7 days ago' documents/` — anything indexed that shouldn't
   have been? Random junk can pollute retrieval.
2. Check the 3 failing queries — do they all touch a specific category?
   That suggests a category-specific issue.
3. If a new large batch was ingested with a different tone (e.g. auto-generated
   docs), consider a category filter or re-tuning min_score.

Next step: I can run per-category evaluation to isolate the regression."
```

---

## Edge cases

- **No evaluation set exists** — offer to help build one. Ask the user for 5-10 queries where they know the correct doc. Bootstrap that as `tests/evaluation-queries.json`.
- **Corpus < 50 docs** — evaluate_retrieval is not very meaningful at this scale. Numbers will jitter. Skip until corpus grows.
- **Cache hit rate always 0%** — either no repeat queries (fine, expected for one-shot use) OR `query_cache` is disabled in config. Not necessarily a problem.
- **Prometheus available** — chain to `/metrics` scrape for latency percentiles: `knowledge_rag_search_latency_seconds` histogram. Combine with retrieval quality for a full picture.

---

## Related skills

- **[`rag-onboard-context`](rag-onboard-context.md)** — the light-touch version (get_index_stats only, no evaluation).
- **[`rag-index-decisions`](rag-index-decisions.md)** — after quality tuning, index the decision so next reader knows what changed and why.
- **[`rag-check-first`](rag-check-first.md)** — the workhorse that BENEFITS from the quality tracked here.
