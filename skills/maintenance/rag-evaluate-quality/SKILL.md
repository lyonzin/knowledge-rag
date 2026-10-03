---
name: rag-evaluate-quality
description: Measure retrieval quality using evaluate_retrieval (MRR@5 and Recall@5) and get_index_stats. Use after ingestion, a model or configuration change, or a reported search regression. Compare representative questions against a recorded baseline.
metadata:
  type: rag-workflow
  kind: maintenance
  target: any-mcp-client
---

# rag-evaluate-quality — measure, do not guess

Use this workflow when the user asks to evaluate retrieval, after a relevant change, or as part of an already authorized recurring evaluation. Do not infer that every session needs a benchmark or that invoking this skill creates a recurring schedule.

## Workflow

1. Call `get_index_stats()`. The response contains a `stats` object. Record `stats.total_documents`, `stats.total_chunks`, `stats.embedding_model`, `stats.embedding_dim`, and `stats.query_cache.hit_rate`. The cache rate describes repeated-query reuse, not relevance.

2. Reuse an independently selected evaluation set, or prepare questions with known answer documents. Each case has one non-empty question and one non-empty expected path:

   ```json
   [
     {"query": "authentication design", "expected_filepath": "docs/adr/0018-auth.md"},
     {"query": "retry policy", "expected_filepath": "docs/adr/0031-retries.md"}
   ]
   ```

   These are illustrative paths. Verify that expected documents exist in the actual corpus. Do not derive the expected answer from whichever source the current search happens to rank first.

3. Serialize the array as a JSON **string** and pass the `test_cases` parameter:

   ```python
   import json

   cases = [
       {"query": "authentication design", "expected_filepath": "docs/adr/0018-auth.md"},
       {"query": "retry policy", "expected_filepath": "docs/adr/0031-retries.md"},
   ]
   evaluate_retrieval(test_cases=json.dumps(cases))
   ```

   The MCP tool returns `mrr_at_5`, `recall_at_5`, `total_queries`, and `per_query`. It does **not** return Precision@5. Invalid cases are rejected before searches execute.

4. Inspect individual misses and rank changes before interpreting aggregates:

   | Metric | Meaning |
   |--------|---------|
   | MRR@5 | Mean reciprocal rank of the expected document; a miss contributes zero |
   | Recall@5 | Fraction of cases whose expected document occurs in the first five results |
   | `found_at_rank` | Rank for each case, or null if the expected document was not found |

   A small corpus can still be evaluated. Report the number and coverage of questions; do not infer a universal quality threshold or declare a fixed delta statistically significant. With five cases, one changed result has a large effect.

5. Compare with a previous run only after recording corpus revision, model, dimensions, query/passage prefixes, search configuration, and question set. The tool uses the server's default query settings; it does not accept `hybrid_alpha`, `search_method`, or `min_score` arguments. To compare those options, execute a separate controlled search workload with the same questions.

6. Investigate changes before recommending a rebuild:

   | Observation | Next check |
   |-------------|------------|
   | Expected document missing | Verify file discovery/exclusions, parse errors, indexed chunks, and category |
   | Rank changed after new ingestion | Inspect competing hits and per-query evidence |
   | Poor results in a non-English corpus | Evaluate an appropriate multilingual model and its required prefixes |
   | Model or passage prefix changed | Follow the full model migration procedure; incremental indexing cannot convert old vectors |
   | Cache hit rate is zero | Check whether queries actually repeat; no hit-rate target establishes retrieval quality |

7. Save an evaluation report only where the user has authorized writing. Keep evaluation questions and results outside the measured corpus unless deliberately testing their effect; indexing the answer key can contaminate later measurements.

## Example report

The following is a format example, not an observed benchmark:

```text
Corpus revision: <commit or snapshot>
Configuration: <model, dimensions, prefixes, search settings>
Cases: 12 unchanged questions; 2 Portuguese, 10 English
MRR@5: <measured value> (previous <value>)
Recall@5: <measured value> (previous <value>)
Changed cases: <query IDs and before/after ranks>
Latency: <separately measured; specify warm/cold, sample count and units>
Next action: inspect <specific missed document or configuration change>
```

Do not say that nothing broke based only on a small retrieval set. Persistence, indexing completeness, concurrent access, memory, and runtime compatibility require their own checks.

## Metrics endpoint

The current generic tool-duration metric provides `knowledge_rag_tool_duration_seconds_count` and `knowledge_rag_tool_duration_seconds_sum`; these support a mean, not a p95. The optional FTS5 latency histogram has `knowledge_rag_fast_path_latency_seconds_bucket` buckets. Do not invent a `knowledge_rag_search_latency_seconds` metric or derive percentiles from a count and sum.

## Related skills

- [rag-onboard-context](../../foundation/rag-onboard-context/SKILL.md) — inspect available corpus and configuration.
- [rag-index-decisions](../rag-index-decisions/SKILL.md) — record an approved decision separately from its evaluation dataset.
- [rag-check-first](../../foundation/rag-check-first/SKILL.md) — retrieve source material for the user's question.
