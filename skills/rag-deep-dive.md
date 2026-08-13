---
name: rag-deep-dive
description: Three-step multi-tool workflow — search the corpus, fetch the most relevant document in full, then find similar documents. Use when a single search_knowledge hit is not enough because the user asked a "how does X work end to end" or "explain the pattern" or "give me the full picture" question. Prevents shallow answers on complex topics.
metadata:
  type: rag-workflow
  kind: workflow
  target: any-mcp-client
---

# rag-deep-dive — search + fetch + find similar

## When to use this skill

Trigger this skill when the user asks something that **needs breadth AND depth**:

- "How does X work end-to-end?"
- "Walk me through the authentication flow"
- "Explain the ingestion pipeline"
- "Give me the full picture on X"
- "How is X implemented across our services?"
- Any question where a 500-char snippet is obviously not enough

**Do NOT trigger for:**

- Simple factual lookups ("what port does X use") — `rag-check-first` alone is enough
- User explicitly wants a short answer ("TL;DR", "one-liner")
- Time-sensitive triage where speed matters more than depth

---

## What this skill commits to

The agent runs a **3-tool chain** in a fixed order:

1. `search_knowledge` — find candidates
2. `get_document` — read the top match in full
3. `search_similar` — find related material

Then synthesizes an answer that pulls from all three, cites each source, and flags gaps.

---

## Steps

1. **Search — cast a wide net:**
   ```
   search_knowledge(query="<user's topic>", max_results=8, snippet_mode=true, min_score=0.15)
   ```
   Wider than usual (8 not 5) because we want candidate diversity for the similar-search step.

2. **Fetch — read the winner in full:**
   ```
   get_document(filepath="<top_hit.source>")
   ```
   The full document, not just the chunk. This gives you sections that adjacent chunks did not surface.

3. **Find similar — discover the surrounding graph:**
   ```
   search_similar(filepath="<top_hit.source>", max_results=5)
   ```
   These are documents ChromaDB considers semantically close to the top hit. Often surfaces the "obvious next document" that keyword search missed.

4. **Cross-reference the 3 result sets.** Look for:
   - **Consensus** — do multiple docs describe the same thing? Cite them together.
   - **Contradiction** — flag it explicitly to the user, do not silently pick one.
   - **Gap** — if similar-search surfaces a doc you did not read yet, decide if it is worth a second `get_document`.

5. **Structure the answer**:
   - **Overview** — 3-4 sentence summary drawn from the full document.
   - **Details** — sections/subsections quoted or paraphrased with citations.
   - **Related** — 2-3 bullets pointing at the similar documents with 1-line why-it-matters each.
   - **Gaps** — anything the corpus does NOT cover that the user might expect. Be honest.

---

## Examples

**Example 1 — "How does the ingestion pipeline work?"**

Agent (internal):
```
1. search_knowledge(query="document ingestion pipeline chunks embedding", max_results=8)
   → 6 hits: ingestion.md (0.88), parsers.md (0.72), chunker.md (0.69), embeddings.md (0.65),
             fastembed.md (0.61), watchdog.md (0.55)

2. get_document(filepath="docs/ingestion.md")
   → 4200-char full doc: 5 sections (parsing / chunking / embedding / storage / watcher)

3. search_similar(filepath="docs/ingestion.md", max_results=5)
   → 5 hits: ADR-0007-parser-choice.md, storage-format.md, chromadb-tuning.md,
             benchmark-ingestion.md, migration-v4.md

4. Synthesized reply:
   "The ingestion pipeline has 5 stages [docs/ingestion.md]:
    1. Parsing — dispatcher maps extension to parser [ingestion.py:174]
    2. Chunking — markdown-aware or code-aware per format [chunker.md]
    3. Embedding — FastEmbed ONNX lazy-loaded [embeddings.md]
    4. Deduplication — SHA256 [ingestion.md#dedup]
    5. Storage — ChromaDB with WAL mode [chromadb-tuning.md]

    The parser choice per format is documented in ADR-0007 [docs/adr/0007-parser-choice.md].
    Recent migration notes: [docs/migration-v4.md].

    Not covered by the corpus: the exact chunk-size tuning for large PDFs — you may want
    to open an ADR on this if it comes up often."
```

**Example 2 — "Walk me through OAuth2 flow in our services"**

Agent (internal):
```
1. search_knowledge(query="oauth2 flow authorization refresh", max_results=8)
   → 4 hits: adr/0018-auth.md, oauth-runbook.md, auth-service.md, token-storage.md

2. get_document(filepath="docs/adr/0018-auth.md")
   → Full ADR with 3 diagrams + decision + consequences

3. search_similar(filepath="docs/adr/0018-auth.md", max_results=5)
   → adr/0019-mtls.md, session-management.md, refresh-token-rotation.md,
      audit-logging.md, revocation.md

4. Reply weaves them together with explicit citations, calls out that
   adr/0019-mtls.md is about the S2S path (adjacent decision), and flags
   that "token revocation on user logout" is only mentioned in passing —
   worth clarifying with the security team.
```

---

## Edge cases

- **Top hit is very short** (<500 chars) — the full document may not be much larger than the chunk. Still call `get_document` for completeness; then either lean on similar-search or run a 2nd search with different keywords.
- **`search_similar` returns the same file** — expected for near-unique docs; move on.
- **Time budget matters** — this is 3 tool calls and reasoning between them. For quick chat, skip to `rag-check-first`.
- **Overlapping content across many docs** — dedupe by summarizing the common thread once and citing all the sources at the end of that summary paragraph.

---

## Related skills

- **[`rag-check-first`](rag-check-first.md)** — the prerequisite (deep-dive is `check-first` + 2 more tools).
- **[`rag-cite-sources`](rag-cite-sources.md)** — even more important on deep-dive because you are quoting many sources.
- **[`rag-web-fallback`](rag-web-fallback.md)** — if the corpus does not cover the topic in depth, the deep-dive itself will surface that gap and you can chain to web search.
