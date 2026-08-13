---
name: rag-check-first
description: Before answering any technical question, code request, architecture decision, or factual claim, call search_knowledge to check the local corpus. Trigger on any query that could be answered with prior work, indexed docs, ADRs, runbooks, or team context. Prevents hallucination and forces reliance on the indexed knowledge base.
metadata:
  type: rag-workflow
  kind: foundation
  target: any-mcp-client
---

# rag-check-first — search before you speak

## When to use this skill

Trigger this skill **before answering** whenever the user asks:

- A technical "how" or "why" question (design, implementation, security, ops)
- Something about "our" / "the project" / "the team" / a named component
- A request to write, refactor, review, or debug code
- A question that could plausibly be answered by an ADR, runbook, README, or spec

**Trigger keywords / patterns (non-exhaustive):**

- "how does X work", "why did we", "what is the pattern for"
- "add a", "implement", "refactor", "fix", "debug"
- Any mention of a file, module, function, feature, or component by name
- Any question about historical decisions ("we used to", "the old way")

**Do NOT trigger for:**

- Pure conversation / meta requests ("hi", "thanks", "what's your name")
- Requests that are explicitly about the AI itself
- Well-known facts already in training data with no team-specific angle

---

## What this skill commits to

**Before drafting a single line of the answer**, the agent will call `search_knowledge` at least once with a query derived from the user's message. The results shape the answer — training-data knowledge is only a fallback when RAG returns nothing relevant.

---

## Steps

1. **Extract the search query** from the user's message. Prefer 2–5 keywords, not full sentences. Include domain-specific identifiers (class names, error codes, feature names, ADR IDs).

2. **Call `search_knowledge`:**
   ```
   search_knowledge(query="<extracted keywords>", max_results=5, snippet_mode=true, min_score=0.15)
   ```
   - `snippet_mode=true` keeps token consumption sane
   - `min_score=0.15` cuts obvious noise (adjust higher for terse corpora)

3. **Read the top 3 results.** Pay attention to `search_method` (`hybrid` > `semantic` / `keyword` alone), `score`, and `reranker_score`.

4. **Decide the answer strategy** based on what came back:

   | RAG result | What to do |
   |---|---|
   | ≥1 result with `score > 0.5` matching the intent | Answer **primarily from the RAG content**, quote relevant snippets, cite the source (`path:line`) |
   | 1–3 results with lower scores (0.2–0.5) | Use as **partial context**, be explicit that the corpus does not cover this in depth, mix with training-data knowledge |
   | 0 results after 2 attempts with varied phrasing | State it: "Local corpus does not have this. Answering from general knowledge — verify before acting." Optionally chain to `rag-web-fallback` |

5. **If unsure between 2 corpus interpretations**, call `get_document` on the most promising `source` to fetch full context before answering.

---

## Examples

**Example 1 — Design question**

User: *"How do we handle authentication in the payment service?"*

Agent (internal):
```
search_knowledge(query="authentication payment service", max_results=5, min_score=0.15)
→ 3 hits: ADR-0018-auth.md (0.82), payment-service-design.md (0.71), auth-runbook.md (0.55)

Answer: "Per ADR-0018 (path: docs/adr/0018-auth.md), the payment service
uses mTLS between services and OAuth2 for user-facing endpoints..."
```

**Example 2 — Implementation request**

User: *"Add a new endpoint for refund cancellation."*

Agent (internal):
```
search_knowledge(query="refund cancellation endpoint", max_results=5)
→ 2 hits: refund-service.md (0.68), refunds-api-v2.md (0.61)

Then: read the top result, model the new endpoint on the existing pattern,
cite the reference file in the reply.
```

**Example 3 — Empty corpus**

User: *"How do we deploy to Kubernetes?"*

Agent (internal):
```
search_knowledge(query="kubernetes deploy", max_results=5)
→ 0 results

Reply: "Nothing in the local corpus about Kubernetes deployment. Answering
from general practices — please verify against your actual cluster setup
before running: [general k8s deployment steps]"
```

---

## Edge cases

- **Very short queries** (1 word) — expand to 2–3 keywords before calling `search_knowledge`. Single-word queries have poor recall.
- **Ambiguous acronyms** — if "MFA" could mean multi-factor auth or mean-field approximation, either ask the user or run 2 targeted searches.
- **Non-English corpus** — the default embedding model is English. If the corpus is multilingual, either switch to the `multilingual` embedding profile OR translate the query to English before searching.
- **Repeated searches on the same session** — the query cache handles this (LRU + 5-min TTL), no need to skip.

---

## Related skills

- **[`rag-cite-sources`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/foundation/rag-cite-sources/SKILL.md)** — the natural follow-up: every claim carries a `path:line` citation.
- **[`rag-deep-dive`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/workflow/rag-deep-dive/SKILL.md)** — if the top result needs full context, chain into `get_document` + `search_similar`.
- **[`rag-web-fallback`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/workflow/rag-web-fallback/SKILL.md)** — the escape hatch when the corpus is empty.
- **[`rag-onboard-context`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/foundation/rag-onboard-context/SKILL.md)** — call once at session start, then `rag-check-first` handles every subsequent request.
