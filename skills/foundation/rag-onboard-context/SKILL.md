---
name: rag-onboard-context
description: At the start of every new session or when the topic shifts significantly, probe the knowledge base to learn what is indexed. Calls get_index_stats + list_categories + a couple of exploratory search_knowledge queries. Prevents the agent from operating blind or making wrong assumptions about what the corpus contains.
metadata:
  type: rag-workflow
  kind: foundation
  target: any-mcp-client
---

# rag-onboard-context — know your corpus before you use it

## When to use this skill

Run this skill:

- **At the start of any new conversation** where knowledge-rag is available and the user is about to ask substantive questions
- **When the topic shifts significantly** (from security to infrastructure, from dev to research)
- **After a major reindex** (the corpus content changed under you)
- **When the user says "I just indexed new docs"** or similar

**Do NOT run repeatedly** — once per session is usually enough. The `query_cache` keeps it cheap even if you do.

---

## What this skill commits to

Before diving into task-specific work, the agent gathers a mental map of the corpus:

- **How big** — chunk count, document count, cache health
- **What is in it** — which categories exist, roughly how many docs per category
- **What flavor** — probe 1–2 topics the user is likely to ask about, see if there are hits

That mental map informs every subsequent decision: whether to search first, whether to fall back to the web, how to phrase queries.

---

## Steps

1. **Get index health:**
   ```
   get_index_stats()
   ```
   Note: `documents_count`, `chunks_count`, `cache_hit_rate`, `embedding_model`, `embedding_dim`.

2. **Enumerate categories:**
   ```
   list_categories()
   ```
   Note: which categories exist and how many docs each has. This tells you the corpus's shape.

3. **Probe 1–2 topics the user is likely to touch.** If the user's first message mentions a domain, probe it. Otherwise, probe the top 2 largest categories:
   ```
   search_knowledge(query="<category-typical-term>", max_results=3, snippet_mode=true)
   ```

4. **Optionally, if you need concrete file names,** call:
   ```
   list_documents(category="<top-category>", limit=10)
   ```

5. **Store the summary internally** — do not necessarily surface it to the user unless they ask. The value is that YOU now know:
   - "The corpus has 15K chunks across 8 categories"
   - "Biggest categories are security (5K), development (3K), research (2K)"
   - "Search for `mitre attack` returned 3 solid hits — MITRE content is well indexed"
   - "Search for `kubernetes` returned nothing — no k8s docs indexed, defer to general knowledge"

6. **From here on, `rag-check-first` handles every subsequent request** with this context in mind.

---

## Examples

**Example 1 — Fresh session, dev context**

User: *"Hi. I want to add a new REST endpoint."*

Agent (internal, before answering):
```
get_index_stats() → 12,043 chunks / 3,214 docs, cache hit 0%, model bge-small-en-v1.5
list_categories() → {development: 1200, security: 800, general: 214, adrs: 45}
search_knowledge(query="REST endpoint", max_results=3) → 3 hits, top score 0.71

Now I know: this project has ADRs indexed. I should look at those before designing the endpoint.

Reply: "Let me check how the existing endpoints are structured..."
[calls search_knowledge with more specific queries]
```

**Example 2 — Fresh session, security context**

User: *"We have a suspected LSASS dump alert. Can you help triage?"*

Agent (internal):
```
get_index_stats() → 45,000 chunks / 5,800 docs
list_categories() → {redteam: 12000, blueteam: 8000, ctf: 5000, mitre: 4200}
search_knowledge(query="LSASS credential dump", max_results=5) → 8 hits, top score 0.89

Now I know: this is a security-focused RAG. LSASS is well-covered. Full triage is
going to be corpus-driven.

Reply: "Pulling triage steps from the runbooks and MITRE mapping..."
```

**Example 3 — Empty / near-empty corpus**

Agent (internal):
```
get_index_stats() → 12 chunks / 3 docs, cache hit 0%
list_categories() → {general: 3}

Signal: this is a fresh install. The corpus is nearly empty. Do NOT
"rag-check-first" aggressively — most queries will miss. Behave more
like a normal (RAG-optional) assistant and gently suggest the user
run `reindex_documents` after adding their docs.

Reply: "I notice your knowledge base only has 3 documents indexed. Before I dive in,
would you like to point me at your docs folder so we can index them first?"
```

---

## Edge cases

- **Very large corpus (>100K chunks)** — the probing queries stay fast (RAG is designed for this), no worry.
- **Cache warm from prior session** — `get_index_stats` returns cache_hit_rate > 0, meaning past queries are cached. That is fine and useful, no action needed.
- **User immediately asks a task-specific question** — do onboarding silently in the background and continue answering. Do not stall the user with a "let me look around first" message unless the corpus is empty.
- **Categories are empty (`{}`)** — the user has not enabled `category_mappings` in `config.yaml`. Fall back to unbounded searches; do not filter by category.

---

## Related skills

- **[`rag-check-first`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/foundation/rag-check-first/SKILL.md)** — the workhorse skill that runs on every subsequent turn, informed by what onboarding revealed.
- **[`rag-deep-dive`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/workflow/rag-deep-dive/SKILL.md)** — chained after check-first when a topic needs more depth.
- **[`rag-evaluate-quality`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/maintenance/rag-evaluate-quality/SKILL.md)** — periodic checkup (weekly, not per-session).
