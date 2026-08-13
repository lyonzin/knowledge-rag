---
name: rag-web-fallback
description: Only reach for external web search when the local corpus comes back empty or clearly insufficient. Forces the agent to try knowledge-rag first, then explicitly document why it needed to escalate. Prevents wasted API cost, latency, and (in air-gapped deployments) accidental network calls.
metadata:
  type: rag-workflow
  kind: workflow
  target: any-mcp-client
---

# rag-web-fallback — local first, web only when necessary

## When to use this skill

Whenever the agent is tempted to call an external tool for information (`WebSearch`, `WebFetch`, `mcp__exa__*`, `mcp__context7__*`, etc.), route through this skill's checklist first.

**Applies to:**

- Any request that could be answered from indexed docs before hitting the network
- Any factual claim the agent is about to look up externally
- Any "give me the latest on X" question where X might be in the corpus

**Does NOT apply to:**

- Deliberate freshness requests: "what happened today", "current news"
- User explicitly asks for a web search
- Live infra queries (external API calls that are the actual task, not for context)

---

## What this skill commits to

The agent will:

1. Attempt at least **one** `search_knowledge` call before any web call, using appropriately extracted keywords.
2. Attempt a **second** call with alternative phrasing if the first returned nothing relevant.
3. Only then, if genuinely no local coverage, escalate to web search — with an **explicit note** to the user that RAG was checked and came back empty.

---

## Steps

1. **First-pass RAG:**
   ```
   search_knowledge(query="<extracted keywords>", max_results=5, min_score=0.15)
   ```

2. **If 0 hits (or top score < 0.3), try a paraphrase.** Common variants:
   - Swap synonyms ("auth" ↔ "authentication", "reindex" ↔ "rebuild", "ttl" ↔ "expiration")
   - Broaden ("payment refund cancellation" → "refund process")
   - Narrow ("architecture" → "authentication architecture")

3. **Second-pass RAG:**
   ```
   search_knowledge(query="<paraphrased keywords>", max_results=5, min_score=0.10)
   ```
   Note: lower `min_score` (0.10) to catch weaker matches.

4. **Decide:**

   | Local result quality | Action |
   |---|---|
   | Solid hit (score > 0.5) | Answer from RAG, do NOT hit web |
   | Weak hit (score 0.15–0.5) | Answer from RAG **and** add a web-search suggestion "for more context" |
   | No hit | **Explicitly** tell the user "corpus does not have this" then optionally invoke web search |

5. **If escalating to web,** prefix the answer with:
   > "The local knowledge base does not cover this (searched `<query1>`, `<query2>`, 0 relevant hits). Falling back to external web search."

   Then run the web tool (`WebSearch`, `mcp__exa__web_search_exa`, etc.).

---

## Examples

**Example 1 — RAG covers it, no web needed**

User: *"What is our retry policy for external HTTP calls?"*

Agent:
```
search_knowledge(query="retry policy external http") → 2 hits, top score 0.79

Reply: "Per [docs/adr/0031-retries.md], the policy is 3 attempts with exponential
backoff starting at 1s, max 30s. See services/common/retry.py:12–28 for the impl."

(No web search performed.)
```

**Example 2 — RAG empty, web fallback explicit**

User: *"What are the top CVEs in Python cryptography package in Q3 2026?"*

Agent:
```
search_knowledge(query="cve python cryptography 2026", min_score=0.10) → 0 hits
search_knowledge(query="python cryptography vulnerabilities", min_score=0.10) → 0 hits

Reply: "The local corpus does not track CVE data (searched 'cve python cryptography 2026'
and 'python cryptography vulnerabilities', 0 hits). Falling back to external web search..."

[Then calls WebSearch or mcp__cti__vuln_cve]
```

**Example 3 — Weak local hit + web supplement**

User: *"Explain the LangChain LCEL syntax."*

Agent:
```
search_knowledge(query="langchain lcel syntax") → 1 hit, score 0.22
(a passing mention in one of your ADRs comparing frameworks)

Reply: "The corpus has one passing mention of LangChain LCEL in
[docs/adr/0044-framework-choice.md]. That does not fully explain the syntax
though — for the syntax reference itself, I will pull from LangChain's public docs."

[Then calls WebFetch on the LangChain docs URL]
```

---

## Edge cases

- **Air-gapped deployment** — some enterprise setups have NO web tools enabled at all. In that case, this skill collapses to "search RAG; if empty, tell the user the corpus does not have it and stop." Do not fabricate.
- **User is impatient / one-shot query** — you can shorten the 2-pass check to 1 pass. Do not skip it entirely.
- **Freshness-critical query** ("what is the current CVE score for CVE-2024-1234") — skip RAG; the corpus is likely stale on live data. But state that you are skipping and why.
- **Related MCP tools available** — if the workspace has `mcp__cti__*`, `mcp__shodan__*`, `mcp__virustotal__*`, `mcp__context7__*`, etc., prefer those over generic web search — they are usually more precise and faster.

---

## Related skills

- **[`rag-check-first`](rag-check-first.md)** — the prerequisite (this skill is `check-first` with an explicit escalation rule).
- **[`rag-cite-sources`](rag-cite-sources.md)** — if you DO answer from RAG, cite. If from web, cite the URL.
- **[`rag-troubleshoot`](rag-troubleshoot.md)** — troubleshooting has its own escalation path (StackOverflow, GitHub issues) that follows the same pattern.
