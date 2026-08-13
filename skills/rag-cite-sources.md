---
name: rag-cite-sources
description: Every technical claim drawn from the local corpus must ship with a source citation formatted as path:line or path:section. Trigger whenever the response quotes, paraphrases, or acts on knowledge that came from a search_knowledge or get_document call. Makes answers auditable and lets the user jump to source in one click.
metadata:
  type: rag-workflow
  kind: foundation
  target: any-mcp-client
---

# rag-cite-sources — every claim traces to a source

## When to use this skill

Whenever an answer contains **any of these**:

- A specific number, threshold, config value, or version taken from the corpus
- A statement about how the system currently works, was designed, or was decided
- A code snippet, function signature, or API contract quoted from indexed files
- A "the runbook says", "the ADR says", "the RFC decided", "we agreed" claim
- A recommendation grounded in prior team practice

Basically: **if a curious reader would want to double-check where you got that**, cite.

---

## What this skill commits to

Every technical claim in the response carries a citation of the form:

- `[path/to/file.md:42]` — line-anchored (from `search_knowledge` result's `source` field)
- `[path/to/file.md#section-heading]` — section-anchored for Markdown
- `[path/to/file.md]` — file-only, when line/section is not available

Citations are **inline** at the point of the claim, not batched at the end. One claim = one citation.

---

## Steps

1. **When you call `search_knowledge` or `get_document`**, capture:
   - The `source` field (usually the file path)
   - The `filename` (short name for display)
   - Any `chunk_id` or line hint in the metadata
   - The actual `content` you are about to quote or paraphrase

2. **When you draft the response**, mark every fact-carrying sentence with its source. Preferred inline syntax:

   ```markdown
   Per ADR-0018 [docs/adr/0018-auth.md], the payment service uses mTLS between services.
   The retry policy is 3 attempts with exponential backoff [services/payment/retry.py].
   ```

3. **If quoting verbatim**, use fenced blockquote + citation:

   ```markdown
   > "All external API calls must timeout after 30 seconds."
   > — [docs/architecture/http-contracts.md#timeouts]
   ```

4. **If synthesizing across multiple sources**, cite each:

   ```markdown
   The service uses OAuth2 [docs/adr/0018-auth.md] with a 15-minute access token TTL
   [config/auth.yaml:12] and refresh-token rotation on every use [docs/security/oauth-flow.md].
   ```

5. **When a claim is NOT from the corpus** (general knowledge, external doc, your own reasoning), mark it explicitly:

   ```markdown
   Per [docs/adr/0018-auth.md] we use OAuth2. Note that OAuth2 spec (RFC 6749, external)
   also allows client-credentials flow, which the ADR does not mention.
   ```

---

## Examples

**Example 1 — Answering a config question**

User: *"What is our default request timeout?"*

Agent:
```
search_knowledge(query="default request timeout") →
1 hit: config/http.yaml (line 8, score 0.87)

Reply: "The default request timeout is 30 seconds [config/http.yaml:8].
This is enforced globally by the HTTP client middleware [services/common/http_client.py:45]."
```

**Example 2 — Code review**

User: *"Is this retry logic correct?"*

Agent:
```
search_knowledge(query="retry policy exponential backoff") →
2 hits: docs/adr/0031-retries.md (0.79), services/common/retry.py (0.65)

Reply: "Per our retry ADR [docs/adr/0031-retries.md], the policy is
'3 attempts, exponential backoff starting at 1s, max 30s'. Your code
retries 5 times which exceeds the standard. Adjust to match
services/common/retry.py:12–28 or open a new ADR to document
the exception."
```

**Example 3 — Explicit no-source note**

User: *"Should we use gRPC instead of REST for the new service?"*

Agent:
```
search_knowledge(query="grpc rest choice") → 0 hits

Reply: "The corpus does not have a decision on gRPC vs REST for new services
(no ADR indexed on this). Answering from general considerations [not from corpus]:
gRPC gives you… [reasoning]. Recommend: open an ADR with the tradeoffs before
committing."
```

---

## Edge cases

- **Chunk mid-sentence** — if `search_knowledge` returns a chunk that starts mid-paragraph, call `get_document(filepath=<source>)` to fetch context before quoting.
- **Very long paths** — abbreviate for readability, keep the full path in a footnote:
  ```markdown
  Per the retry ADR [ADR-0031¹], we use exponential backoff.
  ¹ docs/architecture/decisions/0031-retries-and-timeouts.md
  ```
- **Non-file sources** (URL-ingested content) — cite the original URL from the document's metadata:
  ```markdown
  Per RFC 9110 §15.5.9 [https://datatracker.ietf.org/doc/html/rfc9110, indexed
  via add_from_url on 2026-05-14], 425 Too Early is idempotent-safe.
  ```
- **The corpus contradicts itself** — cite both sources and flag the conflict explicitly:
  ```markdown
  Conflicting sources: [docs/adr/0018-auth.md] says 15-min TTL, but
  [config/auth.yaml:12] shows 30-min. Recommend resolving the ADR vs config
  drift before answering with certainty.
  ```

---

## Related skills

- **[`rag-check-first`](rag-check-first.md)** — the prerequisite: search happens first, citations follow.
- **[`rag-deep-dive`](rag-deep-dive.md)** — chained citations across a full drill-down.
- **[`rag-code-review`](rag-code-review.md)** — code review is where citation discipline pays off most.
