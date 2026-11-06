---
name: rag-cite-sources
description: Every technical claim drawn from the local corpus must ship with a source citation — the file path from the result's source field, plus a section, or a line once it was checked in get_document. Trigger whenever the response quotes, paraphrases, or acts on knowledge that came from a search_knowledge or get_document call. Makes answers auditable and lets the user jump to the source.
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

- `[path/to/file.md]` — the default: the result's `source` path, shortened to the part inside the documents folder
- `[path/to/file.md#section-heading]` — section-anchored, for Markdown, when the heading above the quoted text shows in the chunk or in `get_document`
- `[path/to/file.md:42]` — line-anchored, only after you found the quoted text in `get_document`'s content and counted the lines

No tool returns line numbers. A `search_knowledge` result carries `source`, `chunk_index` (which chunk of the file matched) and, with `snippet_mode=false`, `match_offset` (a character offset into the returned `content`). `get_document` returns the parsed text: Markdown loses its YAML front matter, JSON comes back re-indented, and PDF, Office, notebook and CSV files come back as extracted text. A line counted there matches the file only for Markdown without front matter and for code, YAML, XML and plain text; otherwise cite the section.

Citations are **inline** at the point of the claim, not batched at the end. One claim = one citation.

---

## Steps

1. **When you call `search_knowledge` or `get_document`**, capture:
   - The `source` field (the file path)
   - The `filename` (short name for display)
   - `chunk_index` and, when present, `match_offset` — they locate the match, but they are not line numbers
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
   [config/auth.yaml] and refresh-token rotation on every use [docs/security/oauth-flow.md].
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
1 hit: config/http.yaml, whose snippet shows `timeout: 30s`
get_document(filepath="config/http.yaml") → the setting is on line 8

Reply: "The default request timeout is 30 seconds [config/http.yaml:8].
This is enforced globally by the HTTP client middleware [services/common/http_client.py]."
```

**Example 2 — Code review**

User: *"Is this retry logic correct?"*

Agent:
```
search_knowledge(query="retry policy exponential backoff") →
2 hits: docs/adr/0031-retries.md, services/common/retry.py

Reply: "Per our retry ADR [docs/adr/0031-retries.md], the policy is
'3 attempts, exponential backoff starting at 1s, max 30s'. Your code
retries 5 times which exceeds the standard. Adjust to match
`retry_with_backoff` in [services/common/retry.py] or open a new ADR
to document the exception."
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
  [config/auth.yaml] shows 30-min. Recommend resolving the ADR vs config
  drift before answering with certainty.
  ```

---

## Related skills

- **[`rag-check-first`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/foundation/rag-check-first/SKILL.md)** — the prerequisite: search happens first, citations follow.
- **[`rag-deep-dive`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/workflow/rag-deep-dive/SKILL.md)** — chained citations across a full drill-down.
- **[`rag-code-review`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/workflow/rag-code-review/SKILL.md)** — code review is where citation discipline pays off most.
