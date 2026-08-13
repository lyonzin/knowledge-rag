---
name: rag-code-review
description: When performing code review on a PR, diff, snippet, or "look at this change" request, first consult the corpus for related ADRs, coding standards, prior patterns, and similar files. Grounds review comments in the team's actual decisions instead of generic best practices. Trigger on any review-style request — "review", "look at", "any issues with", "does this make sense".
metadata:
  type: rag-workflow
  kind: workflow
  target: any-mcp-client
---

# rag-code-review — review against team standards, not the internet

## When to use this skill

Trigger when the user asks:

- "Review this PR / code / diff"
- "Any issues with this?"
- "Does this look right?"
- "Is this idiomatic?"
- "Should I merge this?"
- Any critique-style prompt on a code artifact

---

## What this skill commits to

Before offering ANY review comment, the agent consults the corpus for:

1. **ADRs** governing the area (auth, retries, error handling, naming, dependencies…)
2. **Existing patterns** — similar files that show the "how we do this" convention
3. **Prior incidents** touching the code being changed

Review comments then read as *"per ADR-XXXX we do Y here"* rather than *"generally you should Z"*.

---

## Steps

1. **Identify the "area" of the change.** From the diff / snippet, extract:
   - The file path or module name (e.g. `services/payment/`, `mcp_server/security.py`)
   - The concern touched (auth, retries, logging, config, ingestion, search, storage…)
   - Any new dependencies, endpoints, or public API changes

2. **Search for governing ADRs / standards:**
   ```
   search_knowledge(query="<concern> ADR standard", max_results=5)
   ```
   Example: `search_knowledge(query="retry policy ADR")`

3. **Find similar existing files** (patterns to follow):
   ```
   search_similar(filepath="<the file being changed>", max_results=5)
   ```
   If the file is new, use `search_knowledge` with the file's purpose instead.

4. **Look for prior incidents / postmortems** on adjacent code:
   ```
   search_knowledge(query="<component> incident postmortem", max_results=3)
   ```

5. **Structure the review comments:**

   ```markdown
   ### Findings

   1. **[required]** Per ADR-0031 [docs/adr/0031-retries.md] the retry policy is
      "3 attempts, exponential backoff". This PR uses 5 attempts — either match
      the ADR or open a follow-up ADR to document the exception.

   2. **[suggestion]** The error-handling shape here differs from what
      `services/payment/refund_service.py` does [search_similar hit].
      Consider aligning for consistency.

   3. **[nit]** Missing docstring on `_new_helper()`. Team standard is ≥80%
      docstring coverage (interrogate gate, per CONTRIBUTING.md).

   ### Not commented on

   Style / naming / whitespace — those are ruff's job. Reviewed only what
   ruff cannot catch.
   ```

6. **Every finding cites a source** (ADR, similar file, incident, coding standard). If a finding is opinion-only, mark it `[opinion]` explicitly.

---

## Examples

**Example 1 — Retry policy diff**

User: *"Review this. `for attempt in range(5): try: ... except: sleep(2)` — new call site added to payment service."*

Agent:
```
search_knowledge(query="retry policy exponential backoff") → 2 hits
  → docs/adr/0031-retries.md (0.83), services/common/retry.py (0.71)

search_similar(filepath="services/payment/refund_service.py") → 5 hits
  → All use `services/common/retry.retry_with_backoff` decorator

Reply: "3 findings:

1. **[required]** ADR-0031 [docs/adr/0031-retries.md] says '3 attempts, exponential
   backoff, max 30s'. This code has 5 attempts + fixed 2s sleep — violates both.
   Fix: use `@retry_with_backoff` decorator from `services/common/retry.py:12`.

2. **[required]** Bare `except:` catches BaseException including SystemExit and
   KeyboardInterrupt. Team standard is to catch specific exceptions [PEP 8].
   Change to `except (ConnectionError, TimeoutError) as e:`.

3. **[suggestion]** Every other payment service call site uses the decorator
   pattern [services/payment/refund_service.py, capture_service.py]. This
   ad-hoc loop is the odd one out — align for consistency."
```

**Example 2 — New auth middleware**

User: *"Look at this new middleware — it adds bearer auth to a new endpoint."*

Agent:
```
search_knowledge(query="bearer auth middleware ADR") → 3 hits
  → docs/adr/0018-auth.md, mcp_server/security.py (existing BearerAuthMiddleware), CHANGELOG.md#v451

Reply: "The project already has `BearerAuthMiddleware` in `mcp_server/security.py:434`
with constant-time comparison, RFC 6750 challenge, exempt paths, etc. This new
middleware duplicates that logic — worse, uses `==` (timing side-channel).

Fix: import and reuse the existing `BearerAuthMiddleware` instead. If the new
endpoint has different auth needs (different token, different exempt paths),
compose the existing middleware or open an ADR before duplicating."
```

**Example 3 — Nothing to cite, honest**

User: *"Review this small helper for parsing dates."*

Agent:
```
search_knowledge(query="date parsing helper standard") → 0 hits
search_similar(filepath="<the new file>") → 2 weak hits (unrelated helpers)

Reply: "Nothing in the corpus about date-parsing conventions here (searched
'date parsing helper standard' and adjacent files, 0 strong hits). Reviewing
against general Python practice:

1. **[opinion]** `datetime.strptime` is slow — if this is a hot path, consider
   `dateutil.parser` or pre-compiling.
2. **[nit]** Missing timezone handling — will surprise on DST edges.

Flagging both as opinion/nit because there is no ADR to cite. If date parsing
becomes a repeated topic, worth opening an ADR."
```

---

## Edge cases

- **Very small diff** (1-2 lines) — still search once for the governing ADR of the area, then a light review. Skip similar-search.
- **Very large diff** (whole feature) — do the ADR search for the top 2-3 concerns; do not try to search every file. Focus review energy on the ADR-governed parts.
- **Newly-created file** — `search_similar` needs an existing file. Use `search_knowledge` with the file's purpose instead.
- **User pasted code without file context** — ask "what area/service is this from?" so search queries can be targeted.

---

## Related skills

- **[`rag-check-first`](rag-check-first.md)** — the base pattern (this is `check-first` specialized for review).
- **[`rag-cite-sources`](rag-cite-sources.md)** — critical here; a review comment without a citation is just opinion.
- **[`rag-deep-dive`](rag-deep-dive.md)** — for very architectural reviews, chain into deep-dive to understand the full context.
- **[`rag-index-decisions`](rag-index-decisions.md)** — if the review surfaces a new pattern, index the decision.
