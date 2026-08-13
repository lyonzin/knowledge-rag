---
name: rag-index-decisions
description: After making a non-obvious architectural decision, solving a novel bug, agreeing on a coding standard, or reaching a conclusion worth remembering, index it back into the knowledge base so the next occurrence is one search away. Uses add_document or add_from_url. Closes the feedback loop that makes a RAG-backed team compound over time.
metadata:
  type: rag-workflow
  kind: maintenance
  target: any-mcp-client
---

# rag-index-decisions — close the feedback loop

## When to use this skill

Trigger this skill when, during a session, the team (or the agent + user together) produces:

- **A design decision** with tradeoffs discussed (worth an ADR)
- **A novel bug fix** whose root cause is non-obvious
- **A convention agreed on** ("from now on we do X for Y")
- **A postmortem summary** — even a paragraph
- **A URL / doc / paper** that shaped the decision (worth ingesting via `add_from_url`)

**Do NOT trigger for:**

- Trivial fixes (typo, formatting, obvious one-liner)
- Session-only context that will not matter next time
- Highly sensitive material that should NOT be in the RAG (secrets, PII, WIP negotiations)
- Duplicates of things already indexed

---

## What this skill commits to

When the session produces something worth remembering, the agent proactively suggests indexing it — either:

- As a **new file** written to `documents/` and indexed via `add_document`, OR
- As a **URL ingestion** via `add_from_url` if the reference lives elsewhere

The agent does NOT silently index things — always asks the user first, offering a preview of the content and where it will go.

---

## Steps

1. **Recognize the moment.** The session hit one of the triggers above.

2. **Draft the artifact** in your head. Typical shapes:

   - **ADR (Architecture Decision Record):** ~200-500 words, structured as *Context / Decision / Consequences*.
   - **Runbook update:** step-by-step reproducible playbook.
   - **Postmortem:** what happened / root cause / fix / prevention.
   - **Coding standard:** "for X, do Y (not Z), because W."

3. **Propose to the user:**

   > "This decision looks worth remembering. I can:
   >
   > **(a)** Create `docs/adr/0045-token-storage.md` with a 300-word ADR summarizing what we agreed, then index it.
   > **(b)** Just record the summary here, no indexing.
   >
   > Which?"

4. **If the user picks indexing, write the file** to an appropriate path under `documents/` (or wherever the config's `documents_dir` points), then:

   ```
   add_document(filepath="documents/adr/0045-token-storage.md",
                category="adrs")
   ```

5. **For external references** (a URL that shaped the decision):

   ```
   add_from_url(url="https://datatracker.ietf.org/doc/html/rfc9110",
                category="external-refs")
   ```

   The `add_from_url` tool fetches, sanitizes, wraps in a provenance fence
   (defends against prompt injection per OWASP LLM01:2025), then indexes.

6. **Confirm success** by immediately searching for it:

   ```
   search_knowledge(query="<decision keywords>", max_results=3)
   ```

   Verify the new content shows up. If not, `reindex_documents(force=false)`
   forces a smart reindex.

7. **Cross-link** in the new document — reference related ADRs, runbooks,
   incident reports. This builds the semantic graph that makes
   `search_similar` powerful later.

---

## Examples

**Example 1 — New ADR from a design discussion**

User: *"OK, we decided: rotate refresh tokens on every use, 30-day max lifetime, revoke all tokens on password change."*

Agent:
```
"That is a clean architectural decision worth an ADR. I can draft:

  docs/adr/0045-refresh-token-rotation.md
  ---
  # ADR-0045: Refresh Token Rotation on Every Use
  
  ## Context
  Refresh tokens were fixed-lifetime, revealed once per session. Attacker
  who intercepts a token gets 30 days of access.
  
  ## Decision
  - Rotate refresh token on every use (issue new, invalidate old)
  - Max lifetime: 30 days
  - Password change revokes all tokens for the account
  
  ## Consequences
  + Compromised token has ≤ 1 request lifespan
  + Detection: reuse of a rotated token = clear signal of theft
  - Complexity: need atomic rotate+revoke, race conditions
  - Storage: revoked-token set grows (mitigate with TTL)
  
  ## Related
  - ADR-0018 (base auth design)
  - RFC 6819 §5.2.2.3 (token rotation guidance)
  ---

Then: add_document(filepath='docs/adr/0045-refresh-token-rotation.md', category='adrs')

OK to proceed?"

User: "Yes, do it."

Agent:
[Writes file, calls add_document, then confirms with a search]
search_knowledge(query='refresh token rotation') → new ADR appears at score 0.92

"Done. ADR-0045 is indexed and searchable. Next time anyone asks about refresh
tokens, rag-check-first will surface it immediately."
```

**Example 2 — Postmortem from a bug fix**

User: *"OK that was painful. FTS5 index went silent after nuclear rebuild because the marker file had stale 'complete' but the actual FTS5 table was empty."*

Agent:
```
"Worth a short postmortem so this bites nobody again. Draft:

  documents/incidents/2026-08-15-fts5-silent-after-rebuild.md
  ---
  # Incident: FTS5 Fast-Path Silent After Nuclear Rebuild
  
  Date: 2026-08-15
  Component: mcp_server/fts5_index.py + server.py
  Severity: silent — no error surfaced, queries returned empty
  
  ## What happened
  ...
  
  ## Root cause
  ...
  
  ## Fix
  ...
  
  ## Detection for next time
  Search for 'FTS5 silent' or 'fast-path empty' in RAG.
  ---

add_document then search to verify. Proceed?"
```

**Example 3 — External URL worth ingesting**

User: *"This paper on multi-vector retrieval changed my mind on our reranker choice."*

Agent:
```
"I can ingest it directly:

  add_from_url(url='https://arxiv.org/abs/2402.15059', category='research')

The URL ingest will:
1. Fetch the HTML / PDF
2. Strip navigation, keep content
3. Wrap in a provenance fence (marks external, defends against prompt injection)
4. Index like any other doc

Proceed?"
```

---

## Edge cases

- **User is not sure what category** — check `list_categories()` output and suggest the closest one, or offer to create a new category.
- **Secret material** — if the user pastes credentials, tokens, or PII, do NOT index it. Say: "I noticed this contains what looks like a credential — indexing would put it in the corpus permanently. Redact before indexing?"
- **Duplicate detection** — before writing, `search_knowledge(query='<title of the new doc>')` to check if something very similar already exists. If yes, offer to `update_document` instead of adding a new one.
- **Very sensitive project** — some orgs forbid AI writing to the corpus. Respect the setting; offer just to draft the file for the human to commit.

---

## Related skills

- **[`rag-troubleshoot`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/workflow/rag-troubleshoot/SKILL.md)** — the natural upstream: after a novel bug fix, index the postmortem.
- **[`rag-code-review`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/workflow/rag-code-review/SKILL.md)** — the other upstream: after a review surfaces a new pattern, index it.
- **[`rag-onboard-context`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/foundation/rag-onboard-context/SKILL.md)** — the next session's onboarding will surface the new index; this closes the loop.
- **[`rag-evaluate-quality`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/maintenance/rag-evaluate-quality/SKILL.md)** — after significant indexing activity, worth measuring quality delta.
