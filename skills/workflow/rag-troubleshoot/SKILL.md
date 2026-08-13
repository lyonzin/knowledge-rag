---
name: rag-troubleshoot
description: When the user reports a bug, error message, stack trace, unexpected behavior, or "why is this broken" question, search the corpus first for prior occurrences, known fixes, or related runbooks. Prevents re-solving problems the team already solved. Trigger on any error signature, exception name, stack trace snippet, or "why does X fail" query.
metadata:
  type: rag-workflow
  kind: workflow
  target: any-mcp-client
---

# rag-troubleshoot — RAG-first debugging

## When to use this skill

Trigger the moment the user reports:

- A specific error message, stack trace, or exception
- "It broke", "it fails", "not working", "returns null when it shouldn't"
- Unexpected behavior in a component
- Alert / incident triage
- "Why does X do Y" where Y is wrong
- A pasted log line
- A failed CI/CD run

The core insight: **many bugs are already solved somewhere in your corpus** — runbook, postmortem, incident report, prior fix commit, ADR, chat thread indexed via `add_from_url`. Search first.

---

## What this skill commits to

Before proposing a fix, the agent searches for:

1. The **error signature itself** — exception name, first line of stack, unique error code
2. The **affected component / function** — module name, service name, feature
3. **Prior incidents** with similar symptoms — even if the error message differs

Only after those three come back empty does the agent apply general debugging techniques.

---

## Steps

1. **Extract error signatures** from the user's message:
   - Exception class name (`ValueError`, `ConnectionError`, `TimeoutError`, etc.)
   - Error code (`ERR_INVALID_TOKEN`, `E42_INDEX_MISS`, HTTP status)
   - First distinctive line of the stack trace
   - Unique keywords ("segfault at ...", "cannot connect to ...")

2. **First search — exact error signature:**
   ```
   search_knowledge(query="<exception name> <error keyword>", max_results=5, min_score=0.15)
   ```
   Example: `search_knowledge(query="ConnectionError refused chromadb")`

3. **Second search — affected component + symptom:**
   ```
   search_knowledge(query="<component name> <what happens>", max_results=5)
   ```
   Example: `search_knowledge(query="reindex fails halfway large corpus")`

4. **Third search — historical incidents / postmortems:**
   ```
   search_knowledge(query="incident postmortem <component>", max_results=5)
   ```
   Or with a category filter if your corpus categorizes by document type:
   ```
   search_knowledge(query="<component> failure", category="incidents")
   ```

5. **Evaluate:**

   | Result | Action |
   |---|---|
   | Prior fix exists | Apply it, cite the source, verify it still applies |
   | Related runbook exists | Follow the runbook, cite it |
   | Related incident, different fix | Read the incident, adapt |
   | Nothing found | Do general-purpose debugging, but flag that "this may be a new failure mode" |

6. **After solving**, if the fix is non-obvious, chain to [`rag-index-decisions`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/maintenance/rag-index-decisions/SKILL.md) to add a short note back to the corpus so the next occurrence is one search away.

---

## Examples

**Example 1 — Prior fix**

User: *"Getting `EmbeddingModelLoadError: sticky failure` when starting knowledge-rag. What's happening?"*

Agent:
```
search_knowledge(query="EmbeddingModelLoadError sticky failure") → 3 hits, top 0.91

Reply: "This is documented in [CHANGELOG.md#v381]. The `sticky _load_failed` flag was
added in v3.8.1 as a hotfix — when the ONNX model fails to load, subsequent calls
re-raise immediately instead of looping through HF Hub download attempts. The fix is
in [providers/embeddings/fastembed.py:145]. Root cause is usually one of:
  1. HF Hub unreachable at first startup (offline / DNS)
  2. Corrupt cached model in models_cache/
  3. Disk full during model download

Try: (a) check network to huggingface.co, (b) rm -rf models_cache/ and restart,
(c) df -h. If none apply, this may be a new failure mode."
```

**Example 2 — Runbook**

User: *"Our search results are empty after nuclear rebuild. What now?"*

Agent:
```
search_knowledge(query="empty results nuclear rebuild") → 4 hits
1st hit: docs/reindex-operations.md (0.87)

get_document(filepath="docs/reindex-operations.md")
→ Full runbook including "post-rebuild sanity check" section

Reply: "Per [docs/reindex-operations.md#post-rebuild-sanity-check], after a nuclear
rebuild you should:
1. Confirm `get_index_stats().chunks_count` > 0
2. Run one `search_knowledge('any indexed keyword')` — should return hits
3. If empty, check for stale FTS5 marker (v4.8.2+ specific) — delete
   `data/fts5_migration.state` and restart.
The most common cause is exactly the FTS5 stale marker case, added in v4.8.3
[CHANGELOG.md#v483]."
```

**Example 3 — Nothing found, general debug**

User: *"My Rust binding for the MCP server segfaults on second call. Ideas?"*

Agent:
```
search_knowledge(query="rust binding segfault MCP") → 0 hits
search_knowledge(query="rust FFI second call crash") → 0 hits
search_knowledge(query="rust bindings segfault") → 0 hits

Reply: "Nothing in the local corpus about Rust bindings (searched 'rust binding
segfault MCP' + 2 paraphrases, 0 hits). This appears to be a new failure mode.
General-purpose debugging directions:
  1. Rebuild with `RUST_BACKTRACE=full` to get the panic location
  2. Check FFI lifetime — is a Python object dropped while Rust still holds a ref?
  3. Try with `--single-instance` mode to rule out concurrency
  4. Reduce to a minimal repro: 1 call, 2 calls, does it always fail on N=2?

Once you find the root cause, worth indexing back — see rag-index-decisions."
```

---

## Edge cases

- **Very generic error** ("KeyError") — 1st search will have low precision. Add the component name early or skip the exact-error search and start with component + symptom.
- **Multi-line stack traces** — extract the deepest custom frame (not stdlib) as the search seed.
- **Error is intermittent / rare** — still search, but weight the "prior incidents" query more heavily.
- **User pasted only the symptom, not the error** — ask for the error message + a stack trace before searching. Guessing wastes search calls.

---

## Related skills

- **[`rag-check-first`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/foundation/rag-check-first/SKILL.md)** — the parent skill (troubleshooting is a specialized variant).
- **[`rag-cite-sources`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/foundation/rag-cite-sources/SKILL.md)** — when you propose a fix, cite the source that documented it.
- **[`rag-index-decisions`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/maintenance/rag-index-decisions/SKILL.md)** — after solving a novel bug, index the postmortem for next time.
- **[`rag-web-fallback`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/workflow/rag-web-fallback/SKILL.md)** — for truly novel errors, escalate to GitHub / StackOverflow after RAG comes back empty.
