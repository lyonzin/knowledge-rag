# Reindex Operations Guide (v4.8.0+)

## Sync vs. Async APIs

The MCP tool `reindex_documents(force=True, full_rebuild=True)` returns immediately after spawning a daemon thread that performs the rebuild in the background. This is intentional for the MCP protocol (the client shouldn't block for hours waiting on a large reindex), BUT it has a subtle failure mode:

**If the Python process running the MCP server dies mid-rebuild** (crash, OOM kill, `Ctrl+C`, remote SSH disconnect, laptop suspend, etc.), the daemon thread dies with it — leaving the ChromaDB collection **deleted but never repopulated**. Result: RAG returns zero results until the operator manually re-runs the reindex.

**Rule of thumb:**

- **MCP tool calls:** use `reindex_documents(...)` as documented — the async wrapper is correct. The whole point of MCP is that clients (Claude, Cursor, etc.) get a response inside the request budget.
- **CLI scripts / one-off jobs:** prefer `get_orchestrator().nuclear_rebuild()` (sync) directly. This guarantees the rebuild completes before Python exits. If the script gets killed, at least the collection is not in a half-destroyed state.
- **Long-running rebuilds on unstable connections:** use `reindex_documents(force=True)` (smart reindex, not full rebuild) and rely on the resume checkpoint (see below) to recover.

## Resume Kwarg (v4.8.0+)

`reindex_documents(resume=True)` recovers from `data/reindex_checkpoint.json` after an interrupted smart reindex.

**Rules:**

- `resume=True` is only valid when `full_rebuild=False`. The combination `full_rebuild=True + resume=True` is rejected up front with a structured error — nuclear rebuild has no meaningful checkpoint semantic because the entire collection is thrown away and rebuilt from scratch, so "resuming" would leave a partial collection with no coherent state.
- `resume=True` implicitly forces `mode='smart_reindex'`, even when `force=False` was passed. An interrupted smart reindex must be resumed with smart, not silently downgraded to `incremental` (which would ignore the checkpoint entirely).
- When no valid checkpoint exists, `resume=True` prints an informational log and starts a fresh smart reindex.

**Checkpoint cadence:** written every 500 docs OR every 30 seconds, whichever comes first.

- 500-doc cadence covers fast runs (small markdown docs mostly cached by mtime skip, thousands per minute).
- 30-second cadence covers slow runs (one PDF with 5000 chunks could take longer than 30s, leaving too long a gap between checkpoints).

**Checkpoint invalidation:** the checkpoint stores a `config_signature` — an SHA256 hash of the current `embedding_model | embedding_dim | chunk_size | chunk_overlap` tuple. If any of those config values change between checkpoint write and resume load, the checkpoint is discarded with a WARN and the reindex starts fresh. This prevents a mixed collection (partial old vectors with the previous model + partial new vectors with the current one) which would silently degrade retrieval quality.

**Checkpoint lifecycle:**

- Written every 500 docs / 30s during a smart reindex, alongside a metadata flush so `_indexed_docs` on disk stays in sync with the checkpoint's `indexed_doc_ids` list.
- Cleared automatically on successful reindex completion.
- Cleared automatically if `_load_checkpoint()` returns None on a `resume=True` attempt (missing/corrupt/version-mismatch/signature-mismatch).

## Progress Fields (v4.8.0+)

`get_reindex_status()` returns these fields while a reindex is active:

| Field                 | Type    | Meaning                                                                                       |
| --------------------- | ------- | --------------------------------------------------------------------------------------------- |
| `chunks_processed`    | int     | Chunks committed to ChromaDB so far                                                           |
| `chunks_total`        | int     | Rolling estimate (0 during warmup, then running average from completed docs × total_files)    |
| `throughput_cps`      | float   | Chunks per second, sliding window (last 30s OR 100 samples, whichever smaller — for stability) |
| `eta_seconds`         | int     | Estimated seconds to completion, derived from throughput + remaining chunks                   |
| `checkpoint_saved_at` | str/None | ISO timestamp of the last checkpoint write (None until first checkpoint)                     |
| `resumed`             | bool    | True when this run recovered from a checkpoint via `resume=True`                              |

**Warmup behavior:** `chunks_total` starts at 0 (or an estimate from previously indexed docs if any exist) and is refined each iteration using the running average across docs already processed. `throughput_cps` stays at 0.0 until at least 2 samples land in the sliding window. `eta_seconds` stays at 0 during throughput warmup and near completion (when `chunks_processed` catches up to the estimate).

**Sliding window rationale:** using a fixed-size deque bounded at 100 entries AND pruning entries older than 30 seconds keeps the throughput number stable during transient stalls (e.g. one giant PDF that takes 60s to embed) without diluting current speed with ancient samples from the start of the run.

## Example Recovery Flow

```
# Terminal 1: start a smart reindex (say, 5000 docs)
mcp> reindex_documents(force=True)
{"status": "started", "operation": "smart_reindex", ...}

# ... 3 minutes in, laptop crashes at doc 2100 / 5000 ...

# Terminal 1 restart: check if a checkpoint survived
$ ls data/reindex_checkpoint.json
data/reindex_checkpoint.json

# Resume from where it stopped
mcp> reindex_documents(resume=True)
{"status": "started", "operation": "smart_reindex", ...}
[REINDEX] Resuming smart reindex from checkpoint (2100 docs already processed, 12345 chunks)

# Poll status — resumed=True marker + chunks_processed continues from checkpoint
mcp> get_reindex_status()
{
  "active": true,
  "operation": "smart_reindex",
  "progress": "2103/5000",
  "chunks_processed": 12360,
  "chunks_total": 29400,
  "throughput_cps": 45.2,
  "eta_seconds": 377,
  "resumed": true,
  ...
}
```

## Escalation Rules of Thumb

- Reindex started, hours later still `active: false`? Check `last_error` in status. If Python was killed, the collection may be half-populated — run `reindex_documents(resume=True)` if the operation was smart, or `reindex_documents(force=True, full_rebuild=True)` if it was nuclear.
- Getting `resume=True is only valid for smart reindex` error? Drop `full_rebuild=True`. Nuclear rebuild does not support resume by design (see above).
- Checkpoint keeps invalidating with "config_signature mismatch"? Something in `config.yaml` (or the effective merged config) changed embedding model / dim / chunk size / chunk overlap between runs. Either revert the config or accept that the next reindex will start fresh.
