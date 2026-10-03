# FTS5 Lexical Fast-Path — Migration Runbook

Operational guide for the FTS5 lexical fast-path (v4.8.2+, Task 05). Covers
enable, wait, verify, manual rebuild, and large-corpus caveats.

References: [feature guide](../features/fts5_fast_path.md), [reindex operations](../reindex-operations.md), and the implementation in `mcp_server/fts5_index.py`.

## 1. Enable the feature

Edit `config.yaml`:

```yaml
search:
  lexical_fast_path:
    enabled: true
```

Restart the daemon. On the next `KnowledgeOrchestrator` init the migration
detects a missing (or non-`complete`) marker file and dispatches a background
thread that populates `<data_dir>/fts5_index.db` from ChromaDB.

## 2. Wait for the migration to finish

While the migration runs, `search_method="auto"` falls back to the hybrid
path and increments `knowledge_rag_fast_path_fallback_total{reason="disabled"}`.
Explicit `search_method="fts5"` requests report that the index is not ready.
The migration shares disk and CPU resources with retrieval; measure latency
on the actual workload rather than assuming it has no effect on requests.

Progress counts refer to **chunks**, despite the historical `docs_*` field
names. A single source file can contribute many chunks. Duration depends on
chunk count, text size, disk, and concurrent ingestion.

Progress is logged every 10 % (`[FTS5] migration progress: 30% (30/100)`)
and exposed on `/metrics` when the separate metrics server is enabled:

- `knowledge_rag_fast_path_migration_docs_indexed` gauge
- `knowledge_rag_fast_path_migration_docs_total` gauge

## 3. Verify the marker file

`<data_dir>/fts5_migration.state` — canonical JSON schema:

```json
{
  "status": "complete",
  "docs_total": 3865,
  "docs_indexed": 3865,
  "started_at": "2026-08-07T12:00:00+00:00",
  "completed_at": "2026-08-07T12:01:02+00:00",
  "error": null
}
```

- `status: "complete"` → the last migration reported completion. Confirm row
  consistency and a known identifier query; a marker alone is not an integrity check.
- `status: "in_progress"` → migration still running (or was interrupted).
  A valid checkpoint can resume on restart. Invalid state or a changed corpus
  can require replaying the source rows. Chunk IDs make replay idempotent.
- `status: "failed"` → see `error` field for the exception class + message.
  Automatic queries fall back while the index is unavailable. Restart can
  retry the migration; investigate persistent errors before rebuilding manually.

## 4. Manual rebuild

Use `scripts/build_fts5_index.py` when the marker file shows `failed`,
when you suspect index corruption, or when a maintenance window makes a
foreground rebuild convenient. Stop the knowledge-rag process using that
data directory first, so another connection is not writing the database or
marker while `--force` replaces them:

```bash
# Drop the DB + marker and rebuild synchronously.
python scripts/build_fts5_index.py --data-dir data/ --force --foreground --verbose
```

Flags:
- `--data-dir <path>` — directory containing `chroma_db/`; defaults to
  `config.data_dir`. The command opens an existing Chroma database and the
  configured `search.collection_name`; it does not create a replacement corpus.
- `--force` — remove `fts5_index.db`, `fts5_index.db-wal`, `fts5_index.db-shm`,
  and `fts5_migration.state` before starting.
- `--foreground` — block until complete (default; kept for parity).
- `--verbose` / `-v` — emit a log line per 100-row batch.

The script exits `0` on success and leaves the marker at `status: "complete"`.
Missing databases, failed batches, and incomplete migrations are failures.
`--force` deletes only the derived FTS5 files listed above; it does not
re-embed documents or remove Chroma's source vectors.

## 5. Large corpora and recovery

For large corpora:

- Kick the migration off intentionally (edit config, restart daemon)
  during a low-traffic window so the fallback logs and metric spikes are
  expected.
- Prefer `scripts/build_fts5_index.py --foreground` in ops runbooks: the
  operator sees progress synchronously. Killing that process still interrupts
  the operation; a foreground call is not a durability guarantee.
- If the daemon is killed mid-migration, the marker file preserves the
  last checkpointed `docs_indexed`. Restart the daemon and the worker
  resumes from that batch.
- Normal indexing and CRUD synchronize exact stored chunk IDs. If a sync
  fails, inspect the error diagnostic and rebuild the derived index from
  Chroma after resolving the cause.

## Related

- [FTS5 feature guide](../features/fts5_fast_path.md)
- [API reference](../API.md)
- [Reindex operations and checkpoints](../reindex-operations.md)
