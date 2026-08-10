# FTS5 Lexical Fast-Path — Migration Runbook

Operational guide for the FTS5 lexical fast-path (v4.8.2+, Task 05). Covers
enable, wait, verify, manual rebuild, and large-corpus caveats.

References: `_techspec.md` §Migration Plan · ADR-001 · ADR-008.

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

While the migration runs, lexical queries fall back to the hybrid path and
emit `fast_path_fallback_total{reason="disabled"}` plus a warning log
`FTS5 migration in progress`. The daemon continues serving queries the
entire time — the migration never blocks the request path.

Expected timings (SSD SATA):
- 3865 docs (canonical bench corpus): ~60 s
- 10 000 docs: ~2–3 min
- 100 000 docs: ~15–30 min

Progress is logged every 10 % (`[FTS5] migration progress: 30% (30/100)`)
and exposed on `/metrics`:

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

- `status: "complete"` → fast-path is live, queries dispatch to FTS5.
- `status: "in_progress"` → migration still running (or was interrupted).
  The daemon resumes from `docs_indexed` on the next restart — it never
  rebuilds from zero.
- `status: "failed"` → see `error` field for the exception class + message.
  Queries fall back permanently until you rebuild manually.

## 4. Manual rebuild

Use `scripts/build_fts5_index.py` when the marker file shows `failed`,
when you suspect index corruption, or when a maintenance window makes a
foreground rebuild convenient:

```bash
# Drop the DB + marker and rebuild synchronously.
python scripts/build_fts5_index.py --data-dir data/ --force --foreground --verbose
```

Flags:
- `--data-dir <path>` — defaults to `config.data_dir`; override for tests.
- `--force` — remove `fts5_index.db`, `fts5_index.db-wal`, `fts5_index.db-shm`,
  and `fts5_migration.state` before starting.
- `--foreground` — block until complete (default; kept for parity).
- `--verbose` / `-v` — emit a log line per 100-row batch.

The script exits `0` on success, prints an elapsed-time banner, and leaves
the marker file at `status: "complete"`.

## 5. Large corpora — dont interrupt the first rebuild

For corpora over ~10 k docs, the initial rebuild takes minutes. Best
practices:

- Kick the migration off intentionally (edit config, restart daemon)
  during a low-traffic window so the fallback logs and metric spikes are
  expected.
- Prefer `scripts/build_fts5_index.py --foreground` in ops runbooks: the
  operator sees progress synchronously and cannot accidentally reboot the
  daemon mid-rebuild.
- If the daemon is killed mid-migration, the marker file preserves the
  last checkpointed `docs_indexed`. Restart the daemon and the worker
  resumes from that batch.
- CRUD writes that happen during migration are appended incrementally via
  `add_document` (ADR-008), so ingestion is never blocked.

## Related

- `_techspec.md` §Migration Plan — full lifecycle description.
- ADR-008 — CRUD sync incremental (why FTS5 diverges from BM25 full-rebuild).
- ADR-001 — SQLite dedicated storage + WAL + `busy_timeout=5000ms`.
