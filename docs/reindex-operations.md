# Reindex operations guide

Use the mode that matches the required work. The MCP tool returns immediately;
the worker runs in the server process and stops if that process exits.

| MCP call | Work performed | Recovery |
| --- | --- | --- |
| `reindex_documents()` | Scan paths, skip unchanged files, index changes and prune deleted sources | Repeat the incremental scan |
| `reindex_documents(force=True)` | Reprocess and re-embed every discovered file, including unchanged files | Smart-reindex checkpoint |
| `reindex_documents(resume=True)` | Resume a compatible smart checkpoint; otherwise start a fresh smart run | Smart-reindex checkpoint |
| `reindex_documents(full_rebuild=True)` | Populate and validate a staging collection, then replace the live collection | Preserve/inspect the previous collection and retry; no smart checkpoint |

`resume=True` and `full_rebuild=True` are incompatible. The wrapper rejects
that combination before starting a worker.

## Observe completion

After a call returns `status: "started"`, poll `get_reindex_status()`.
When `reindex.active` becomes false, inspect both `last_error` and
`last_result.errors`. A stopped worker or an HTTP/MCP success response does not
prove all documents were indexed.

Inspect representative sources and retrieval results after a large operation.
A parser error should retain the previous indexed content for that source and
be reported as an error; it must not be treated as a file deletion.

For an operator-owned foreground script, `get_orchestrator().nuclear_rebuild()`
waits for completion in that process:

```python
from mcp_server.server import get_orchestrator

result = get_orchestrator().nuclear_rebuild()
print(result)
if result.get("errors") or result.get("error"):
    raise SystemExit(1)
```

Configure the intended data directory before importing the server. A standalone
script must not run alongside another writer using the same data directory.
Foreground execution prevents normal script exit from abandoning a daemon
worker; it does not protect against termination, power loss or storage failure.

## Smart-reindex checkpoints

Checkpoints are stored as `reindex_checkpoint.json` in the configured data
directory. `resume=True` selects smart reindex even if `force` was omitted.
Missing, malformed, incompatible or mismatched checkpoints are ignored and a
fresh smart run begins.

The checkpoint records completed document IDs, a chunk counter, timestamps,
operation, schema version and a configuration signature. The signature covers
the embedding model, dimensions, query and passage prefixes, chunk size and
overlap. Changing any of them invalidates the checkpoint. This validation is not
a migration mechanism for existing vectors.

Checkpoint cadence is checked **between documents**: every 500 processed
documents or after 30 seconds since the last save. One large document can take
longer than 30 seconds, so this is not a strict wall-clock persistence guarantee.
Metadata is saved alongside checkpoint progress, with each JSON file written
through a temporary sibling and replacement. The pair of files is not one
cross-file database transaction.

A successful smart run clears the checkpoint. A run with partial indexing errors
retains progress so completed work need not be repeated while failed documents
are retried. Keep the source corpus and effective configuration stable during
recovery; validate changed sources afterwards.

Example MCP sequence:

```python
reindex_documents(force=True)
# Poll get_reindex_status() while active.

# If the process was interrupted, restart the same configuration, then:
reindex_documents(resume=True)
# Poll again and inspect last_error and last_result.errors.
```

## Progress fields

| Field inside `reindex` | Meaning |
| --- | --- |
| `operation` | Incremental, smart reindex or nuclear rebuild |
| `progress`, `percent` | File-level progress |
| `indexed`, `skipped`, `errors` | Document processing counters |
| `chunks_processed` | Chunks committed by the running operation |
| `chunks_total` | Rolling estimate, not a final count |
| `throughput_cps` | Recent chunk throughput |
| `eta_seconds` | Estimate based on remaining estimated chunks and throughput |
| `checkpoint_saved_at` | Last checkpoint timestamp, or null |
| `resumed` | Whether a compatible checkpoint was loaded |

During warmup, throughput and ETA may be zero. Estimates change as documents with
different sizes are processed. The throughput window retains at most 100 samples
and drops samples older than 30 seconds; it does not measure GPU utilization.

## Staging rebuild and rollback

`nuclear_rebuild(swap=True)` is the default underlying full-rebuild operation:

1. Clean up sufficiently old staging/backup collections from earlier operations.
2. Preserve the current live state and create a separate staging collection.
3. Populate staging using isolated document maps and BM25 state. The production
   collection remains available for queries.
4. Validate staging counts and backend queries. With a nonempty baseline, the
   existing count gate requires at least 90% of the old chunk count, and at least
   four of five sanity queries must return a result.
5. Rename the production collection to a backup name and staging to the
   production name. Publish the replacement state and persist metadata.
6. Delete the old collection only after successful publication. Refresh derived
   FTS5 state and invalidate query caches.

The count and sanity-query gates catch gross failures. They are not a retrieval
quality evaluation, and semantic queries returning any nearest neighbor do not
prove the expected source was found. Use an independent set of expected-source
queries for release validation.

The renames and metadata publication are separate operations, **not one atomic
ChromaDB transaction**. Handled failures restore the previous state while its
vectors are still available. An abrupt crash between operations can leave
staging or backup collections requiring inspection.

| Failure point | Expected handling |
| --- | --- |
| Parsing/population or validation | Abort publication and retain the previous live state |
| First rename | Keep the original collection |
| Promotion rename | Attempt to restore the production name; report rollback failures |
| State publication/metadata persistence | Restore the previous collection and maps while its backup still exists |
| Cleanup after successful publication | Report/retain a leftover backup for later cleanup |
| Process termination | Inspect persisted names and metadata before retrying; Python rollback did not run |

Do not delete the data directory to resolve a generic opening error. Collection
initialization now preserves stored data and reports the failure so permissions,
model mismatches and storage errors can be investigated.

## Storage and legacy mode

Budget space for the old and new vector collections during staging, plus
metadata, FTS5 and database journal/WAL overhead. Two similarly sized collections
can approximately double their vector storage, but the complete directory size
does not have a fixed multiplier.

Staging/backup cleanup uses a 24-hour age threshold and runs at startup and before
rebuilds. It is not a continuously scheduled timer: an orphan may remain longer
than one day if no cleanup opportunity occurs.

The Python API retains `nuclear_rebuild(swap=False)` as a legacy destructive
mode. It deletes the active collection before indexing and can leave an empty
or partial index if interrupted. It is not the default MCP rebuild path.
Use a verified backup and an exclusive maintenance window if this legacy mode
is deliberately chosen.

## Changing the embedding model

The embedding model, vector dimensions and passage prefix define the stored
vector space. Incremental indexing cannot safely mix profiles. Even models
with the same dimensions can produce incompatible vectors.

A model change can prevent the old collection from opening before an MCP tool is
available. A safe migration therefore uses a separate data directory:

1. Preserve the old configuration and index. Create a migration configuration
   using the new model, dimensions and required query/passage prefixes.
2. Keep `paths.documents_dir` pointing to the existing source corpus. Set
   `paths.data_dir` to a new empty directory; ChromaDB, document metadata,
   checkpoints and FTS5 must all be isolated. Merely changing
   `search.collection_name` is not sufficient isolation.
3. Start the migration instance with its own working/configuration directory.
   Disable its watcher while validating, and avoid document mutation tools
   against the shared source corpus.
4. Index the corpus, inspect errors and verify counts, expected sources and
   retrieval quality. Confirm the actual embedding provider and memory use.
5. Point clients to the validated replacement configuration. Retain the old
   index until the migration is accepted; reverting clients then provides a
   straightforward rollback.

See [configuration](CONFIGURATION.md) for path resolution and
[GPU setup](gpu-setup.md) for provider-specific requirements. The multilingual
preset also requires its matching query and passage prefixes.

## FTS5 recovery is separate

FTS5 is derived from ChromaDB. Its migration marker and chunk counters are
separate from the smart-reindex checkpoint. See
[the FTS5 migration runbook](runbooks/fts5_migration.md) before rebuilding or
removing that derived database.
