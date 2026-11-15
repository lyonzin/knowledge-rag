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
`last_result.errors`. `last_result.failed_files` lists which files failed and
why (deduped by path, at most 100 entries; `failed_files_truncated: true` means
more failed). A failed `full_rebuild=True` run reports the first failing paths
in `last_error` and keeps the partial `last_result` with the full list. A
stopped worker or an HTTP/MCP success response does not prove all documents
were indexed.

Inspect representative sources and retrieval results after a large operation.
A parser error should retain the previous indexed content for that source and
be reported as an error; it must not be treated as a file deletion.

Storage consistency and retrieval quality are separate checks. Before a large
validation run, choose questions, expected sources and required evidence from
the original documents. Keep that set unchanged when comparing configurations,
and retain failed-to-ingest sources in the end-to-end denominator. A matching
filename does not prove that the returned chunks contain the answer. When a
source is relevant but its retrieved context is incomplete, use `get_document`
to inspect that source. Record nearest-neighbor results for unanswerable queries
separately; a nonempty search result is not itself an answered question.

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

## Recovery after a failed document rollback

A mutation error can be followed by a second error while restoring its vectors.
The original error remains visible. The server preserves the last committed
document inventory and records the affected source in `document_repairs.json`,
next to `index_metadata.json` (normally in `data_dir`). This is an exceptional
recovery record, separate from the smart-reindex checkpoint.

Run `reindex_documents()` and inspect completion and errors. The next incremental
scan reprocesses marked sources even when their timestamps are unchanged or a
checkpoint lists them as completed. It also removes recorded chunks from failed
new additions whose uncommitted source file no longer exists. Healthy sources
keep their ordinary unchanged-file fast path. A record is removed only after
the corrected vectors and document metadata have committed. FTS5 remains
unavailable while repair is pending; treat retrieval as degraded until the
repair run has completed and representative sources have been checked.

The record validates the collection, canonical source containment and document
version IDs. It is limited to 1 MiB, 1,024 affected sources and 16 version IDs per
source; version IDs are not chunk IDs, so a document may have many more chunks.
A malformed, oversized or mismatched record causes an explicit startup/reindex
error and is preserved for inspection. Do not discard the record to hide that
error: rebuild into isolated data and metadata paths if recovery cannot proceed
against the existing state.

If saving the record itself fails, the exception explains that recovery is not
durable. `reindex_documents(force=True)` can use the in-memory repair state in
the current process. After a restart without a saved record, a protected full
rebuild may be needed to remove unregistered orphan chunks. Its 90% count gate
can reject a legitimately reduced corpus; in that case rebuild into a separate
data directory and retain the old index for inspection, rather than weakening
the gate or deleting stored data. Use the path-isolation procedure below and
keep the current embedding profile when recovering without a model change.

A successful staging rebuild clears validated repair records only after
publication and metadata persistence. A later record-cleanup failure is logged
as a post-commit cleanup problem; it does not mean the published index was
rolled back. These records support handled recovery, not an atomic transaction
across source files, Chroma, FTS5 and metadata or a guarantee against power loss.

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
# Poll again and inspect last_error, last_result.errors and last_result.failed_files.
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

## Repair of unreachable chunks (GH #313)

A query for a chunk's own stored vector does not always return the chunk: the
HNSW graph can leave a vector that no search path leads to, and then no semantic
search finds that chunk. `self_recall` in `get_index_stats()` reports the
symptom. How many chunks end up like that depends on the order in which the
documents are written. A pass of the server writes them in the order of its
folder scan, and on a 70,752-chunk index of a real corpus that left 4.4% of the
chunks unreachable; the same vectors written in random order left 0.59%.

Every operation that writes chunks ends by checking the chunks it wrote: a
reindex pass, a watcher pass, the staging pass of a full rebuild (before
validation and swap), `add_document`, `update_document` and `add_from_url`. A
chunk counts as reachable when a query by its own stored vector returns its id
among the 10 nearest, or returns a first result at distance zero: another id
then holds the same vector, and the `self_recall` probe counts that the same
way. An unreachable chunk is updated in place with the vector it already holds,
which makes ChromaDB compute its links in the graph again. The chunk never
leaves the collection, nothing is embedded again, and BM25 and FTS5 are left as
they are: ids and texts do not change.

Linking a chunk again can cost other chunks their links, so every chunk of the
operation is checked again after a round of updates, up to three checks and two
rounds per operation. Unreachable chunks at each check, on collections built
straight on ChromaDB 1.5.9 from the vectors of that index (default HNSW
parameters, one upsert per document) and checked by another process:

| Documents written | Check 1 | Check 2 | Check 3 | A new process afterwards |
| --- | --- | --- | --- | --- |
| In the order of the server's folder scan | 3,109 (4.39%) | 272 (0.38%) | 63 (0.09%) | 108 (0.15%) |
| In random order | 418 (0.59%) | 88 (0.12%) | 34 (0.05%) | 51 (0.07%) |

A pass of the server itself over that corpus found 3,303, 311 and 73
unreachable chunks at its three checks. Two runs through the MCP tools, an
initial pass and a full rebuild each, reported 3,015 to 3,960 chunks repaired
and 55 to 72 remaining per pass, and a new process then found 109 and 146
unreachable chunks (0.15% and 0.21%).

One check of all 70,752 chunks took 11 to 13 seconds, about 0.17 ms per chunk
and check; the updates took under a second. The counts move from one build of
the same vectors to the next: five builds in random order ended their third
check between 34 and 88 chunks. The last column is the same check from a new
process: ChromaDB saves the graph after every 1,000 records written and applies
the later ones again when it opens the collection, so a restart does not load
exactly the graph the running process had.

The checks run after the last document of a pass, with the index lock held:
writes wait until they end, searches do not, and `get_reindex_status()` can
show every file as processed while the pass is still active. An operation that
checks more than 500 chunks writes an `[INDEX] Checking ...` line to stderr
before the first check.

`last_result` of `get_reindex_status()` reports `unreachable_chunks_repaired`
and `unreachable_chunks_remaining`. The second is what the last check still
found, or `null` when no check completed: the switch is off, the index has
fewer than 1,000 chunks, the operation wrote nothing, or a check could not run.
An operation whose checks found any and completed writes one
`[INDEX] Repaired ...` line to stderr. A check or an update that cannot run is
one `[WARN]` line on stderr, which names the error; it does not fail the
operation, which keeps every chunk it wrote.

- Only the chunks an operation wrote are checked. Chunks indexed by an earlier
  version, or by a pass that was interrupted, are checked when they are written
  again: `reindex_documents(force=True)` or `reindex_documents(full_rebuild=True)`.
- An update can cost a chunk the operation did not write its links, and such a
  chunk is not checked. `self_recall` in `get_index_stats()` measures the index
  as a whole.
- An index with fewer than 1,000 chunks is not checked.
- `search.repair_unreachable_chunks: false` switches the check off.

## Changing `search.ef_search` requires a rebuild (GH #287)

`search.ef_search` sets the HNSW search-time exploration depth when a collection
is **created**. ChromaDB does not apply configuration changes to an existing
index: `collection.modify(configuration=...)` updates the stored metadata but
leaves search behavior unchanged — verified empirically on chromadb 1.5.9, even
after reopening the `PersistentClient`. So changing the value in `config.yaml`
alone has no effect on a live index.

To apply a new value, rebuild:

```python
reindex_documents(full_rebuild=True)   # nuclear rebuild: staging collection is created with the new ef_search
```

The staging rebuild path creates a fresh collection that picks up the new
`ef_search`. A plain `reindex_documents(force=True)` re-embeds in place and is
**not** enough — it reuses the existing collection. After the rebuild, verify
with `get_index_stats()`: `self_recall.top1` should increase (the probe samples
300 chunks, so expect ~±0.7 percentage points of sampling error at the 95%
level — compare before/after, not against a hard 1.0), and `hnsw_ef_search`
should now equal the configured value. On the real 70k-chunk index the default
100 showed a top-1 miss rate of ~1.4%; higher values reduce it at a roughly
linear latency cost.

## Storage and legacy mode

Budget space for the old and new vector collections during staging, plus
metadata, FTS5 and database journal/WAL overhead. Two similarly sized collections
can approximately double their vector storage, but the complete directory size
does not have a fixed multiplier.

Staging/backup cleanup uses a 24-hour age threshold and runs at startup and before
rebuilds. It is not a continuously scheduled timer: an orphan may remain longer
than one day if no cleanup opportunity occurs.

A retired collection also needs a persisted retirement marker linked to the
UUID of its successfully committed replacement. Expiration is allowed only
while that exact replacement remains the primary collection. An empty primary
created on a later startup does not authorize deleting an older backup.
Unmarked legacy backups and ambiguous recovery copies remain preserved rather
than expiring automatically; inspect their data and metadata before deciding
which copy to restore or remove.

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
