# Troubleshooting Guide

> Common problems and their solutions when running knowledge-rag on Windows, Linux, or macOS.

**Not finding your issue?**
- Search existing issues → https://github.com/lyonzin/knowledge-rag/issues
- Open a new bug report → https://github.com/lyonzin/knowledge-rag/issues/new/choose
- Discussions → https://github.com/lyonzin/knowledge-rag/discussions

**Related docs:**
- [Installation guide →](INSTALLATION.md)
- [Configuration reference →](CONFIGURATION.md)
- [API reference →](API.md)

---

### Python version mismatch

Requires Python 3.11 or newer.

```bash
python --version    # Must be 3.11+
```

### FastEmbed model download fails

On first use, FastEmbed downloads to the configured `paths.models_cache_dir`.
Check the download error, network access, available disk space and permissions
for that directory. Preserve working cached models: deleting the entire cache
also removes unrelated models and prevents offline startup. If a specific
download is corrupt, identify that model/revision from the error and retry only
its failed download after preserving any usable cached copy.

### Reranker model download fails

The reranker is lazy-loaded on the first query. If the model is not cached and the machine is offline, search continues without reranking and uses the RRF order from hybrid retrieval. To keep reranking enabled offline, run one query while online or pre-populate `models_cache/` on the target machine.

You can still disable reranking explicitly in `config.yaml`:

```yaml
models:
  reranker:
    enabled: false
```

Disabling reranking reduces memory use and avoids first-query model loading,
but changes result ordering. Measure precision and source recall on your own
languages and questions with it both enabled and disabled. The default reranker
was trained on English MS-MARCO; it can also move a relevant candidate below
irrelevant results. Its score is not a probability that an answer is correct.

### ChromaDB index crashes on startup

The server answers the MCP handshake before the index is ready. It checks ChromaDB in a child process so a native failure cannot take down the transport. A probe that cannot import its dependencies did not establish that the index is damaged: inspect `<data_dir>/preflight.log` and fix the installation rather than deleting the index. A busy or timed-out probe does not trigger a move either. When repeated reads confirm damage to the same index, recovery moves `chroma_db/` and `index_metadata.json` to `<data_dir>/backups/auto-repair-*` so the server can rebuild. It keeps the three newest auto-repair backups.

If the log reports `ImportError` or `ModuleNotFoundError`, make sure the launcher and its dependencies are installed in the same virtual environment. The probe runs Python in isolated mode; it does not load dependencies provided only by `PYTHONPATH` or `pip install --user`. This error does not require an index rebuild.

The same guarded behavior is available through either console script:

```bash
knowledge-rag
knowledge-rag-guarded
```

### Interrupted index repair (`.auto-repair-pending`)

Exit **65** means the repair needs inspection. After a successful move or complete rollback, preflight retries a busy marker removal after 1, 2 and 4 seconds, without moving the files again for that reason. If removal still fails, or the catalog rollback itself failed, the marker blocks the next preflight before it opens Chroma or prunes backups. Its contents name the **intended** backup for inspection; never execute them or treat them as proof of a move.

Stop **all** servers and clients using this store. Make a separate copy of the active index, catalog, marker, `preflight.log` and `<data_dir>/backups/` outside the backup rotation. Inspect the backup for **this attempt** (same timestamp and reason); do not guess from the newest folder.

| What you verify on disk | What to do |
|---|---|
| **Pair active:** `<data_dir>/index_metadata.json` and the configured `chroma_dir/chroma.sqlite3` are present; the attempt's backup contains neither moved member (the empty folder may be absent). | Verify both against the separate copy. Leave the active catalog and the entire index directory in place. |
| **Pair in backup:** `index_metadata.<reason>.json` and `chroma_db.<reason>/chroma.sqlite3` are together in the indicated backup. The live catalog is absent and the active index is absent or an empty directory. | Verify the catalog and the entire backed-up index against the separate copy. Keep them together in the backup; **do not restore just the catalog**. Restarting one server after archiving the marker lets it rebuild. |
| **Split pair:** `preflight.log` says `Catalog rollback failed`, the original index is active, the live catalog is absent, and the log identifies the matching backed-up catalog. | **Copy**, do not move, that catalog to `<data_dir>/index_metadata.json`. Verify that both catalogs have the same SHA-256 and keep the backup. Never overwrite an existing live catalog. |

For a complete pair, compare SHA-256 of the catalog and the index files at matching relative paths in the separate copy. Only after verifying the appropriate state above, release any handle on the marker and **move only `<data_dir>/.auto-repair-pending`** into the separate copy under a new name. Restart one server. If either member is ambiguous, the catalog is unexpectedly absent or the marker still cannot be moved, keep everything in place; do not bypass the stop.

### Index is empty

```bash
# Check documents directory has files
ls documents/

# Force reindex via Claude Code:
# reindex_documents(force=True)

```

For model or dimension changes, follow the isolated migration procedure in
[the reindex guide](reindex-operations.md). A dimension mismatch can prevent
startup before any MCP reindex tool becomes available.

### PDF exists but does not index

Check the actual file content as well as its extension. Binary PDFs use
PyMuPDF; encrypted, damaged or image-only documents may not yield indexable
text. OCR is not included. Explicit UTF-8 extractions beginning with `[Page N]`
are supported within the strict encoding, control-character and 4 MiB limits
described in the [format reference](../README.md). A `.pdf` suffix alone does
not make arbitrary text or binary data a valid PDF. Empty files produce no
document; inspect per-file diagnostics rather than treating a completed worker
as proof that every source was indexed.

### MCP server not loading

1. Check `~/.claude.json` exists and has valid JSON in the `mcpServers` section
2. Verify paths use double backslashes (`\\`) on Windows
3. Restart Claude Code completely
4. Run `claude mcp list` to check connection status

### "Failed to connect" error

The MCP server uses stdout for JSON-RPC communication. If a library prints to stdout during init, the stream gets corrupted. v3.4.3+ includes stdout protection that prevents this. If you're on an older version, upgrade:

```bash
pip install --upgrade knowledge-rag
```

### Slow first query

The cross-encoder reranker model is lazy-loaded on the first query. This adds a one-time ~2-3 second delay for model download and loading. Subsequent queries are fast. If the model cannot be loaded, search falls back to RRF ordering and does not retry loading the reranker until the server restarts.

### Memory usage

Memory depends on the model, chunk lengths, batch size, corpus, and runtime allocator. The embedding model and reranker load lazily when needed. Measure peak process RSS during indexing; document count alone does not predict it. Reduce `documents.embed_batch_size` to limit ONNX inference memory and use exclude patterns to limit index scope. GPU execution still consumes host RAM and adds VRAM requirements; see [GPU setup](gpu-setup.md) before changing providers.

### Multiple MCP clients spawn duplicate servers

MCP stdio is one process per client by protocol — multiple Claude Code windows, Claude Desktop + IDE, etc. each spawn their own `knowledge-rag` process. Only the first one for a `data_dir` opens the index. The others stay up, load no model and open no index, and answer every tool with `"status": "error"` and the owner's PID in `owner_pid`; the first call after the owner exits takes over. To make a second process exit instead, opt in:

```bash
export KNOWLEDGE_RAG_SINGLE_INSTANCE=1
```

A second instance then exits immediately with code 75. Full guide: [docs/single-instance.md](single-instance.md). Sample MCP config: [examples/mcp-config-single-instance.json](../examples/mcp-config-single-instance.json).

### Second MCP client refuses every tool / exits with code 75

If two clients both launch `knowledge-rag` over `stdio` against the same `data_dir`, the second process does not open the index, to prevent ChromaDB index corruption (discussed in [#217](https://github.com/lyonzin/knowledge-rag/pull/217)): it refuses every tool, or exits with code 75 when `KNOWLEDGE_RAG_SINGLE_INSTANCE=1` is set.

To share a single index across multiple clients concurrently, run one shared server instead:

```bash
knowledge-rag --transport streamable-http
```

Then connect every local HTTP-capable client (Cursor, LM Studio, etc.) to `http://127.0.0.1:8179/mcp`. For Claude custom connectors, use a public/tunneled HTTPS URL (custom connectors connect from Anthropic's cloud). See the [Multi-client setup](../README.md#multi-client-setup-shared-server-via-streamable-http) section in the README.

### SSE server won't start

```bash
# Check if port 8179 is already in use
# Windows:
netstat -aon | findstr :8179
# Linux/macOS:
lsof -i :8179
```

If `uvicorn` is not found, install the server extras: `pip install knowledge-rag[server]`

### Can't connect to SSE server

Verify the server is running and the URL is correct:

```bash
curl http://127.0.0.1:8179/sse
```

Common issues:
- Wrong URL: must end with `/sse` (not just the port)
- Firewall blocking the port
- Server started with a different host/port than configured in the MCP client

---
