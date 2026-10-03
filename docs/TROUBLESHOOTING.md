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

Native ChromaDB failures can terminate Python before normal exception handling runs. Startup now probes ChromaDB in a child process before initializing the MCP server. If the probe crashes, the active `chroma_db/` and `index_metadata.json` are moved to `data/backups/auto-repair-*`, and the next startup can rebuild a clean index.

The same guarded behavior is available through either console script:

```bash
knowledge-rag
knowledge-rag-guarded
```

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

MCP stdio is one process per client by protocol — multiple Claude Code windows, Claude Desktop + IDE, etc. each spawn their own `knowledge-rag` process. Since v3.8.0 idle processes are cheap (no embedding model loaded until first query). If you've measured and want a hard cap of one server per data directory, opt in:

```bash
export KNOWLEDGE_RAG_SINGLE_INSTANCE=1
```

A second instance exits immediately with code 75. Default is OFF (multi-client friendly). Full guide: [docs/single-instance.md](single-instance.md). Sample MCP config: [examples/mcp-config-single-instance.json](../examples/mcp-config-single-instance.json).

### Second MCP client disconnects / exits with code 75

If two clients both launch `knowledge-rag` over `stdio` against the same `data_dir`, the single-instance guard (`KNOWLEDGE_RAG_SINGLE_INSTANCE=1`) can reject the second process to prevent ChromaDB index corruption (discussed in [#217](https://github.com/lyonzin/knowledge-rag/pull/217)).

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
