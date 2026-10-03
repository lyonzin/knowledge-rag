# Operating Model & Single-Instance Mode

> **TL;DR**: MCP stdio clients (Claude Desktop, Claude Code, Cursor, etc.) **spawn and own the knowledge-rag server process themselves**. You do **NOT** need to run `knowledge-rag` in a terminal separately, and you **must not** run it in a terminal at the same time as your MCP client — that produces two writer processes against the same data directory, which is exactly what causes the HNSW segment corruption documented in [issue #216](https://github.com/lyonzin/knowledge-rag/issues/216). The opt-in `KNOWLEDGE_RAG_SINGLE_INSTANCE=1` flag adds a hard cap of one process per data directory. **Recommended for anyone using knowledge-rag as a writer** (any normal MCP setup with the file watcher enabled).

---

## The operating model — who runs the server

MCP servers over stdio are **1-process-per-client, spawned and killed by the client**. When you add knowledge-rag to your Claude Desktop / Cursor / etc. config, the client:

1. Reads the `command` + `args` from your MCP config on startup
2. Spawns the Python server as a child process, wires its stdin/stdout to itself
3. Holds the process alive as long as the client is running
4. Kills the child when the client closes

Everything is inside that one child process:

- MCP JSON-RPC dispatcher (the 13 tools your agent calls)
- ChromaDB client + SQLite handles
- BM25 in-memory index
- Cross-encoder reranker (lazy-loaded)
- **File watcher** — auto-reindexes when you drop / edit / delete a file in `documents/`

You do **not** need to run `knowledge-rag` in a terminal to "keep the watcher alive" or "start the service" — that model is HTTP-server thinking and doesn't apply here. Running `knowledge-rag` from a terminal launches an **additional** process that will fight the MCP-managed one for the same `data_dir`, and the ChromaDB HNSW segment writes are not safe under concurrent writers on Windows.

If you need to run `knowledge-rag` command-line for administrative work (`init`, backup, forced reindex from CLI), see [Administrative commands](#administrative-commands-manual-operations) below.

---

## When you might want the single-instance flag

MCP stdio servers are 1-process-per-client by protocol design. Some scenarios spawn more processes than you expected:

- Some MCP clients open extra internal connections during approval / review / multi-agent flows
- Long-running CI jobs accidentally launching parallel server processes
- A misbehaving wrapper script that re-launches knowledge-rag in a loop

Each `knowledge-rag` process holds its own:
- Embedding model (lazy-loaded; queries or indexing trigger loading, and memory depends on model, batch size, and runtime)
- ChromaDB client + SQLite handles
- BM25 in-memory index
- Watchdog file observer

If you've measured and confirmed you really do want a hard cap of one server per data directory, this flag is for you.

## When it is safe to leave the flag unset

Leaving the guard off requires coordinating every process accessing the data directory. `KNOWLEDGE_RAG_WATCHER_DISABLED=1` disables filesystem watching; it does not make the server read-only, disable write tools, or prevent initialization/migration writes. For multiple interactive clients, use one HTTP server process and connect clients to it.

If any process can index, receive watcher events, or call an MCP write tool (`add_document`, `add_from_url`, `update_document`, `remove_document`, `reindex_documents`), that process is a writer and **must be the only writer** against the `data_dir`. Enable the flag or manually guarantee no other process touches the directory.

> **Windows note.** A native Chroma preflight failure with code `3221225477` is an access violation, not proof of one specific cause. Preserve the database and collect diagnostics; do not delete the index automatically. Keep one writer process per data directory. A shared HTTP deployment is already supported; see below.

## Administrative commands (manual operations)

For operations that open or replace index files, such as an offline FTS5 rebuild, backup, or moving `data_dir`:

1. **Close** your MCP client (Claude Desktop / Code / Cursor). This kills the child process holding the `data_dir`.
2. Run the documented maintenance command or synchronous Python operation. There is no general `knowledge-rag reindex` CLI subcommand; see [reindex operations](reindex-operations.md).
3. Wait for the command to finish.
4. **Reopen** your MCP client. It respawns the server, watcher picks up any new files.

The lock is acquired by the server entry point. Scripts that access Chroma directly must also be coordinated; do not assume they acquire the same guard. `knowledge-rag init` exports configuration templates and presets rather than rebuilding the index.

## Activation

For stdio, set the environment variable in your MCP client config. Accepted truthy values (case-insensitive, surrounding whitespace ignored): `1`, `true`, `yes`, `on`. SSE/streamable-HTTP automatically enables the guard; clearing this variable does not disable it for those transports.

### Claude Code / Claude Desktop (`mcp.json` / `claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "knowledge-rag": {
      "command": "knowledge-rag",
      "env": {
        "KNOWLEDGE_RAG_SINGLE_INSTANCE": "1"
      }
    }
  }
}
```

### Shell

```bash
export KNOWLEDGE_RAG_SINGLE_INSTANCE=1
knowledge-rag
```

### PowerShell

```powershell
$env:KNOWLEDGE_RAG_SINGLE_INSTANCE = "1"
knowledge-rag
```

## What you'll see

A second process starting against the same data directory exits immediately with code `75` (`EX_TEMPFAIL` from `sysexits.h`) and writes to stderr:

```
[ERROR] knowledge-rag MCP server is already running (pid 12345). Refusing to start a second instance because KNOWLEDGE_RAG_SINGLE_INSTANCE is enabled.
```

## How it works

- On startup, the server opens the stable `<data_dir>/knowledge-rag.lock.guard` sidecar and acquires a nonblocking OS lock: `fcntl.flock` on Linux/macOS, or a one-byte `msvcrt.locking` range on Windows. The descriptor stays open for the entire server lifetime. Another owner is rejected before it can change PID metadata.
- While holding the OS lock, the server creates `<data_dir>/knowledge-rag.lock` with `O_CREAT | O_EXCL` and writes its PID. An existing live PID is respected for compatibility with already-running older servers. A dead or malformed PID file is recovered while the OS lock serializes competing startups.
- Normal exit removes this process's PID file and closes the guarded descriptor. **The `.guard` sidecar is never removed.** Its presence alone does not mean a server is running. Keeping the same file prevents an unlink/recreate race from producing two independent kernel locks.
- `SIGINT` / `SIGTERM` delegate to the prior Python handler or default exit behavior. Ignored signals, returning handlers, and exceptions caught by the server keep the lock until the server leaves its context. Windows never sends a termination signal merely to check or clean up a PID.
- `SIGKILL`, abrupt termination, or a hard crash releases the OS lock automatically. The next process recovers stale PID metadata before starting. PID checks use a read-only process handle on Windows; an inaccessible or inconclusive process is conservatively treated as alive.

The lock is per-data-directory. Two RAGs configured with different `data_dir` values do not collide. Stop older server versions before upgrading: an older executable does not participate in the new OS-lock protocol. Every process using a shared data directory must cooperate with the guard; it does not coordinate remote copies synchronized by a cloud-drive service.

## Troubleshooting

**"It says already running but I just killed the process"**
The OS releases ownership when the process exits, and the next startup checks the remaining PID. Verify the PID and all processes using the directory if rejection continues; permissions or PID reuse can make liveness inconclusive. Do not delete `.guard` to force startup: on Unix, replacing a file that is still locked creates a second independent lock. Remove stale PID metadata only after confirming that all owners have exited.

**"I want this off again"**
For stdio, remove the environment variable or set it to `0` / `false`. HTTP transports continue to enable the guard automatically.

**"Stale lock left after Windows shutdown"**
A hard shutdown can leave PID metadata behind; the next startup recovers it after acquiring the OS lock. The `.guard` sidecar is intentionally permanent and requires no cleanup.

## Sharing one server across clients

The current server supports `server.transport: "streamable-http"` (the `/mcp` endpoint) and the legacy SSE transport. Run one server process holding the models and index; configure HTTP-capable clients to connect to that process rather than launching another stdio server. See [configuration](CONFIGURATION.md#server) and [installation](INSTALLATION.md).

Loopback addresses are reachable only on the machine making the connection. A cloud-hosted connector cannot reach a server on your laptop through that connector's `localhost`. Exposing a server beyond the local host requires deliberate authentication, TLS, and network configuration; client connection methods differ.

## Credits

- Original guard concept and reproduction: [Sergey Khokhlov (@Hohlas)](https://github.com/Hohlas) in [PR #31](https://github.com/lyonzin/knowledge-rag/pull/31).
- Reworked as opt-in (default off), signal handlers wired, expanded test coverage: knowledge-rag maintainers.
