# Operating Model & Single-Instance Mode

> **TL;DR**: MCP stdio clients (Claude Desktop, Claude Code, Cursor, etc.) **spawn and own the knowledge-rag server process themselves**; you do not need to run `knowledge-rag` in a terminal. Only one process per data directory owns the index: the first one claims it and keeps it until that process exits. Every other process started for the same data directory (a second client window, a terminal) stays up, never opens the index and answers all 13 tools with a refusal that names the owner; its first call after the owner exits takes over. To use one index from several clients at the same time, run one HTTP server and connect every client to it. `KNOWLEDGE_RAG_SINGLE_INSTANCE=1` makes a second process exit with code 75 instead of waiting.

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

You do **not** need to run `knowledge-rag` in a terminal to "keep the watcher alive" or "start the service" — that model is HTTP-server thinking and doesn't apply here. A `knowledge-rag` started from a terminal while a client runs is a second process for the same `data_dir`: it does not open the index and serves no tool (see below).

If you need to run `knowledge-rag` command-line for administrative work (`init`, backup, forced reindex from CLI), see [Administrative commands](#administrative-commands-manual-operations) below.

---

## One owner per data directory

Two processes writing to one ChromaDB directory, and even a process reading it while another writes, left HNSW segments unloadable ([issue #216](https://github.com/lyonzin/knowledge-rag/issues/216)). So exactly one process opens the index:

- The first knowledge-rag process for a `data_dir` takes a nonblocking OS lock on `<data_dir>/knowledge-rag.lock.guard` (`fcntl.flock` on Linux/macOS, a one-byte `msvcrt.locking` range on Windows) and holds it until the process exits. It is the **owner**.
- The owner answers the MCP `initialize` handshake at once and checks, opens and prepares the index on a background thread. Until the index is open and prepared, embedding-model migration included, every tool answers `{"status": "repairing", "message": "...", "retry_after_seconds": 5}`. If that startup fails, every tool answers `{"status": "error", "message": "..."}` with the reason, until the client restarts the server.
- Any other process for the same `data_dir` is a **follower**. It never opens the index and answers every one of the 13 tools with:

```json
{
  "status": "error",
  "message": "Another knowledge-rag process (pid 12345) owns this data directory, so this session does not open the index and serves no tool. It takes over on the first call after that process exits.",
  "owner_pid": 12345,
  "owner_running": true,
  "hint": "To use one index from several clients at once, run one shared server (knowledge-rag --transport streamable-http) and connect every client to it; see docs/single-instance.md."
}
```

- `owner_pid` is diagnostic. If the OS lock is busy but the PID cannot be read, the follower still refuses the call (`owner_pid: null`, `owner_running: false`). A missing PID is **not** permission to open the index or remove a lock.
- The first tool call a follower receives after the owner's process exited takes over: that call answers `repairing`, and the follower runs the owner's startup. With several followers, the OS lock picks one; the refusals of the others then name the new owner when its PID is readable.

## The startup check and the access lock

The startup check (preflight) reads the index in a child process so a native crash cannot kill the server. The child uses the parent's installed package, resolved index directory and collection, even if the working directory contains an older checkout or the config file changes before a follower takes over. Only repeated evidence of damage while reading that same index permits moving it to `data/backups/auto-repair-*`. A probe that cannot import or start is a startup error, not a reason to move data. The child is a reader and must not overlap with the owner's own use of the index, even if the owner exits, the child starts late or it runs past its timeout:

- The child takes a second OS lock, `<data_dir>/knowledge-rag.lock.access`, before it touches the index, and gives it up only by exiting.
- The owner takes that lock after its checks, before it moves, repairs or opens the index, and keeps it until it exits. A check still inside, such as one left by a session that ended abruptly, makes the owner wait; tools answer `repairing` meanwhile.
- A check that finds the lock taken exits without touching the index. That is contention, not damage, and it validates nothing: the owner runs the check again every 5 s, and moves or opens the index only after a check got inside. A check that times out is still inconclusive and keeps the index, as before ([#216](https://github.com/lyonzin/knowledge-rag/issues/216)).
- If no check gets inside within 120 s, the owner neither opens nor moves the index, and every tool answers an `error` that says so. End the process that stays inside (a Python process started with `-c` that imports `mcp_server`), then restart the client.
- The probe and owner compare the filesystem identities of the index directory and its SQLite database. If either is replaced, or the filesystem cannot identify it reliably, the owner refuses to move it. Do not point `chroma_db` or its SQLite file through a symlink, junction or hard-linked database file: this could put the data and its guard in different places. To relocate storage, set `paths.data_dir` for the **whole** directory, including its locks.
- If the index is damaged and another program holds one of its files (an antivirus scan, a backup or sync client, a knowledge-rag older than this lock), moving it aside fails. The owner retries after 1, 2 and 4 s, then leaves the index untouched and answers `error` until the client restarts it. Details go to `<data_dir>/preflight.log`.

## When to set `KNOWLEDGE_RAG_SINGLE_INSTANCE`

Without the variable, a second stdio process stays up as a follower, which suits clients that start several processes. With it, a second process exits with code `75` (`EX_TEMPFAIL` from `sysexits.h`) instead, which suits wrapper scripts or CI jobs that should fail fast. HTTP transports always set it: an HTTP server never follows, so a second one cannot bind the same data directory.

Accepted truthy values (case-insensitive, surrounding whitespace ignored): `1`, `true`, `yes`, `on`. Clearing the variable does not change SSE/streamable-HTTP, which set it themselves.

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

With the variable set, a second process writes to stderr and exits with code 75:

```
[ERROR] knowledge-rag MCP server is already running (pid 12345). Refusing to start a second instance because KNOWLEDGE_RAG_SINGLE_INSTANCE is enabled (HTTP transports enable it). Close the other knowledge-rag sessions on this data directory, stdio clients included, and retry.
```

## HTTP health probes

An HTTP server listens as soon as it starts, while the index is still being checked, so liveness and readiness are separate URLs on the MCP port:

| URL | Use it for | Answers |
| --- | --- | --- |
| `/health`, `/healthz` | Liveness: is the process up | Always `200`, with `status`, `version`, `uptime_seconds`, `index` (`ready`, `repairing` or `error`) and query-cache statistics, the latter only from a ready index |
| `/readyz` | Readiness: can it serve tools | `200 {"status":"ready"}`; `503 {"status":"repairing"}` with `Retry-After: 5` while the index is checked, opened or repaired; `503 {"status":"error"}` after a startup failure, without `Retry-After` |

None of them needs a bearer token, and none returns paths, exception messages, tokens or PIDs. A liveness probe that restarted the server on `repairing` would cut a long repair short, which is why `/health` stays `200`.

**Migration.** Before this version, the HTTP server started listening only after the index was open, so a `200` from `/health` also meant ready. If you used the status code of `/health` as a readiness check, switch it to `/readyz` and keep `/health` or `/healthz` for liveness:

```yaml
livenessProbe:
  httpGet: { path: /healthz, port: 8179 }
readinessProbe:
  httpGet: { path: /readyz, port: 8179 }
```

## How the lock works

- The owner's PID stays on line 1 of `<data_dir>/knowledge-rag.lock`, for diagnostics and for older versions, which read only that line; line 2 is `guarded`. The OS lock decides who owns the directory, never the PID.
- While holding the OS lock, a new owner replaces a PID file marked `guarded`: its writer held the same lock and no longer does, even if its PID now belongs to a live process (a reused number, or the same number after a container restart). A PID file without the mark comes from an older version that takes no OS lock; if its PID is alive, the new process respects it and follows (or exits 75 with the variable set). A dead or malformed PID is replaced.
- Normal exit removes this process's PID file; the OS releases both locks when the process ends, however it ends. **The `.guard` and `.access` sidecars are never removed.** Their presence alone does not mean a server is running, and replacing a file that is still locked would create a second, independent lock on Unix.
- `SIGINT` / `SIGTERM` delegate to the prior Python handler or default exit behavior. Windows never sends a termination signal merely to check a PID; PID checks use a read-only process handle, and an inaccessible or inconclusive process counts as alive.

The lock is per data directory: two RAGs configured with different `data_dir` values do not collide. It does not coordinate copies synchronized by a cloud-drive service, and startup refuses a `data_dir` under one (exit code 78) unless `paths.allow_cloud_sync: true` is set.

## Upgrading from an older version

Stop **every** `knowledge-rag` process using the data directory before upgrading from v5.1.0. Its default stdio launcher takes no writer guard, and even with `KNOWLEDGE_RAG_SINGLE_INSTANCE=1` it does not take the new index-access lock. Closing just one client is not enough when another old process still has the index open.

An older version killed abruptly with the variable set can leave a PID file without the `guarded` mark. If that PID now belongs to an unrelated live process, new sessions keep following it. After checking that no knowledge-rag process runs, delete `<data_dir>/knowledge-rag.lock`, never the `.guard` or `.access` sidecar.

## Administrative commands (manual operations)

For operations that open or replace index files, such as an offline FTS5 rebuild, backup, or moving `data_dir`:

1. **Close** your MCP client (Claude Desktop / Code / Cursor). This kills the child process holding the `data_dir`.
2. Run the documented maintenance command or synchronous Python operation. There is no general `knowledge-rag reindex` CLI subcommand; see [reindex operations](reindex-operations.md).
3. Wait for the command to finish.
4. **Reopen** your MCP client. It respawns the server, watcher picks up any new files.

The server entry point acquires the lock. A maintenance script that already holds `single_instance_lock()` can call `run_preflight()` on the same thread **before it opens the index**; the check leaves the enclosing lock in place. It cannot run against an index the process is already serving or take ownership away from another process. Scripts that open Chroma directly still need coordination. `knowledge-rag init` exports configuration templates and presets rather than rebuilding the index.

## Troubleshooting

**"Every tool answers that another knowledge-rag process owns this data directory"**
Another session owns the index, and `owner_pid` names it. Close that session, and this one takes over on its next call; or run one shared HTTP server for all clients (below).

**"Tools keep answering `repairing`"**
The owner is checking, opening or repairing the index; an embedding-model migration or a rebuild after a repair can take minutes on a large corpus. See `<data_dir>/preflight.log` and the server's stderr.

**"It says already running but I just killed the process"**
The OS releases ownership when the process exits. If refusals continue, check `owner_pid` and the processes using the directory; a PID file from an older version can name a reused PID (see [Upgrading](#upgrading-from-an-older-version)). Do not delete `.guard` or `.access` to force startup.

**"Stale lock left after Windows shutdown"**
A hard shutdown can leave the PID file behind; the next owner replaces it once it holds the OS lock. The sidecars are intentionally permanent and need no cleanup.

## Sharing one server across clients

The current server supports `server.transport: "streamable-http"` (the `/mcp` endpoint) and the legacy SSE transport. Run one server process holding the models and index; configure HTTP-capable clients to connect to that process rather than launching another stdio server. See [configuration](CONFIGURATION.md#server) and [installation](INSTALLATION.md).

Loopback addresses are reachable only on the machine making the connection. A cloud-hosted connector cannot reach a server on your laptop through that connector's `localhost`. Exposing a server beyond the local host requires deliberate authentication, TLS, and network configuration; client connection methods differ.

## Credits

- Original guard concept and reproduction: [Sergey Khokhlov (@Hohlas)](https://github.com/Hohlas) in [PR #31](https://github.com/lyonzin/knowledge-rag/pull/31).
- Reworked as opt-in with signal handlers and expanded tests, then made the default single owner per data directory with followers, take-over and the access lock: knowledge-rag maintainers.
