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
- Embedding model (since v3.8.0 this is **lazy-loaded** — idle processes are cheap; only processes that actually serve queries pay the ~200MB cost)
- ChromaDB client + SQLite handles
- BM25 in-memory index
- Watchdog file observer

If you've measured and confirmed you really do want a hard cap of one server per data directory, this flag is for you.

## When it is safe to leave the flag unset

The flag is only safe to leave **off** if every process against the same `data_dir` is a **read-only** consumer (no indexing, no watcher, no writes). Concrete cases:

- Multiple Claude Code / Desktop windows configured with `KNOWLEDGE_RAG_WATCHER_DISABLED=1`, doing only `search_knowledge` / `get_document`
- An analytics pipeline running `search_similar` in parallel against a static, already-indexed corpus

If any process can index, receive watcher events, or call an MCP write tool (`add_document`, `add_from_url`, `update_document`, `remove_document`, `reindex_documents`), that process is a writer and **must be the only writer** against the `data_dir`. Enable the flag or manually guarantee no other process touches the directory.

> **Windows note.** The ChromaDB HNSW segment format is not safe under concurrent writers on NTFS — the segment can corrupt silently and only shows up minutes later as a `chroma preflight failed with code 3221225477` (access violation) on restart. The single-instance flag is the pragmatic guard until the shared-service architecture in the [Roadmap](#roadmap) ships.

## Administrative commands (manual operations)

If you need to run `knowledge-rag` command-line for administrative work — `knowledge-rag init`, forced full reindex, backup, moving `data_dir` — do this order:

1. **Close** your MCP client (Claude Desktop / Code / Cursor). This kills the child process holding the `data_dir`.
2. Run your command in the terminal: `knowledge-rag <subcommand>`.
3. Wait for the command to finish.
4. **Reopen** your MCP client. It respawns the server, watcher picks up any new files.

Running the terminal command **while the MCP client is open** is the failure mode: two processes hold the same segment files, watcher events interleave with your CLI writes, and the HNSW segment can wedge. If the single-instance flag is enabled, the terminal command will refuse to start with exit code 75 — read that as a safety net telling you to close the MCP client first, not as a bug.

## Activation

Set the environment variable in your MCP client config. Accepted truthy values (case-insensitive, surrounding whitespace ignored): `1`, `true`, `yes`, `on`. Anything else, including unset, leaves the guard disabled.

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

- On startup, the server creates `<data_dir>/knowledge-rag.lock` with `O_CREAT | O_EXCL` and writes its PID inside.
- A second startup attempt finds the existing file, reads the PID, and probes whether that PID is still alive.
  - **Alive** -> exits with `AlreadyRunningError`.
  - **Dead** (crashed, killed) -> the lock is recognized as stale, removed, and the new process acquires it.
- Cleanup runs in three places so the lock never outlives the process:
  1. Normal exit -> `finally` block in the contextmanager removes the lock.
  2. `SIGINT` / `SIGTERM` -> handlers remove the lock and re-raise the signal so the original disposition fires.
  3. `SIGKILL` / hard crash -> stale-PID detection on the next startup recovers it.

The lock is per-data-directory. Two RAGs configured with different `data_dir` values do not collide.

## Troubleshooting

**"It says already running but I just killed the process"**
The lock should self-recover via stale-PID detection on the next startup. If it doesn't (e.g. PID was reused by an unrelated process), delete `<data_dir>/knowledge-rag.lock` manually.

**"I want this off again"**
Remove the env var from your MCP client config (or set it to `0` / `false`). On the next launch the guard is a complete no-op — no lock file is created and no checks happen.

**"Stale lock left after Windows shutdown"**
Expected. Stale-PID detection clears it on the next startup. You can also delete the file manually.

## Roadmap

The single-instance guard is a stop-gap. The proper fix for shared resources across multiple MCP clients is a **shared service architecture** (one daemon holding the model + index, many thin MCP clients connecting via socket). That work is tracked for v4.0.

## Credits

- Original guard concept and reproduction: [Sergey Khokhlov (@Hohlas)](https://github.com/Hohlas) in [PR #31](https://github.com/lyonzin/knowledge-rag/pull/31).
- Reworked as opt-in (default off), signal handlers wired, expanded test coverage: knowledge-rag maintainers.
