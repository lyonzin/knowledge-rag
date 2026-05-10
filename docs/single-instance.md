# Single-Instance Mode (opt-in)

> **TL;DR**: knowledge-rag now supports an opt-in flag that prevents more than one server process from running against the same data directory at the same time. **It is OFF by default** — multi-client MCP usage continues to work exactly as before. Turn it on only if you know you want it.

---

## When you might want this

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

## When you should NOT enable it

- You run multiple Claude Code windows simultaneously and want all of them to use this RAG
- You have Claude Desktop + an IDE + a terminal session all wired to the same RAG
- You run knowledge-rag inside an automation that expects to be able to spin up parallel readers
- You don't have a measured memory problem

For all of the above, **leave the flag unset**. The lazy-load improvement in v3.8.0 already cuts most of the per-process cost.

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
