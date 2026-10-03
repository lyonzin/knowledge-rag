# Installation Guide

> Complete step-by-step installation for **knowledge-rag** on Windows, Linux, and macOS — plus MCP client configuration for Claude Code, Claude Desktop, Cursor, Windsurf, VS Code, Cline, Gemini CLI, and Zed.

**Related docs:**
- [Configuration reference →](CONFIGURATION.md)
- [API reference →](API.md)
- [Architecture →](ARCHITECTURE.md)
- [Troubleshooting →](TROUBLESHOOTING.md)

**Quick links:**
- [Prerequisites](#prerequisites) · [GPU acceleration](#gpu-acceleration) · [Install methods](#install-methods) · [MCP client integrations](#use-with-other-mcp-clients) · [Verify](#verify)

---

### Prerequisites

- Python 3.11–3.13, the versions covered by the release CI matrix
- Claude Code CLI
- *…or any other MCP client (Claude Desktop, Cursor, VS Code, Antigravity, opencode, Windsurf) — see [Use with other MCP clients](#use-with-other-mcp-clients)*
- ~200MB disk for model cache (auto-downloaded on first run)
- *Optional:* NVIDIA GPU + CUDA 12 for accelerated embeddings (see [GPU Acceleration](#gpu-acceleration) below)

### GPU Acceleration

The default `models.embedding.gpu: "auto"` tries NVIDIA CUDA on the **first embedding request**, then falls back to CPU if the probe or CUDA model load fails. Set `gpu: "false"` to skip GPU probing. CPU works on Windows, Linux, and macOS. Windows AMD users can opt into experimental `gpu: "directml"` with an explicit DXGI `device_id`; see the [tested setup and safety checks](gpu-setup.md#amd-and-apple-gpu-status). Linux AMD and Apple GPU providers are not selected.

Use the [GPU setup guide](gpu-setup.md) before installing GPU packages. As verified on **2026-10-03**, ONNX Runtime 1.27+ wheels from PyPI default to CUDA 13, while this project's `[gpu]` extra targets CUDA 12 with `onnxruntime-gpu>=1.21,<1.27`, matching runtime libraries, and cuDNN 9. Create an isolated GPU environment and check its driver/runtime compatibility.

The extra includes eight NVIDIA CUDA 12 packages. It can also pull in the CPU `onnxruntime` distribution through transitive dependencies; CPU/GPU wheels share a module namespace. The guide documents this packaging limitation and the controlled runtime substitution needed in a separate environment. Installing the extra alone or seeing CUDA in `get_available_providers()` does not prove GPU inference works.

After an actual embedding, look for the stderr banner and inspect the loaded session's providers. The automatic ONNX micro-batch is CPU=32, CUDA=256, or DirectML=8 based on the actual session; override with `documents.embed_batch_size` or a positive `KNOWLEDGE_RAG_EMBED_BATCH_SIZE` (maximum 512). This is independent of ChromaDB's write batch.

GPU selection here controls embeddings. Reranking uses a separate FastEmbed session with upstream defaults. If CUDA fails, CPU fallback keeps the same embedding model; if CPU loading or inference also fails, the server reports the error rather than manufacturing vectors. See the guide for real-hardware validation limits and AMD/CoreML status.

### Install Methods

Pick one — all produce the same running server.

#### Option A: NPX (fastest)

Requires Node.js 16+. Handles Python venv, pip install, and version upgrades automatically.

```bash
claude mcp add knowledge-rag -s user -- npx -y knowledge-rag
```

That's it. On first run, `npx` creates a venv at `~/.knowledge-rag/`, installs the PyPI package, and starts the MCP server. Subsequent runs reuse the cached venv.

#### Option B: Installer from a source checkout

The shell wrappers delegate to the adjacent `install.py`; downloading only a
wrapper and piping it into a shell does not provide that file. Clone the
repository first:

```bash
# Linux/macOS:
git clone https://github.com/lyonzin/knowledge-rag.git ~/knowledge-rag
cd ~/knowledge-rag
bash install.sh --from-source
```

```powershell
# Windows:
git clone https://github.com/lyonzin/knowledge-rag.git "$HOME/knowledge-rag"
Set-Location "$HOME/knowledge-rag"
.\install.ps1 -SkipPython --from-source
```

The installer detects clients and prepares their configurations. Use
`python install.py --help` to select clients or inspect dry-run options.
If configuring Claude Code manually:

```bash
claude mcp add knowledge-rag -s user -- ~/knowledge-rag/venv/bin/python -m mcp_server.server
```

> **Windows**: `claude mcp add knowledge-rag -s user -- %USERPROFILE%\knowledge-rag\venv\Scripts\python.exe -m mcp_server.server`

#### Option C: pip install

```bash
mkdir ~/knowledge-rag && cd ~/knowledge-rag
python3 -m venv venv && source venv/bin/activate
pip install knowledge-rag
knowledge-rag init              # Exports config template, presets, creates documents/
```

Then configure Claude Code:

```bash
claude mcp add knowledge-rag -s user -- ~/knowledge-rag/venv/bin/python -m mcp_server.server
```

> **Windows users**: Use `python` instead of `python3`, `venv\Scripts\activate` instead of `source venv/bin/activate`.
> **Windows path**: `claude mcp add knowledge-rag -s user -- %USERPROFILE%\knowledge-rag\venv\Scripts\python.exe -m mcp_server.server`

#### Option D: Clone from source

```bash
git clone https://github.com/lyonzin/knowledge-rag.git ~/knowledge-rag
cd ~/knowledge-rag
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Then configure Claude Code:

```bash
claude mcp add knowledge-rag -s user -- ~/knowledge-rag/venv/bin/python -m mcp_server.server
```

#### Option E: Docker

```bash
docker pull ghcr.io/lyonzin/knowledge-rag:latest
```

```bash
claude mcp add knowledge-rag -s user -- \
  docker run -i --rm \
  -v ~/knowledge-rag/documents:/app/documents \
  -v ~/knowledge-rag/data:/app/data \
  ghcr.io/lyonzin/knowledge-rag:latest
```

Models are pre-downloaded in the image — no first-run delay.

<details>
<summary>Alternative: manual JSON config</summary>

Add to `~/.claude.json`:

**Windows:**
```json
{
  "mcpServers": {
    "knowledge-rag": {
      "command": "C:\\Users\\YOUR_USER\\knowledge-rag\\venv\\Scripts\\python.exe",
      "args": ["-m", "mcp_server.server"]
    }
  }
}
```

**Linux / macOS:**
```json
{
  "mcpServers": {
    "knowledge-rag": {
      "command": "/home/YOUR_USER/knowledge-rag/venv/bin/python",
      "args": ["-m", "mcp_server.server"]
    }
  }
}
```
> Replace `YOUR_USER` with your username, or use the full path from `echo $HOME`.
</details>

#### Option F: SSE Server Mode (multi-agent)

For multi-agent setups where multiple clients query the same knowledge base simultaneously:

```bash
pip install knowledge-rag[server]    # Adds uvicorn for SSE/HTTP
knowledge-rag --transport sse        # Starts on http://127.0.0.1:8179
```

Then configure each MCP client to connect via SSE:

```json
{
  "mcpServers": {
    "knowledge-rag": {
      "type": "sse",
      "url": "http://127.0.0.1:8179/sse"
    }
  }
}
```

One server process serves all agents — shared embedding model, shared cache, shared ChromaDB. See [Configuration > Server](#server) for rate limiting, metrics, and auth options.

### Use with other MCP clients

`knowledge-rag` supports both **stdio** (default, 1:1) and **SSE** (1:N) transport modes. In stdio mode, it works with any MCP-compatible client, not only Claude Code. The launch command is the same everywhere (the `python -m mcp_server.server` from whichever install method you picked); only the **config file location** and **JSON shape** differ per client.

#### Clients using the standard `mcpServers` format

For **Claude Desktop, Cursor, Antigravity, and Windsurf**, use the same block — only the file location changes:

```json
{
  "mcpServers": {
    "knowledge-rag": {
      "command": "/home/YOUR_USER/knowledge-rag/venv/bin/python",
      "args": ["-m", "mcp_server.server"]
    }
  }
}
```

> **Windows**: set `command` to the full path of `venv\Scripts\python.exe`.

| Client | Config file | Notes |
|---|---|---|
| **Claude Code** | use `claude mcp add …` (see install methods above) | The CLI writes `~/.claude.json` for you — manual edits to it aren't reliably picked up. |
| **Claude Desktop** | macOS: `~/Library/Application Support/Claude/claude_desktop_config.json` · Windows: `%APPDATA%\Claude\claude_desktop_config.json` | Easiest: **Settings → Developer → Edit Config** opens the correct file (avoids the Windows Store/MSIX path quirk). |
| **Cursor** | `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per project) | — |
| **Antigravity** | macOS/Linux: `~/.gemini/antigravity/mcp_config.json` · Windows: `%USERPROFILE%\.gemini\antigravity\mcp_config.json` | Open via Agent panel → **"…" → Manage MCP Servers → View raw config**. |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` (global only) | Easiest: Cascade panel → MCP → **View raw config**. |

#### VS Code — uses a `servers` key

VS Code (Copilot MCP) nests servers under **`servers`**, not `mcpServers`. Put this in `.vscode/mcp.json` (workspace) or the file opened by the **MCP: Open User Configuration** command:

```json
{
  "servers": {
    "knowledge-rag": {
      "type": "stdio",
      "command": "/home/YOUR_USER/knowledge-rag/venv/bin/python",
      "args": ["-m", "mcp_server.server"]
    }
  }
}
```

#### opencode — uses an `mcp` key

opencode nests servers under **`mcp`**, takes `command` as a single **array**, and uses `environment` instead of `env`. Put this in `opencode.json` (project root) or `~/.config/opencode/opencode.json` (global):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "knowledge-rag": {
      "type": "local",
      "command": ["/home/YOUR_USER/knowledge-rag/venv/bin/python", "-m", "mcp_server.server"],
      "enabled": true
    }
  }
}
```

> **Any other MCP client**: point it at the same command + args (`…/venv/bin/python -m mcp_server.server`). If it speaks stdio MCP, knowledge-rag works — only the config file's location and key naming differ. Check your client's docs for the exact path.

### Verify

```bash
claude mcp list
```

On first start, the server will:
1. Download the embedding model (~50MB, cached in `models_cache/`)
2. Auto-index any documents in the `documents/` directory
3. Start watching for file changes (auto-reindex)

---
