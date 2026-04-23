[![npm version](https://img.shields.io/npm/v/knowledge-rag.svg)](https://www.npmjs.com/package/knowledge-rag)

# Knowledge RAG

Local RAG system for Claude Code. Hybrid BM25 + semantic search with cross-encoder reranking. 12 MCP tools. Zero external servers. Everything runs on your machine.

## Quick Start

```bash
npx knowledge-rag
```

The wrapper handles Python venv creation, package installation, and server startup automatically.

## Claude Code Configuration

Add to your MCP settings (`~/.claude.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "knowledge-rag": {
      "command": "npx",
      "args": ["-y", "knowledge-rag"]
    }
  }
}
```

## CLI Flags

| Flag | Description |
|------|-------------|
| `--version` | Print version and exit |
| `--install-only` | Install the Python package into the venv without starting the server |

## Requirements

- **Node.js** >= 16
- **Python** >= 3.11

The wrapper auto-detects your Python installation across platforms (Windows: `python`, `py -3`; Linux/macOS: `python3`, `python`).

## How It Works

1. Finds a compatible Python 3.11+ interpreter on your system
2. Creates a persistent virtual environment at `~/.knowledge-rag/venv`
3. Installs the `knowledge-rag` PyPI package (skips if already up to date)
4. Starts the MCP server over stdio

The venv and installed package persist between runs. Reinstallation only happens when the NPM package version changes.

## Full Documentation

See the main repository for complete docs, configuration options, and tool reference:

**https://github.com/lyonzin/knowledge-rag**

## License

MIT
