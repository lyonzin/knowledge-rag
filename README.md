# Knowledge RAG System

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.1-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)

**Local RAG (Retrieval-Augmented Generation) System for Claude Code**

*Semantic search + keyword routing for your personal knowledge base*

[Features](#features) | [Installation](#installation) | [Usage](#usage) | [API Reference](#api-reference) | [Architecture](#architecture)

</div>

---

## Overview

Knowledge RAG is a **100% local** semantic search system that integrates with Claude Code via MCP (Model Context Protocol). It enables Claude to search through your documents (PDFs, Markdown, code, etc.) and retrieve relevant context for answering questions.

### Why Knowledge RAG?

- **Privacy First**: All processing happens locally - no data leaves your machine
- **Multi-Format**: Supports MD, PDF, TXT, Python, JSON files
- **Smart Routing**: Keyword-based routing ensures accurate category matching
- **Claude Integration**: Native MCP tools for seamless Claude Code integration
- **Fast**: Vector search with ChromaDB + local Ollama embeddings

---

## Features

| Feature | Description |
|---------|-------------|
| **Semantic Search** | Find documents by meaning, not just keywords |
| **Keyword Routing** | Deterministic routing for domain-specific queries |
| **Multi-Format Parser** | PDF, Markdown, TXT, Python, JSON support |
| **Chunking with Overlap** | Smart text splitting with context preservation |
| **Category Organization** | Organize docs by security, development, logscale, etc. |
| **MCP Integration** | Native Claude Code tools |
| **Persistent Storage** | ChromaDB with DuckDB backend |
| **Local Embeddings** | Ollama + nomic-embed-text (768 dimensions) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      KNOWLEDGE ORCHESTRATOR                         │
│                       (MCP Server via FastMCP)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────────┐  │
│  │   KEYWORD    │ → │   SEMANTIC   │ → │   CONTEXT ASSEMBLER     │  │
│  │   ROUTER     │   │   SEARCH     │   │   (Ranking + Merge)     │  │
│  └──────────────┘   └──────────────┘   └─────────────────────────┘  │
│         │                  │                        │               │
│         ▼                  ▼                        ▼               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                VECTOR DATABASE (ChromaDB)                   │    │
│  │   Collections: security | ctf | logscale | development      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  OLLAMA EMBEDDINGS                          │    │
│  │                  (nomic-embed-text)                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                 DOCUMENT INGESTION                          │    │
│  │   Parsers: Markdown | PDF (PyMuPDF) | TXT | Python | JSON   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Document Ingestion**: Files are parsed, chunked (1000 chars, 200 overlap)
2. **Embedding Generation**: Ollama creates 768-dim vectors via nomic-embed-text
3. **Vector Storage**: ChromaDB stores vectors with metadata
4. **Query Processing**:
   - Keyword router checks for domain-specific terms
   - If matched, filters to specific category
   - Semantic search finds similar chunks
   - Results ranked and returned

---

## Installation

### Prerequisites

- Windows 10/11
- Python 3.11 or 3.12
- [Ollama](https://ollama.com) (for local embeddings)
- Claude Code CLI

### Quick Install (Automated)

```powershell
# Clone the repository
git clone https://github.com/yourusername/knowledge-rag.git
cd knowledge-rag

# Run the installer
.\install.ps1
```

### Manual Installation

1. **Install Python 3.12**
   ```powershell
   # Download from https://www.python.org/downloads/
   # Or use winget:
   winget install Python.Python.3.12
   ```

2. **Install Ollama**
   ```powershell
   # Download from https://ollama.com
   # Or use winget:
   winget install Ollama.Ollama
   ```

3. **Pull the embedding model**
   ```powershell
   ollama pull nomic-embed-text
   ```

4. **Clone and setup the project**
   ```powershell
   git clone https://github.com/yourusername/knowledge-rag.git
   cd knowledge-rag

   # Create virtual environment
   python -m venv venv
   .\venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

5. **Configure MCP for Claude Code**

   Add to `~/.claude.json` under `mcpServers`:
   ```json
   {
     "mcpServers": {
       "knowledge-rag": {
         "type": "stdio",
         "command": "cmd",
         "args": ["/c", "cd /d C:\\path\\to\\knowledge-rag && .\\venv\\Scripts\\python.exe -m mcp_server.server"],
         "env": {}
       }
     }
   }
   ```

   > **Note**: We use `cmd /c` with `cd /d` to ensure the working directory is set correctly before starting the Python server. This is required because Claude Code may not respect the `cwd` property in MCP configurations.

6. **Restart Claude Code**

---

## Usage

### Adding Documents

Place your documents in the `documents/` directory, organized by category:

```
documents/
├── security/          # Pentest, exploit, vulnerability docs
│   ├── RTFM.pdf
│   └── web-hacking-101.pdf
├── logscale/          # LogScale/LQL documentation
│   └── LQL_REFERENCE.md
├── development/       # Code, APIs, frameworks
│   └── api-docs.md
└── general/           # Everything else
    └── notes.txt
```

### Indexing Documents

Documents are automatically indexed when Claude Code starts. To manually reindex:

```
# In Claude Code chat:
Use the reindex_documents tool to reindex all documents
```

### Searching

Simply ask Claude questions! The RAG system automatically provides context:

```
User: How do I use formatTime in LogScale?
Claude: [Uses search_knowledge internally, retrieves relevant chunks]
        Based on your documentation, formatTime in LogScale...
```

### Direct Tool Usage

You can also use the MCP tools directly:

```javascript
// Search knowledge base
search_knowledge("buffer overflow exploitation", max_results=5, category="security")

// Get full document
get_document("C:/path/to/document.pdf")

// List all categories
list_categories()

// Get index statistics
get_index_stats()
```

---

## API Reference

### MCP Tools

#### `search_knowledge`

Search the knowledge base using semantic search with keyword routing.

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | string | required | Search query text |
| `max_results` | int | 5 | Maximum results (1-20) |
| `category` | string | null | Filter by category |

**Returns:** JSON with search results including content, source, and relevance score.

**Example:**
```json
{
  "status": "success",
  "query": "SQL injection prevention",
  "result_count": 3,
  "results": [
    {
      "content": "To prevent SQL injection...",
      "source": "C:/docs/security/web-hacking.pdf",
      "filename": "web-hacking.pdf",
      "category": "security",
      "score": 0.8234,
      "routed_by": "security"
    }
  ]
}
```

#### `get_document`

Retrieve the full content of a specific document.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `filepath` | string | Path to the document |

**Returns:** JSON with document content and metadata.

#### `reindex_documents`

Index or reindex all documents in the knowledge base.

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `force` | bool | false | If true, clears and rebuilds entire index |

**Returns:** JSON with indexing statistics.

#### `list_categories`

List all document categories with their document counts.

**Returns:**
```json
{
  "status": "success",
  "categories": {
    "security": 11,
    "logscale": 1,
    "development": 3
  },
  "total_documents": 15
}
```

#### `list_documents`

List all indexed documents, optionally filtered by category.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `category` | string | Optional category filter |

#### `get_index_stats`

Get statistics about the knowledge base index.

**Returns:**
```json
{
  "status": "success",
  "stats": {
    "total_documents": 15,
    "total_chunks": 8061,
    "categories": {"security": 11, "logscale": 1},
    "embedding_model": "nomic-embed-text",
    "chunk_size": 1000,
    "chunk_overlap": 200
  }
}
```

---

## Configuration

### Keyword Routing

The system uses keyword routing to improve search accuracy. Configure routes in `mcp_server/config.py`:

```python
keyword_routes = {
    "security": ["exploit", "vulnerability", "pentest", "attack", ...],
    "ctf": ["ctf", "flag", "hackthebox", "tryhackme", ...],
    "logscale": ["logscale", "humio", "lql", "formatTime", ...],
    "development": ["python", "javascript", "api", "docker", ...]
}
```

When a query contains these keywords, results are filtered to the matching category.

### Chunking Settings

Adjust chunk size and overlap in `config.py`:

```python
chunk_size = 1000      # Characters per chunk
chunk_overlap = 200    # Overlap between chunks
```

### Embedding Model

The default model is `nomic-embed-text`. To change:

1. Pull a different model: `ollama pull <model-name>`
2. Update `config.py`: `ollama_model = "<model-name>"`

---

## Project Structure

```
knowledge-rag/
├── mcp_server/
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── ingestion.py       # Document parsing & chunking
│   └── server.py          # MCP server & ChromaDB
├── documents/             # Your documents go here
│   ├── security/
│   ├── logscale/
│   ├── development/
│   └── general/
├── chroma_db/             # Vector database storage
├── .claude/
│   └── mcp.json           # Project MCP config
├── venv/                  # Python virtual environment
├── install.ps1            # Automated installer
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Troubleshooting

### Ollama not running

```powershell
# Start Ollama
ollama serve

# Or check if running
curl http://localhost:11434/api/tags
```

### Python version mismatch

ChromaDB requires Python 3.11 or 3.12. Python 3.13+ is NOT supported due to onnxruntime compatibility.

```powershell
# Check version
python --version

# Use specific version
py -3.12 -m venv venv
```

### Index is empty

```powershell
# Check documents directory
ls documents/

# Force reindex
# In Claude Code: use reindex_documents(force=true)
```

### MCP server not loading

1. Check `~/.claude.json` exists and has `mcpServers` section with valid JSON
2. Verify paths use double backslashes (`\\`) on Windows
3. Restart Claude Code completely
4. Run `claude mcp list` to check connection status

### "ModuleNotFoundError: No module named 'mcp_server'"

This error occurs when Claude Code doesn't set the working directory correctly. **Solution**: Use the `cmd /c "cd /d ... && python"` wrapper in your config:

```json
{
  "knowledge-rag": {
    "type": "stdio",
    "command": "cmd",
    "args": ["/c", "cd /d C:\\path\\to\\knowledge-rag && .\\venv\\Scripts\\python.exe -m mcp_server.server"],
    "env": {}
  }
}
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Ollama](https://ollama.com/) - Local LLM & embeddings
- [FastMCP](https://github.com/anthropics/mcp) - MCP framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF parsing

---

## Author

**Ailton Rocha (Lyon)**

AI Operator | Security Researcher | Developer

---

<div align="center">

**[Back to Top](#knowledge-rag-system)**

</div>
