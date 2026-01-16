# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-01-16

### Fixed

- Corrected CQL → LQL naming throughout documentation (LogScale Query Language)
- Simplified server.py imports and removed fallback execution logic
- Added auto-indexing on startup when database is empty

---

## [1.0.0] - 2025-01-14

### Added

- Initial release of Knowledge RAG System
- **Core Features**
  - ChromaDB v1.4.0 integration with PersistentClient API
  - Ollama embeddings via nomic-embed-text model (768 dimensions)
  - Multi-format document parsing (MD, PDF, TXT, PY, JSON)
  - Intelligent chunking with 1000 char size and 200 char overlap
  - Keyword-based routing for domain-specific searches
  - Semantic search with normalized relevance scores

- **MCP Integration**
  - `search_knowledge` - Main search tool with category filtering
  - `get_document` - Retrieve full document content
  - `reindex_documents` - Manual reindexing with force option
  - `list_categories` - List all categories with counts
  - `list_documents` - List indexed documents
  - `get_index_stats` - Index statistics and configuration

- **Document Categories**
  - `security` - Pentest, exploit, vulnerability documentation
  - `ctf` - CTF challenges and writeups
  - `logscale` - LogScale/LQL documentation
  - `development` - Code, APIs, and frameworks
  - `general` - Everything else

- **Automation**
  - PowerShell installation script (`install.ps1`)
  - Automatic Python 3.12 installation
  - Automatic Ollama installation
  - Automatic embedding model download
  - MCP configuration setup

- **Documentation**
  - Comprehensive README with architecture diagrams
  - API reference for all MCP tools
  - Installation guide (automated and manual)
  - Troubleshooting section

### Technical Details

- **Dependencies**
  - chromadb >= 1.4.0
  - pymupdf >= 1.23.0
  - ollama >= 0.6.0
  - mcp >= 1.0.0

- **Compatibility**
  - Python 3.11, 3.12 (3.13+ NOT supported due to onnxruntime)
  - Windows 10/11
  - Claude Code CLI

### Known Issues

- ChromaDB telemetry is enabled by default (can be ignored)
- PDF parsing may be slow for very large files (100+ pages)
- Ollama must be running before starting the MCP server

---

## [Unreleased]

### Planned

- [ ] File watcher for automatic reindexing
- [ ] Query caching for frequent searches
- [ ] Web UI for document management
- [ ] Linux/macOS support
- [ ] Docker containerization
- [ ] Support for additional embedding models
- [ ] Incremental indexing (only changed files)
