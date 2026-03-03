# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-03-03

### Added

- **Incremental Indexing**: Tracks file `mtime` and `size` to detect changes. Only re-indexes new or modified files instead of reprocessing everything. Automatically cleans up orphaned chunks from deleted/modified files.
- **Query Cache (LRU + TTL)**: Caches recent search queries with 5-minute TTL and 100-entry LRU eviction. Instant response for repeat queries. Cache is automatically invalidated on reindex.
- **Chunk Deduplication**: Content-hash based dedup using SHA256. Identical chunks across documents are stored only once, reducing index bloat.
- **Score Normalization**: RRF scores are now normalized to 0-1 range (1.0 = best match, 0.0 = worst in result set). Raw RRF score preserved in `raw_rrf_score` field.
- **Cache Statistics**: `get_index_stats` now returns query cache hit rate, unique content hashes, and dedup metrics.
- **Orphan Cleanup**: `index_all()` now detects and removes chunks from files that were deleted from disk.

### Changed

- `index_all()` stats now include `updated`, `deleted`, `chunks_removed`, and `dedup_skipped` counters
- `search_knowledge` response includes `cache_hit_rate` field
- `_index_document` now returns `(chunks_added, dedup_skipped)` tuple
- Metadata now stores `file_mtime` and `file_size` for change detection

### Fixed

- Orphaned chunks from modified files were never cleaned up from ChromaDB
- Re-indexing a modified file created duplicate chunks with new IDs

---

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

- [ ] File watcher for automatic reindexing (watchdog)
- [ ] Web UI for document management
- [ ] Linux/macOS support
- [ ] Docker containerization
- [ ] Support for additional embedding models
- [ ] BM25 index persistence (avoid rebuild on restart)
- [ ] Query reformulation (auto-retry with different hybrid_alpha)
- [ ] Parent-child chunk retrieval (get adjacent chunks)
