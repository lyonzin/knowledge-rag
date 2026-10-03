# API Reference — 13 MCP Tools

> Reference for the 13 MCP tools exposed by knowledge-rag. Existing parameter names and defaults are covered by `tests/test_backwards_compat.py`; additive optional parameters are also checked by the public API surface gate.

**Compatibility:** MCP spec 2026-07-28 · Anthropic Tier 1 SDK (`mcp>=2.0.0,<3.0.0`) · Claude Code · Claude Desktop · Cursor · Windsurf · VS Code (Copilot) · Cline · Gemini CLI · Zed.

**Related docs:**
- [Configuration reference →](CONFIGURATION.md)
- [Installation guide →](INSTALLATION.md)
- [Architecture →](ARCHITECTURE.md)
- [Troubleshooting →](TROUBLESHOOTING.md)

---

### Search & Query

#### `search_knowledge`

Search using semantic retrieval and BM25 with optional cross-encoder reranking, or the opt-in FTS5 lexical index.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Search query text; use identifiers for lexical search or a question for semantic search |
| `max_results` | int | 5 | Maximum results to return; clamped to `search.max_results` (100 by default) |
| `category` | string | null | Filter by category |
| `hybrid_alpha` | float | 0.3 | Balance: 0.0 = keyword only, 1.0 = semantic only |
| `min_score` | float | 0.0 | Minimum relevance score (0.0-1.0) to include a result. Use 0.2-0.4 to cut noise |
| `snippet_mode` | bool | true | Truncate content to ~500 chars at natural break points. Adds `content_length` field |
| `search_method` | string | "auto" | `auto`, `hybrid`, or `fts5`; explicit `fts5` requires the feature to be enabled and ready |

`auto` uses the configured lexical router only when FTS5 is enabled. It falls back to hybrid search when the lexical index is unavailable, fails, or returns too few useful hits. `hybrid` always bypasses that router. An explicit `fts5` request returns a structured error if the index is disabled or not ready; it does not silently switch methods. `hybrid_alpha` controls the hybrid path, not FTS5 ranking. Scores are ranking signals, not calibrated probabilities, and should not be compared as confidence across methods.

**Returns:**

```json
{
  "status": "success",
  "query": "mimikatz credential dump",
  "hybrid_alpha": 0.5,
  "result_count": 3,
  "filtered_by_score": 2,
  "cache_hit_rate": "0.0%",
  "results": [
    {
      "content": "Mimikatz can extract credentials from memory...",
      "source": "documents/security/credential-attacks.md",
      "filename": "credential-attacks.md",
      "category": "security",
      "score": 0.9823,
      "raw_rrf_score": 0.016393,
      "reranker_score": 0.987654,
      "semantic_rank": 2,
      "bm25_rank": 1,
      "search_method": "hybrid",
      "keywords": ["mimikatz", "credential", "lsass"],
      "routed_by": "redteam"
    }
  ]
}
```

**Search Method Values:**
- `hybrid`: Found by both semantic and BM25 search
- `semantic`: Found only by semantic search
- `keyword`: Found only by BM25 keyword search
- `fts5`: Retrieved through the lexical fast path; optional reranking retains this label

The `routed_by` field reports category keyword routing for diagnostics. It does not impose a category filter or boost. Pass `category` explicitly to restrict results.

---

#### `get_document`

Retrieve the full content of a specific document.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | string | Path to the document file |

**Returns:** JSON with document content, metadata, keywords, and chunk count.

---

#### `reindex_documents`

Index or reindex all documents in the knowledge base. **Runs in background** — returns immediately. Poll progress via `get_reindex_status()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force` | bool | false | Reprocess and re-embed all discovered documents, including unchanged files, with a smart-reindex checkpoint. Without it, unchanged files are skipped. |
| `full_rebuild` | bool | false | Re-embed all documents into a staging collection and replace the live collection after validation. Requires a running orchestrator and temporary storage for both collections. |
| `resume` | bool | false | Resume a compatible smart-reindex checkpoint; missing, corrupt, or incompatible checkpoints start a fresh smart run. Cannot be combined with `full_rebuild=true`. |

**Returns:** `{"status": "started", "operation": "..."}` immediately. If already running, returns `{"status": "already_running", "progress": "1200/3734"}`.

Poll until `reindex.active` is false, then inspect `last_error` and `last_result.errors`. An idle worker is not proof of a successful rebuild. Smart reindexing writes a resumable checkpoint; see the [operations guide](reindex-operations.md).

---

#### `get_reindex_status`

Get the current status of a background reindex operation. Lightweight — does not compute full index statistics.

**Returns (active):**
```json
{
  "status": "success",
  "reindex": {
    "active": true,
    "operation": "nuclear_rebuild",
    "progress": "1200/3734",
    "percent": 32,
    "indexed": 1200,
    "skipped": 0,
    "errors": 0,
    "started_at": "2026-06-17T18:29:49"
  }
}
```

**Returns (idle):** `{"status": "success", "reindex": {"active": false, ...}}`, with the last operation's result or error when available.

---

#### `list_categories`

List all document categories with their document counts.

**Returns:**

```json
{
  "status": "success",
  "categories": {
    "security": 52,
    "development": 8,
    "ctf": 12,
    "general": 3
  },
  "total_documents": 75
}
```

---

#### `list_documents`

List all indexed documents, optionally filtered by category.

| Parameter | Type | Description |
|-----------|------|-------------|
| `category` | string | Optional category filter |

**Returns:** JSON array of documents with id, source, category, format, chunks, and keywords.

---

#### `get_index_stats`

Get statistics about the knowledge base index.

**Returns:**

```json
{
  "status": "success",
  "stats": {
    "total_documents": 75,
    "total_chunks": 9256,
    "categories": {"security": 52, "development": 8},
    "supported_formats": [".md", ".txt", ".pdf", ".py", ".json", ".docx", ".xlsx", ".pptx", ".csv", ".ipynb"],
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "embedding_dim": 384,
    "reranker_model": "Xenova/ms-marco-MiniLM-L-6-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "query_cache": {
      "size": 12,
      "max_size": 100,
      "ttl_seconds": 300,
      "hits": 45,
      "misses": 23,
      "hit_rate": "66.2%"
    }
  }
}
```

---

### Document Management

#### `add_document`

Add a new document to the knowledge base from raw content. Saves the file to the documents directory and indexes it immediately.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | string | required | Full text content of the document |
| `filepath` | string | required | Relative path within documents dir (e.g., `security/new-technique.md`) |
| `category` | string | "general" | Document category |

**Example (cookbook):**

```python
add_document(
    content="# Kerberoasting\n\nSteal Kerberos service tickets and crack them offline...",
    filepath="security/redteam/kerberoasting.md",
    category="redteam",
)
```

---

#### `update_document`

Update an existing document. Removes old chunks from the index and re-indexes with new content.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | string | Full path to the document file |
| `content` | string | New content for the document |

**Example:**

```python
update_document(
    filepath="security/redteam/kerberoasting.md",
    content="# Kerberoasting (updated 2026-08)\n\nAdded Rubeus /aes flag detail...",
)
```

---

#### `remove_document`

Remove a document from the knowledge base index. Optionally deletes the file from disk.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | string | required | Path to the document file |
| `delete_file` | bool | false | If true, also delete the file from disk |

**Example:**

```python
# Drop from the index only — keep the file
remove_document(filepath="security/legacy/old-runbook.md")

# Drop from the index AND delete the file from disk
remove_document(filepath="security/legacy/old-runbook.md", delete_file=True)
```

---

#### `add_from_url`

Fetch content from a URL, strip HTML (scripts, styles, nav, footer, header), convert to markdown, and add to the knowledge base. The URL body is wrapped in a provenance fence and injection sentinels are neutralized (OWASP LLM01:2025).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | required | URL to fetch content from |
| `category` | string | "general" | Document category |
| `title` | string | null | Custom title (auto-detected from `<title>` tag if not provided) |

**Example:**

```python
add_from_url(
    url="https://attack.mitre.org/techniques/T1558/003/",
    category="mitre",
    title="MITRE ATT&CK T1558.003 — Kerberoasting",
)
```

---

#### `search_similar`

Find documents similar to a given document using embedding similarity.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | string | required | Path to the reference document |
| `max_results` | int | 5 | Number of similar document chunks to return, bounded by `search.max_results` |

**Example:**

```python
# "Show me 5 docs semantically closest to my Kerberoasting note"
search_similar(
    filepath="security/redteam/kerberoasting.md",
    max_results=5,
)
```

---

#### `reindex_documents` — call examples

Four common patterns:

```python
# 1. Smart incremental — detects changes via mtime/size, only reindexes what moved.
#    Runs in the background; call get_reindex_status() to poll progress.
reindex_documents()

# 2. Forced smart reindex — re-embed even unchanged files; supports checkpoints.
reindex_documents(force=True)

# 3. Full rebuild — populate and validate a staging collection before swap.
#    Requires temporary space for both the old and replacement collections.
reindex_documents(full_rebuild=True)

# 4. Resume a compatible interrupted smart reindex.
reindex_documents(resume=True)
```

Changing the embedding model/profile can prevent the existing collection from
opening, so a subsequent MCP rebuild call is not always available. For model
migrations, use a separate data directory and metadata, index the same source
corpus, validate retrieval, then switch clients. See the
[model migration procedure](reindex-operations.md#changing-the-embedding-model).

---

#### `evaluate_retrieval` — full example

Measure retrieval quality against a ground-truth test set. Useful for tuning `hybrid_alpha`, testing query expansion effectiveness, or validating after reindexing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `test_cases` | string (JSON) | Array of test cases: `[{"query": "...", "expected_filepath": "..."}, ...]` |

**Complete example:**

```python
import json

test_cases = json.dumps([
    {"query": "kerberoasting", "expected_filepath": "security/redteam/kerberoasting.md"},
    {"query": "sql injection payloads", "expected_filepath": "security/webapp/sqli-cheatsheet.md"},
    {"query": "prometheus histogram buckets", "expected_filepath": "docs/observability/metrics.md"},
    {"query": "how to rotate refresh tokens", "expected_filepath": "docs/adr/0018-auth.md"},
    {"query": "chromadb sqlite variable limit", "expected_filepath": "CHANGELOG.md"},
])

evaluate_retrieval(test_cases=test_cases)
```

Returns `mrr_at_5`, `recall_at_5`, and a per-query breakdown showing which expected document was found and at what rank. Each case requires non-empty `query` and `expected_filepath` strings. Invalid cases return an error before executing searches. The tool does not return Precision@5.

**Metrics:**
- **MRR@5** (Mean Reciprocal Rank): Average of 1/rank for expected documents. 1.0 = always first result.
- **Recall@5**: Fraction of expected documents found in top 5 results. 1.0 = all found.

Use representative, independently selected questions and keep the corpus, model, prefixes, search settings, and test cases fixed when comparing runs. A single expected document per query measures hit rate and rank; it does not measure all relevant documents or establish a universal quality threshold. English BGE models need a separate evaluation for non-English corpora.

---

