# Configuration Reference

> Complete reference for `config.yaml`. If no `config.yaml` exists, knowledge-rag works out of the box with sensible defaults — configuration is optional.

**Related docs:**
- [API reference →](API.md)
- [Installation guide →](INSTALLATION.md)
- [Architecture →](ARCHITECTURE.md)
- [Troubleshooting →](TROUBLESHOOTING.md)

**Quick links:**
- [Quick Start](#quick-start) · [Full YAML template](#configyaml-structure) · [Presets](#presets) · [Field reference table](#configuration-reference) · [Hybrid tuning](#hybrid-search-tuning) · [FTS5 fast-path](#search-method-v482)

---

### Quick Start

```bash
# Option 1: Use a preset
cp presets/cybersecurity.yaml config.yaml    # Offensive/defensive security, CTFs
cp presets/developer.yaml config.yaml        # Software engineering, APIs, DevOps
cp presets/research.yaml config.yaml         # Academic research, papers, studies
cp presets/general.yaml config.yaml          # Blank slate, pure semantic search

# Option 2: Start from the documented template
cp config.example.yaml config.yaml
# Edit config.yaml to your needs
```

Restart Claude Code after changing `config.yaml`.

### config.yaml Structure

```yaml
# Paths — where your documents live
paths:
  documents_dir: "./documents"    # Scanned recursively
  data_dir: "./data"              # Index storage
  models_cache_dir: "./models_cache"  # Persistent embedding model cache

# Documents — what gets indexed and how
documents:
  supported_formats:              # File types to index
    - .md
    - .txt
    - .pdf
    - .docx
    - .ipynb
    # - .py                       # Uncomment to index code
  exclude_patterns:               # Glob patterns to skip
    - "node_modules"
    - ".venv"
    - "__pycache__"
  chunking:
    chunk_size: 1000              # Max chars per chunk
    chunk_overlap: 200            # Shared chars between chunks

# Models — AI models for search (all run locally, in-process)
models:
  embedding:
    model: "BAAI/bge-small-en-v1.5"   # ONNX, ~33MB, auto-downloaded
    dimensions: 384
    gpu: false                         # Set true + pip install knowledge-rag[gpu]
  reranker:
    enabled: true                      # Falls back to RRF if model is unavailable
    model: "Xenova/ms-marco-MiniLM-L-6-v2"
    top_k_multiplier: 3               # Candidates fetched before reranking

# Search — result limits and collection name
search:
  default_results: 5
  max_results: 20
  collection_name: "knowledge_base"   # Change for separate knowledge bases

# Categories — auto-tag documents by folder path
# Set to {} to disable categorization entirely
category_mappings:
  "security/redteam": "redteam"
  "security/blueteam": "blueteam"
  "notes": "notes"

# Keyword routing — prioritize categories based on query keywords
# Set to {} for pure semantic search with no routing bias
keyword_routes:
  redteam:
    - pentest
    - exploit
    - privilege escalation

# Query expansion — expand abbreviations for better BM25 recall
# Set to {} for no expansion (search terms used as-is)
query_expansions:
  sqli:
    - sql injection
    - sqli
  privesc:
    - privilege escalation
    - privesc

# Server — enterprise features (new in v4.0.0)
server:
  transport: "stdio"              # "stdio" | "sse" | "streamable-http"
  host: "127.0.0.1"              # Bind address (SSE/HTTP only)
  port: 8179                      # Bind port (SSE/HTTP only)
  auth:
    bearer_token: ""              # Set a secret to enable auth (SSE/HTTP only)
  rate_limit:
    enabled: false
    requests_per_minute: 60
    burst: 10
  metrics:
    enabled: false
    port: 9179                    # Separate port for Prometheus scraping
```

> See `config.example.yaml` for the fully documented template with explanations for every field.

### Presets

Pre-built configurations for common use cases:

| Preset | File | Categories | Keywords | Expansions | Best For |
|--------|------|-----------|----------|-----------|----------|
| **Cybersecurity** | `presets/cybersecurity.yaml` | 8 | 200+ | 69 | Red/Blue Team, CTFs, threat hunting, exploit dev |
| **Developer** | `presets/developer.yaml` | 9 | 150+ | 50+ | Full-stack dev, APIs, DevOps, cloud, databases |
| **Research** | `presets/research.yaml` | 9 | 100+ | 40+ | Academic papers, thesis, lab notebooks, datasets |
| **Multilingual** | `presets/multilingual.yaml` | 0 | 0 | 0 | Non-English corpora (100+ languages via `intfloat/multilingual-e5-large`, 1024D). Requires `reindex_documents(force=True)` after switching |
| **General** | `presets/general.yaml` | 0 | 0 | 0 | Blank slate — pure semantic search, no domain logic |

**Creating your own preset**: Copy `config.example.yaml`, fill in your categories/keywords/expansions, save to `presets/your-domain.yaml`.

### Organizing your `documents/` folder

`category_mappings` in `config.yaml` maps folder paths to category slugs. Any doc under a mapped path inherits that category automatically — no manual tagging needed. Example layout used by the **Cybersecurity** preset:

```
documents/
├── security/
│   ├── redteam/         # → category: redteam
│   ├── blueteam/        # → category: blueteam
│   ├── mitre/           # → category: mitre
│   └── ctf/             # → category: ctf
├── development/
│   ├── apis/            # → category: dev
│   ├── infra/           # → category: infra
│   └── runbooks/        # → category: ops
├── research/            # → category: research
├── notes/               # → category: notes
└── general/             # → category: general (fallback)
```

With `category_mappings` defined, a query like `search_knowledge(query="privilege escalation", category="redteam")` narrows the search to `documents/security/redteam/` only. Unmapped paths default to `general`. To disable categorization entirely, set `category_mappings: {}`.

### Configuration Reference

#### Server

| Field | Default | Description |
|-------|---------|-------------|
| `server.transport` | `"stdio"` | Transport protocol: `"stdio"`, `"sse"`, or `"streamable-http"` |
| `server.host` | `"127.0.0.1"` | Bind address for SSE/HTTP mode |
| `server.port` | `8179` | Bind port for SSE/HTTP mode |
| `server.auth.bearer_token` | `""` (disabled) | Bearer token for SSE/HTTP auth. Empty = no auth |
| `server.rate_limit.enabled` | `false` | Enable per-client rate limiting |
| `server.rate_limit.requests_per_minute` | `60` | Max requests per minute |
| `server.rate_limit.burst` | `10` | Burst allowance above steady rate |
| `server.metrics.enabled` | `false` | Enable Prometheus `/metrics` endpoint |
| `server.metrics.port` | `9179` | Port for metrics scraping |

In stdio mode (default), server settings are ignored. SSE/HTTP mode auto-enables the single-instance lock.

#### Paths

| Field | Default | Description |
|-------|---------|-------------|
| `paths.documents_dir` | `./documents` | Root folder scanned recursively for documents |
| `paths.data_dir` | `./data` | Internal storage for ChromaDB and index metadata |
| `paths.models_cache_dir` | `./models_cache` | Persistent cache for embedding models (~250MB). Survives reboots |

Relative paths resolve from the project root. Absolute paths work too.

#### Documents

| Field | Default | Description |
|-------|---------|-------------|
| `documents.supported_formats` | All 33 non-MetaTrader formats (see README table) | File extensions to index; also accepts exact filenames (`Dockerfile`, `Makefile`, `Tiltfile`) |
| `documents.exclude_patterns` | `[]` (empty) | Glob patterns for files/dirs to skip during indexing |
| `documents.chunking.chunk_size` | 1000 | Max characters per chunk |
| `documents.chunking.chunk_overlap` | 200 | Characters shared between consecutive chunks |

**Chunking guidelines**: Short notes → 500/100. General use → 1000/200. Long technical docs → 1500/300.

For `.md` files, chunking splits at `##` and `###` header boundaries first. Sections larger than `chunk_size` are sub-chunked with overlap. Non-markdown files use fixed-size chunking.

#### Models

| Field | Default | Description |
|-------|---------|-------------|
| `models.embedding.model` | `BAAI/bge-small-en-v1.5` | Embedding model (ONNX, runs locally) |
| `models.embedding.dimensions` | 384 | Vector dimensions (must match model) |
| `models.embedding.gpu` | false | Enable CUDA GPU acceleration. See [GPU Acceleration](#gpu-acceleration) for full setup |
| `models.reranker.enabled` | true | Enable cross-encoder reranking |
| `models.reranker.model` | `Xenova/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `models.reranker.top_k_multiplier` | 3 | Fetch N*multiplier candidates for reranking |

If the reranker model is not available locally and the machine cannot download it, search now falls back to the RRF order from hybrid semantic+BM25 retrieval. This keeps `search_knowledge` available offline, but result ordering may be less precise for ambiguous queries until the reranker model is cached.

**Embedding model options** (fastest → most accurate):
- `BAAI/bge-small-en-v1.5` — 384D, ~33MB (default)
- `BAAI/bge-base-en-v1.5` — 768D, ~130MB
- `BAAI/bge-large-en-v1.5` — 1024D, ~335MB
- `intfloat/multilingual-e5-small` — 384D, 100+ languages

> **Warning**: Changing the embedding model after indexing requires `reindex_documents(full_rebuild=True)`.

#### Search

| Field | Default | Description |
|-------|---------|-------------|
| `search.default_results` | 5 | Results returned when no limit specified |
| `search.max_results` | 20 | Hard cap even if client requests more |
| `search.collection_name` | `knowledge_base` | ChromaDB collection — change for separate KBs |

#### Categories

Map folder paths to category names. Documents in matching folders get auto-tagged, enabling filtered searches.

```yaml
category_mappings:
  "security/redteam": "redteam"
  "security": "security"
```

Set `category_mappings: {}` to disable — documents are still searchable, just without category filters.

#### Keyword Routing

Route queries to categories based on keywords. When a query contains listed keywords, results from that category are prioritized (not filtered — other categories still appear, ranked lower).

```yaml
keyword_routes:
  redteam:
    - pentest
    - exploit
    - sqli
```

Single-word keywords use regex word boundaries (`\b`) — "api" won't match "RAPID". Multi-word keywords use substring matching.

Set `keyword_routes: {}` for pure semantic search.

#### Query Expansion

Expand search terms with synonyms before BM25 search. Supports single tokens, bigrams, and full query matches.

```yaml
query_expansions:
  sqli:
    - sql injection
    - sqli
  k8s:
    - kubernetes
    - k8s
```

Set `query_expansions: {}` for no expansion.

`query_expansions` is directional: only the key on the left triggers the terms on the right. If you need mutual expansion without duplicating entries, use `query_expansion_groups`.

```yaml
query_expansion_groups:
  - ["triple barrier", "tb", "trip_barr"]
  - ["profit factor", "pf"]
```

Each group is interpreted symmetrically, so every term expands to the rest of the group. The final internal expansion table is built by merging both sources:

1. `query_expansions` entries are loaded as-is.
2. `query_expansion_groups` adds reciprocal links for every term in each group.
3. Overlaps are merged by union with duplicate terms removed.

This keeps backward compatibility while allowing concise synonym groups.

### Hybrid Search Tuning

| hybrid_alpha | Behavior | Best For |
|--------------|----------|----------|
| 0.0 | Pure BM25 keyword | Exact terms, CVEs, tool names |
| 0.3 | Keyword-heavy **(default)** | Technical queries with specific terms |
| 0.5 | Balanced | General queries |
| 0.7 | Semantic-heavy | Conceptual queries, related topics |
| 1.0 | Pure semantic | "How to..." questions, abstract concepts |

#### `hybrid_alpha` — recipe by intent

Ready-to-copy examples showing when each value fits best. Pass via the `search_knowledge` MCP tool call:

```python
# α = 0.0 → pure keyword (BM25). Best when the query IS the answer (exact
# terminology, tool name, CVE, MITRE ID). Semantic drift is a bug here.
search_knowledge(query="gtfobins suid", hybrid_alpha=0.0)

# α = 0.3 (default) → keyword-heavy. Technical query with specific terms
# where a synonymous document also matters, but the exact terms rank first.
search_knowledge(query="SQL injection techniques", hybrid_alpha=0.3)

# α = 0.5 → balanced. General query where you want both exact matches and
# semantically related material (blog posts, tutorials, adjacent docs).
search_knowledge(query="how to escalate privileges", hybrid_alpha=0.5)

# α = 0.7 → semantic-heavy. Conceptual / strategic query, exact wording of
# the query is unlikely to be in the docs verbatim.
search_knowledge(query="lateral movement strategies", hybrid_alpha=0.7)

# α = 1.0 → pure semantic. Free-form question, no expected keyword overlap.
# Use when BM25 would obviously miss the intent.
search_knowledge(query="mimikatz", hybrid_alpha=1.0)
```

**Rule of thumb:** start at `0.3` (the default). Slide **down** toward `0.0` when the corpus tokenizes cleanly on your query terms (identifiers, codes, filenames). Slide **up** toward `1.0` when queries are phrased as questions or use synonyms not present in the corpus verbatim.

### Search Method (v4.8.2+)

Two paths ship with v4.8.2 — the default `hybrid` path (BM25 + semantic + RRF
+ optional rerank) that has been the sole path since v4.0, and an opt-in
`fts5` lexical fast-path optimised for exact identifier queries (CVEs, MITRE
ATT&CK codes, CWEs, file hashes, bug-bounty IDs). Enable via
`search.lexical_fast_path.enabled: true` in `config.yaml`.

| Method                  | Path              | Latency (typical)     | Best For                                     |
|-------------------------|-------------------|-----------------------|----------------------------------------------|
| `auto` **(default)**    | Router decides    | Router adds ~0.1ms    | Mixed workloads — safe default               |
| `hybrid`                | BM25 + semantic   | 50-150ms (with rerank)| Prose queries, "how does X work", exploration|
| `fts5`                  | SQLite FTS5 only  | <10ms cold / <2ms hot | CVE / MITRE / CWE / hash lookups             |

The `search_knowledge` MCP tool exposes `search_method: Literal["auto",
"hybrid", "fts5"] = "auto"` (ADR-006, additive on the LEI 1 contract). The
default `enabled: false` preserves v4.8.1 behaviour byte-for-byte; the flip
to `enabled: true` by default is reserved for v4.9.0 pending the CI
perf-gate adjudication documented in ADR-004 and ADR-009. Full user guide
in [`docs/features/fts5_fast_path.md`](docs/features/fts5_fast_path.md).

---
