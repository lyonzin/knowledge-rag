# Architecture

> Detailed architecture of **knowledge-rag** — system overview, query processing flow, document ingestion pipeline, and the hybrid_alpha parameter effect.

**Related docs:**
- [Configuration reference →](CONFIGURATION.md)
- [API reference →](API.md)
- [Installation guide →](INSTALLATION.md)

**Diagrams inside:**
- [System Overview](#system-overview)
- [Query Processing Flow](#query-processing-flow)
- [Document Ingestion Flow](#document-ingestion-flow)
- [hybrid_alpha Parameter Effect](#hybrid_alpha-parameter-effect)

---

### System Overview

```mermaid
flowchart TB
    subgraph MCP["MCP SERVER (FastMCP)"]
        direction TB
        TOOLS["13 MCP Tools (frozen, LEI 1)<br/>search_knowledge | get_document | search_similar<br/>add_document | add_from_url | update_document | remove_document<br/>reindex_documents | get_reindex_status<br/>list_categories | list_documents | get_index_stats | evaluate_retrieval"]
    end

    subgraph SEARCH["SEARCH DISPATCH (v4.8.2+)"]
        direction TB
        SM["search_method dispatch<br/>auto | hybrid | fts5"]
        QROUTER["QueryRouter (regex)<br/>lexical vs semantic<br/>(ADR-002)"]
        FTS5PATH["FTS5 Fast-Path<br/>(exact identifiers <10ms)"]
        HYBRIDPATH["Hybrid Pipeline<br/>(conceptual queries)"]

        SM --> QROUTER
        QROUTER -->|lexical| FTS5PATH
        QROUTER -->|semantic| HYBRIDPATH
        FTS5PATH -.->|not_ready / low_hits<br/>fallback| HYBRIDPATH
    end

    subgraph HYBRID["HYBRID PIPELINE"]
        direction LR
        KWROUTER["Keyword Router<br/>(word boundaries → category filter)"]
        SEMANTIC["Semantic Search<br/>(ChromaDB)"]
        BM25["BM25 Keyword<br/>(inverted-index + expansion)"]
        RRF["Reciprocal Rank<br/>Fusion (RRF)"]
        RERANK["Cross-Encoder<br/>Reranker"]

        KWROUTER --> SEMANTIC
        KWROUTER --> BM25
        SEMANTIC --> RRF
        BM25 --> RRF
        RRF --> RERANK
    end

    subgraph STORAGE["STORAGE LAYER"]
        direction LR
        CHROMA[("ChromaDB<br/>data/chroma_db/<br/>(vector store)")]
        FTS5DB[("SQLite FTS5<br/>data/fts5_index.db<br/>(lexical index, ADR-001)")]
        COLLECTIONS["8 Categories<br/>redteam | blueteam | ctf<br/>security | logscale<br/>development | aar | general"]
        CHROMA --- COLLECTIONS
        FTS5DB --- COLLECTIONS
    end

    subgraph EMBED["EMBEDDINGS (In-Process)"]
        FASTEMBED["FastEmbed ONNX<br/>BAAI/bge-small-en-v1.5<br/>(384D, CPU or GPU)"]
        CROSSENC["Cross-Encoder<br/>ms-marco-MiniLM-L-6-v2"]
        FASTEMBED --- CROSSENC
    end

    subgraph INGEST["DOCUMENT INGESTION"]
        PARSERS["35 Parsers<br/>MD | PDF | TXT | PY | C | H | CPP | JS | JSX | TS | TSX | JSON | XML | CSV<br/>DOCX | XLSX | PPTX | IPYNB | MQH | MQ4<br/>Go | Rust | Kotlin | YAML | HuJSON | CUE | Proto | Rego | SQL | Shell | jq | Dockerfile | Makefile | Tiltfile"]
        CHUNKER["Chunking<br/>MD: section-aware<br/>Other: 1000 chars + 200 overlap"]
        FTS5SYNC["FTS5 CRUD sync<br/>(v4.8.2 task 05)"]
        PARSERS --> CHUNKER --> FTS5SYNC
    end

    CLAUDE["Claude Code"] --> MCP
    MCP --> SEARCH
    HYBRIDPATH --> HYBRID
    FTS5PATH --> FTS5DB
    HYBRID --> STORAGE
    STORAGE --> EMBED
    INGEST --> EMBED
    EMBED --> STORAGE
    FTS5SYNC --> FTS5DB
```

### Query Processing Flow

```mermaid
flowchart TB
    QUERY["User Query<br/>'mimikatz credential dump' | 'CVE-2021-4034'"] --> METHOD

    subgraph DISPATCH["Dispatch (v4.8.2+, ADR-006)"]
        METHOD{"search_method<br/>auto | hybrid | fts5"}
        AUTOROUTE["QueryRouter.classify()<br/>regex-based lexical detection"]
        FASTPATH["FTS5 Fast-Path<br/>SQLite MATCH<br/>(&lt;10ms cold, &lt;2ms hot)"]
        NOTREADY{"FTS5 ready<br/>AND hits >= min_hits?"}

        METHOD -->|auto| AUTOROUTE
        METHOD -->|fts5| FASTPATH
        METHOD -->|hybrid| EXPAND
        AUTOROUTE -->|lexical| FASTPATH
        AUTOROUTE -->|semantic| EXPAND
        FASTPATH --> NOTREADY
        NOTREADY -->|no, fallback| EXPAND
    end

    subgraph EXPANSION["Query Expansion (hybrid path)"]
        EXPAND["Synonym Expansion<br/>mimikatz -> mimikatz, sekurlsa, logonpasswords"]
    end

    EXPAND --> KWROUTER

    subgraph ROUTING["Keyword Routing (category filter)"]
        KWROUTER["Keyword Router<br/>(word boundaries)"]
        MATCH{"Word Boundary<br/>Match?"}
        CATEGORY["Filter: redteam"]
        NOFILTER["No Filter"]

        KWROUTER --> MATCH
        MATCH -->|Yes| CATEGORY
        MATCH -->|No| NOFILTER
    end

    subgraph HYBRID["Hybrid Search"]
        direction LR
        SEMANTIC["Semantic Search<br/>(ChromaDB embeddings)<br/>Conceptual similarity"]
        BM25["BM25 Inverted-Index<br/>(posting lists + numpy top-k)<br/>Exact term matching"]
    end

    subgraph FUSION["Result Fusion + Reranking"]
        RRF["Reciprocal Rank Fusion<br/>score = alpha * 1/(k+rank_sem)<br/>+ (1-alpha) * 1/(k+rank_bm25)"]
        RERANK["Cross-Encoder Reranker<br/>Re-scores top 3x candidates<br/>query+doc pair scoring"]
        SORT["Sort by Reranker Score<br/>Normalize to 0-1"]
        ADJ["Adjacent Chunk Expansion<br/>(batch fetch ±1 chunk)"]

        RRF --> RERANK --> SORT --> ADJ
    end

    subgraph OUTPUT["Output Processing"]
        MINSCORE["min_score Filter<br/>(discard below threshold)"]
        SNIPPET["snippet_mode Truncation<br/>(~500 chars at natural break)"]

        MINSCORE --> SNIPPET
    end

    CATEGORY --> HYBRID
    NOFILTER --> HYBRID
    SEMANTIC --> RRF
    BM25 --> RRF

    ADJ --> MINSCORE
    NOTREADY -->|yes, hit| MINSCORE
    SNIPPET --> RESULTS["Results<br/>search_method: fts5 | hybrid | semantic | keyword<br/>routed_by: fts5_router | none<br/>score + filtered_by_score + content_length"]
```

### Document Ingestion Flow

```mermaid
flowchart LR
    subgraph INPUT["Input"]
        FILES["documents/<br/>├── security/<br/>│   ├── redteam/<br/>│   ├── blueteam/<br/>│   └── ctf/<br/>├── aar/<br/>├── logscale/<br/>├── development/<br/>└── general/"]
    end

    subgraph PARSE["Parse (35 formats)"]
        MD["Markdown"]
        PDF["PDF<br/>(PyMuPDF)"]
        OFFICE["DOCX | XLSX<br/>PPTX | CSV"]
        CODE["PY | C | H | CPP | JS | JSX<br/>TS | TSX | JSON | XML | IPYNB"]
    end

    subgraph CHUNK["Chunk"]
        MDSPLIT["MD: Section-Aware<br/>Split at ## headers"]
        TXTSPLIT["Other: Fixed-Size<br/>1000 chars + 200 overlap"]
        DEDUP["SHA256 Dedup<br/>Skip duplicate content"]
    end

    subgraph EMBED["Embed"]
        FASTEMBED["FastEmbed ONNX<br/>bge-small-en-v1.5<br/>(384D, CPU or GPU)"]
    end

    subgraph DISPATCH["Write Dispatch (v4.8.3, #161)"]
        WCOL{"_write_collection<br/>routes writes"}
        STAGING["Staging Collection<br/>(nuclear_rebuild only,<br/>keeps prod serving queries)"]
        PROD["Production Collection<br/>(default)"]
        WCOL -->|_staging_target set| STAGING
        WCOL -->|default| PROD
    end

    subgraph STORE["Store"]
        CHROMADB[("ChromaDB<br/>data/chroma_db/")]
        BM25IDX["BM25 Index<br/>(in-memory,<br/>rebuilt on write)"]
        FTS5DB[("SQLite FTS5<br/>data/fts5_index.db<br/>(CRUD-synced, ADR-008)")]
    end

    FILES --> MD & PDF & OFFICE & CODE
    MD --> MDSPLIT
    PDF & OFFICE & CODE --> TXTSPLIT
    MDSPLIT --> DEDUP
    TXTSPLIT --> DEDUP
    DEDUP --> EMBED
    EMBED --> DISPATCH
    STAGING --> CHROMADB
    PROD --> CHROMADB
    PROD --> BM25IDX
    PROD --> FTS5DB
```

### hybrid_alpha Parameter Effect

`hybrid_alpha` weights RRF fusion between semantic and BM25 rankings on the **hybrid pipeline only**. When `search_method="auto"` and the QueryRouter classifies the query as lexical (e.g. `CVE-2021-4034`, `MDR-AD002`, `T1078.001`, file hashes), the FTS5 fast-path fires and `hybrid_alpha` is not consulted — FTS5 uses SQLite's native `bm25()` scoring exclusively. Pass `search_method="hybrid"` to force RRF fusion + rerank on every query if you want deterministic hybrid semantics regardless of the query shape.

```mermaid
flowchart LR
    subgraph ALPHA["hybrid_alpha values (hybrid path only)"]
        A0["0.0<br/>Pure BM25<br/>Instant"]
        A3["0.3 (default)<br/>Keyword-heavy<br/>Fast"]
        A5["0.5<br/>Balanced"]
        A7["0.7<br/>Semantic-heavy"]
        A10["1.0<br/>Pure Semantic"]
    end

    subgraph USE["Best For"]
        U0["CVEs, tool names<br/>exact matches<br/>(consider FTS5 fast-path)"]
        U3["Technical queries<br/>specific terms"]
        U5["General queries"]
        U7["Conceptual queries<br/>related topics"]
        U10["'How to...' questions<br/>conceptual search"]
    end

    A0 --- U0
    A3 --- U3
    A5 --- U5
    A7 --- U7
    A10 --- U10
```

---

