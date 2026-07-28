"""
╭─╴ RETRIEVAL PACKAGE ╶──────────────────────────────────────────╮
│                                                                │
│   Hybrid retrieval primitives split out of server.py in the    │
│   A2.1 refactor. Public re-exports keep the historical         │
│   ``mcp_server.server.*`` import surface intact.               │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``
"""

# R5.5 (learned fusion) and semantic_cache ship in a follow-up PR — those
# modules register additional fusion strategies + a query-similarity cache.
# Their absence keeps the pipeline on RRF + no semantic cache (both the
# historical defaults), so nothing here breaks when they are missing.
from .embeddings import (
    EmbeddingError,
    EmbeddingModelLoadError,
    FastEmbedEmbeddings,
    GPUStatus,
)
from .impls import _list_impl, _search_impl, _stats_impl
from .orchestrator import KnowledgeOrchestrator, _metadata_path_score
from .query_cache import QueryCache
from .rerank import CrossEncoderReranker
from .truncation import (
    DOCUMENT_CHAR_BUDGET,
    RESULT_CHAR_BUDGET,
    _make_snippet,
    _result_char_cost,
    _truncate_document_safe,
    _truncate_results_safe,
    _truncation_note,
)

__all__ = [
    "CrossEncoderReranker",
    "DOCUMENT_CHAR_BUDGET",
    "EmbeddingError",
    "EmbeddingModelLoadError",
    "FastEmbedEmbeddings",
    "GPUStatus",
    "KnowledgeOrchestrator",
    "QueryCache",
    "RESULT_CHAR_BUDGET",
    "_list_impl",
    "_make_snippet",
    "_metadata_path_score",
    "_result_char_cost",
    "_search_impl",
    "_stats_impl",
    "_truncate_document_safe",
    "_truncate_results_safe",
    "_truncation_note",
]
