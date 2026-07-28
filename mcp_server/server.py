"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                         KNOWLEDGE RAG MCP SERVER                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

MCP Server with hybrid search + cross-encoder reranking for local document retrieval.
Uses ChromaDB for vector storage, FastEmbed for ONNX embeddings, BM25 for keywords.

Features:
    - Hybrid search (semantic + BM25 keyword) with RRF fusion
    - Cross-encoder reranking for precision boost
    - Markdown-aware chunking (splits by ## sections)
    - Multilingual query stopword filtering before BM25 + query expansion
    - Query expansion for security term synonyms
    - Incremental indexing (only re-indexes changed files)
    - Query caching with TTL for instant repeat queries
    - Chunk deduplication via content hashing
    - Truncation-aware tool output (never silently drops results)
    - CRUD operations via MCP tools (add, update, remove docs)

Autor:   Lyon (Ailton Rocha)

Version: single-sourced from ``mcp_server.__version__`` — deliberately NOT
restated here. This header once advertised a release three minor versions
behind what actually shipped. A docstring cannot interpolate, so the only
durable fix is to stop duplicating the value: read it from
``mcp_server.__version__`` or run ``knowledge-rag --version``.
``scripts/check_version_sync.py`` fails the build if a version literal
reappears in this header.

Refactor note (A2.1, v4.6.0):
    This module used to be a 3660-LOC monolith. The A2.1 refactor split the
    implementations across ``mcp_server.retrieval``, ``mcp_server.indexing``,
    ``mcp_server.storage`` and ``mcp_server.mcp_tools`` without any behaviour
    change. The re-exports below preserve every historical import path
    (``from mcp_server.server import X``) that tests and third-party callers
    relied on, including patch targets like ``mcp_server.server.TextEmbedding``
    and ``mcp_server.server.FastEmbedEmbeddings``.
"""

import os
import sys
import threading
from typing import Optional

# FastEmbed re-exports — kept at module scope so unit tests that call
# ``patch("mcp_server.server.TextEmbedding")`` / ``TextCrossEncoder`` remain
# effective. The extracted modules (retrieval/embeddings.py, retrieval/rerank.py)
# look these classes up through ``mcp_server.server`` at call time so patches
# propagate through the split.
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from .config import config
from .indexing.bm25_index import BM25Index
from .indexing.watcher import DocumentWatcher
from .ingestion import Document, DocumentParser  # noqa: F401 — public re-export
from .mcp_tools.instance import mcp
from .mcp_tools.tools import (
    add_document,  # noqa: F401 — public re-export
    add_from_url,  # noqa: F401 — public re-export
    evaluate_retrieval,  # noqa: F401 — public re-export
    get_document,  # noqa: F401 — public re-export
    get_index_stats,  # noqa: F401 — public re-export
    get_reindex_status,  # noqa: F401 — public re-export
    list_categories,  # noqa: F401 — public re-export
    list_dashboard,  # noqa: F401 — public re-export (M4.4)
    list_documents,  # noqa: F401 — public re-export
    reflect,  # noqa: F401 — public re-export (M4.2)
    reindex_documents,  # noqa: F401 — public re-export
    remove_document,  # noqa: F401 — public re-export
    save_result,  # noqa: F401 — public re-export (M4.2)
    search_global,  # noqa: F401 — public re-export (M4.3)
    search_knowledge,  # noqa: F401 — public re-export
    search_similar,  # noqa: F401 — public re-export
    update_document,  # noqa: F401 — public re-export
)
from .retrieval.embeddings import (
    EmbeddingError,
    EmbeddingModelLoadError,
    FastEmbedEmbeddings,
    GPUStatus,
)
from .retrieval.impls import _list_impl, _search_impl, _stats_impl  # noqa: F401 — CLI + test re-exports
from .retrieval.orchestrator import KnowledgeOrchestrator, _metadata_path_score  # noqa: F401 — test re-export
from .retrieval.query_cache import QueryCache
from .retrieval.rerank import CrossEncoderReranker
from .retrieval.truncation import (  # noqa: F401 — historical test re-exports
    DOCUMENT_CHAR_BUDGET,
    RESULT_CHAR_BUDGET,
    _make_snippet,
    _result_char_cost,
    _truncate_document_safe,
    _truncate_results_safe,
    _truncation_note,
)
from .security import BearerAuthMiddleware
from .stopwords import filter_query_stopwords  # noqa: F401 — public re-export

__all__ = [
    # Errors
    "EmbeddingError",
    "EmbeddingModelLoadError",
    # Retrieval primitives
    "BM25Index",
    "CrossEncoderReranker",
    "FastEmbedEmbeddings",
    "GPUStatus",
    "KnowledgeOrchestrator",
    "QueryCache",
    "TextCrossEncoder",
    "TextEmbedding",
    # Filesystem
    "DocumentWatcher",
    # MCP instance + tools
    "add_document",
    "add_from_url",
    "evaluate_retrieval",
    "get_document",
    "get_index_stats",
    "get_orchestrator",
    "get_reindex_status",
    "list_categories",
    "list_dashboard",
    "list_documents",
    "main",
    "mcp",
    "reflect",
    "reindex_documents",
    "remove_document",
    "save_result",
    "search_global",
    "search_knowledge",
    "search_similar",
    "update_document",
]


# =============================================================================
# ORCHESTRATOR SINGLETON
# =============================================================================

_orchestrator: Optional[KnowledgeOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_orchestrator() -> KnowledgeOrchestrator:
    """Get or create the orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:
                _orchestrator = KnowledgeOrchestrator()
    return _orchestrator


# =============================================================================
# Entry point
# =============================================================================


def _handle_init():
    """Export config template and presets to current directory."""
    import shutil
    from pathlib import Path

    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        print("[ERROR] Bundled data not found. If installed from git, use presets/ directly.")
        return

    cwd = Path.cwd()

    try:
        # Copy config.example.yaml
        src = data_dir / "config.example.yaml"
        if src.exists():
            dst = cwd / "config.example.yaml"
            shutil.copy2(src, dst)
            print(f"[OK] {dst}")

        # Copy presets
        presets_dir = cwd / "presets"
        presets_dir.mkdir(exist_ok=True)
        for f in data_dir.glob("*.yaml"):
            if f.name == "config.example.yaml":
                continue
            dst = presets_dir / f.name
            shutil.copy2(f, dst)
            print(f"[OK] {dst}")

        # Create documents dir
        docs_dir = cwd / "documents"
        docs_dir.mkdir(exist_ok=True)
        print(f"[OK] {docs_dir}/")

        print("\nDone. Quick start:")
        print("  cp presets/general.yaml config.yaml     # or cybersecurity, developer, research")
        print("  # Add your documents to documents/")
        print("  # Restart Claude Code")
    except PermissionError:
        print("[ERROR] Permission denied. Run from a writable directory.")
    except OSError as e:
        print(f"[ERROR] Failed to write files: {e}")


def _run_transport(transport: str) -> None:
    """Start the MCP server, enforcing bearer auth on HTTP transports.

    ``config.auth_bearer_token`` was declared since v4.0.0 but never checked —
    an operator who set it believed the port was protected while every tool
    stayed reachable unauthenticated (CWE-287). When a token is configured, the
    Starlette app is wrapped in :class:`~mcp_server.security.BearerAuthMiddleware`
    and served directly, so unauthenticated callers get ``401`` before reaching
    the MCP dispatcher.

    Behaviour is unchanged when no token is set or when the transport is
    ``stdio`` (a local pipe carries no HTTP headers): the stock
    ``FastMCP.run()`` path is used verbatim.

    Args:
        transport: One of ``stdio``, ``sse`` or ``streamable-http``.
    """
    token = (config.auth_bearer_token or "").strip()

    if transport == "stdio" or not token:
        if transport != "stdio":
            print(
                "[WARN] Bearer auth disabled — server.auth.bearer_token is empty. "
                f"Every MCP tool on {config.server_host}:{config.server_port} is reachable "
                "without credentials.",
                file=sys.stderr,
            )
        mcp.run(transport=transport)
        return

    import uvicorn

    if transport == "sse":
        app = mcp.sse_app()
    elif transport == "streamable-http":
        app = mcp.streamable_http_app()
    else:
        raise ValueError(f"Unknown transport: {transport}")

    print("[SERVER] Bearer auth enabled — Authorization: Bearer <token> required", file=sys.stderr)

    uvicorn.run(
        BearerAuthMiddleware(app, token),
        host=config.server_host,
        port=config.server_port,
        log_level=str(getattr(mcp.settings, "log_level", "INFO")).lower(),
    )


def _run_semantic_cache_maintenance() -> None:
    """Run TTL invalidation + orphan-prompt cleanup on the semantic cache.

    Isolated from :func:`main` so unit tests can exercise the maintenance
    branch without spinning up the whole server. Failures are logged to
    stderr but never raise — a corrupt cache should never block startup
    of the retrieval-only path.

    Behaviour:
        * ``SemanticCache.invalidate_stale`` unlinks entries older than
          ``config.semantic_cache_ttl_days``. Safe no-op on an empty
          cache and on ``ttl_days=0`` (interpreted as "keep forever").
        * ``SemanticCache.cleanup_orphan_prompts`` unlinks the
          ``p<fingerprint>/`` sub-directories whose fingerprint is not
          in :data:`ACTIVE_PROMPT_FINGERPRINTS`. The guard on the set
          being non-empty is critical: no active fingerprints (default
          pre-Fase-3 state) would otherwise delete every prompt dir.
    """
    from pathlib import Path

    from .providers.llm.prompts import ACTIVE_PROMPT_FINGERPRINTS
    from .retrieval.semantic_cache import SemanticCache

    cache_dir_raw = config.semantic_cache_dir
    cache_path = Path(cache_dir_raw)
    if not cache_path.is_absolute():
        cache_path = config.data_dir.parent / cache_dir_raw
    # Fallback: if the relative path escapes data_dir.parent, root it there.
    if not str(cache_path).startswith(str(config.data_dir.parent)):
        cache_path = config.data_dir / "cache" / "semantic"

    ttl_seconds = None if config.semantic_cache_ttl_days == 0 else config.semantic_cache_ttl_days * 24 * 3600

    try:
        cache = SemanticCache(
            cache_dir=cache_path,
            ttl_seconds=ttl_seconds,
            max_entries=config.semantic_cache_max_entries,
        )
        stale_removed = cache.invalidate_stale()
        if stale_removed:
            print(f"[SEMCACHE] {stale_removed} stale entries removed", file=sys.stderr)

        if ACTIVE_PROMPT_FINGERPRINTS:
            orphan_removed = cache.cleanup_orphan_prompts(ACTIVE_PROMPT_FINGERPRINTS)
            if orphan_removed:
                print(f"[SEMCACHE] {orphan_removed} orphan prompt dirs removed", file=sys.stderr)
    except Exception as exc:  # pragma: no cover — defensive
        print(f"[SEMCACHE] maintenance skipped: {exc}", file=sys.stderr)


def main():
    """Run the MCP server"""
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        _handle_init()
        return

    from watchdog.observers import Observer

    from .instance_lock import (
        ALREADY_RUNNING_EXIT_CODE,
        AlreadyRunningError,
        single_instance_lock,
    )
    from .preflight import run_preflight

    try:
        # SSE/HTTP mode: auto-enable single-instance lock (port collision prevention)
        transport = config.transport
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == "--transport" and i < len(sys.argv) - 1:
                transport = sys.argv[i + 1]
            elif arg.startswith("--transport="):
                transport = arg.split("=", 1)[1]
        if transport != "stdio":
            os.environ["KNOWLEDGE_RAG_SINGLE_INSTANCE"] = "1"

        with single_instance_lock():
            run_preflight()

            orchestrator = get_orchestrator()

            # Migration: check dimension mismatch AFTER full init (avoids segfault during __init__)
            orchestrator._needs_rebuild = orchestrator._check_dimension_mismatch()
            if orchestrator._needs_rebuild:
                print("[MIGRATION] Running nuclear rebuild for embedding model change...")
                try:
                    stats = orchestrator.nuclear_rebuild()
                    print(
                        f"[MIGRATION] Rebuild complete: {stats['indexed']} docs, "
                        f"{stats['chunks_added']} chunks in {stats.get('elapsed_seconds', '?')}s"
                    )
                except Exception as e:
                    print(f"[ERROR] Migration failed: {e}")
                    print("[FALLBACK] Attempting regular index instead...")
                    stats = orchestrator.index_all(force=True)
            elif orchestrator.collection.count() == 0:
                print("[INFO] No documents indexed. Running initial indexing...")
                stats = orchestrator.index_all()
                print(f"[INFO] Indexed {stats['indexed']} documents with {stats['chunks_added']} chunks")

            # Start file watcher for auto-reindex on document changes
            if os.environ.get("KNOWLEDGE_RAG_WATCHER_DISABLED", "").strip() == "1":
                print("[WATCHER] Disabled via KNOWLEDGE_RAG_WATCHER_DISABLED=1")
            else:
                try:
                    watcher = DocumentWatcher(get_orchestrator, debounce_seconds=10.0)
                    observer = Observer()
                    observer.schedule(watcher, str(config.documents_dir), recursive=True)
                    observer.daemon = True
                    observer.start()
                    print(f"[WATCHER] Monitoring {config.documents_dir} for changes")
                except Exception as e:
                    print(f"[WARN] Failed to start file watcher: {e}")
                    print("[WARN] Auto-reindexing disabled. Use reindex_documents tool manually.")

            # Semantic cache startup maintenance (A2.7 infra + A3.1 wiring):
            # 1) TTL-based invalidation always runs when the cache is enabled.
            # 2) Orphan-prompt cleanup only runs when at least one Fase 3
            #    feature has registered an active prompt fingerprint. With no
            #    active fingerprints (the pre-Fase-3 default) we would delete
            #    every cached prompt dir on every start — so we guard.
            if config.semantic_cache_enabled:
                _run_semantic_cache_maintenance()

            # Start optional metrics server
            if config.metrics_enabled and config.transport != "stdio":
                from .metrics import start_metrics_server

                start_metrics_server(config.metrics_port)

            # Restore real stdout for MCP JSON-RPC, keep print() going to stderr
            from . import _original_stdout

            sys.stdout = _original_stdout

            # Parse --transport CLI override
            transport = config.transport
            for i, arg in enumerate(sys.argv[1:], 1):
                if arg == "--transport" and i < len(sys.argv) - 1:
                    transport = sys.argv[i + 1]
                elif arg.startswith("--transport="):
                    transport = arg.split("=", 1)[1]

            if transport != "stdio":
                print(
                    f"[SERVER] Starting {transport} server on {config.server_host}:{config.server_port}",
                    file=sys.stderr,
                )

            _run_transport(transport)
    except AlreadyRunningError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(ALREADY_RUNNING_EXIT_CODE) from e


if __name__ == "__main__":
    main()
