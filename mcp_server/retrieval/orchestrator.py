"""
╭─╴ KNOWLEDGE ORCHESTRATOR ╶─────────────────────────────────────╮
│                                                                │
│   Main orchestrator: hybrid search (semantic + BM25 + RRF +    │
│   cross-encoder rerank), incremental indexing, CRUD tooling.   │
│   Extracted verbatim from server.py in the A2.1 refactor.      │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``

Test-patch compatibility:
    ``KnowledgeOrchestrator.__init__`` resolves ``FastEmbedEmbeddings`` via
    ``mcp_server.server.FastEmbedEmbeddings`` at call time so ``mock_embedding``
    (see ``tests/conftest.py``) keeps working after the module split.
"""

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb

log = logging.getLogger(__name__)

from ..config import config
from ..indexing.bm25_index import BM25Index
from ..ingestion import Document, DocumentParser
from ..providers import (
    get_embedding,
    get_reranker,
    get_vector_store,
    is_embedding_registered,
    is_reranker_registered,
    is_vector_store_registered,
)
from ..security import PathEscapeError, sanitize_external_content, validate_path_within

# ``ChromaVectorStore`` is imported for its side-effect: it lives in
# ``mcp_server.storage.chroma`` and is what the provider registry hands back
# for the "chromadb" vector store name. The bundled ChromaDB flow still uses
# ``chromadb.PersistentClient`` + ``_enable_wal_mode`` directly here so
# ``_safe_get_collection`` can atomically rebuild the client and collection
# on corruption; the wrapper class only comes into play for non-default
# ``config.vector_store`` values.
from ..storage.chroma import ChromaVectorStore, _enable_wal_mode  # noqa: F401 — public re-export
from ..telemetry import get_tracer
from ..telemetry import trace as _traced_span
from .fusion import RetrievedResult, get_strategy
from .mmr import apply_mmr
from .query_cache import QueryCache

# ``CrossEncoderReranker`` is now resolved via :func:`_resolve_reranker_class`
# (late-bind through ``mcp_server.server`` for test compat) or the provider
# registry. The direct import stays as a public re-export so downstream
# imports like ``from mcp_server.retrieval.orchestrator import CrossEncoderReranker``
# — if any — keep working.
from .rerank import CrossEncoderReranker  # noqa: F401 — public re-export


def _resolve_embeddings_class():
    """Late-bind lookup for ``FastEmbedEmbeddings``.

    ``tests/conftest.py`` patches ``mcp_server.server.FastEmbedEmbeddings``
    to avoid downloading the real ONNX model in CI. Routing the lookup
    through the ``mcp_server.server`` module keeps that fixture effective
    after the A2.1 module split.
    """
    try:
        from mcp_server import server as _srv

        return _srv.FastEmbedEmbeddings
    except (ImportError, AttributeError):
        from .embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings


def _resolve_reranker_class():
    """Late-bind lookup for ``CrossEncoderReranker``.

    Mirrors :func:`_resolve_embeddings_class` — tests read the reranker
    class through :mod:`mcp_server.server` (see
    ``tests/test_reranker_fallback.py``), so keep the lookup indirected.
    """
    try:
        from mcp_server import server as _srv

        return _srv.CrossEncoderReranker
    except (ImportError, AttributeError):
        from .rerank import CrossEncoderReranker as _CE

        return _CE


def _maybe_log_query(query_text: str, results: List[Dict[str, Any]]) -> None:
    """Append one line to the M4.4 dashboard query log when opted in.

    Module-level (rather than an orchestrator method) so it works
    transparently against test doubles that duck-type
    :class:`KnowledgeOrchestrator` without inheriting from it — the
    same pattern used by :func:`_apply_work_memory_overlay` above.

    Zero cost when ``config.query_log_enabled`` is False (the default) —
    this function short-circuits before touching the dashboard subpackage,
    so the import cost is only paid the first time an operator turns the
    log on. Best-effort by design: any failure (bad path, permission
    error, full disk) is swallowed and logged at DEBUG so retrieval
    never breaks because a maintenance hook misfired.

    Args:
        query_text: Raw user query as received by :meth:`KnowledgeOrchestrator.query`
            — recorded verbatim, before any A3.2 rewrite / A3.5 self-query
            rewrite. The log is a fidelity record of what the CALLER asked,
            not what the LLM turned it into.
        results: Formatted results list about to be returned. Only the
            unique ``source`` values are extracted so payload sizes stay
            tiny (< 1 KB per query on typical corpora).
    """
    if not getattr(config, "query_log_enabled", False):
        return
    try:
        from ..dashboard.query_log import append_query_log, resolve_query_log_dir

        log_dir = resolve_query_log_dir(config)
        append_query_log(log_dir, query_text, results)
    except Exception as exc:  # pragma: no cover — defensive
        log.debug("dashboard query_log hook failed: %s", exc)


def _apply_work_memory_overlay(results: List[Dict[str, Any]]) -> None:
    """Stamp each result with a ``learning`` field when the overlay
    classifies its ``source`` (M4.2).

    Module-level (rather than an orchestrator method) so it works
    transparently against test doubles that duck-type
    :class:`KnowledgeOrchestrator` without inheriting from it.

    No-op when ``config.work_memory_enabled`` is False or the overlay
    file is missing / empty / malformed. Fails silent by design:
    retrieval never breaks because the overlay is broken. Mutates
    ``results`` in place — results whose ``source`` is not in the
    overlay are left untouched so the output shape stays sparse.

    Args:
        results: List of formatted result dicts from
            :meth:`KnowledgeOrchestrator.query`.
    """
    if not results:
        return
    if not getattr(config, "work_memory_enabled", False):
        return

    # Deferred imports so the opt-in feature never touches the default
    # retrieval path. ``load_overlay`` is failure-tolerant — missing or
    # malformed files yield an empty dict — so no try/except needed.
    from mcp_server.config import BASE_DIR
    from mcp_server.work_memory.reflect import OVERLAY_FILENAME, load_overlay

    overlay = load_overlay(BASE_DIR / OVERLAY_FILENAME)
    if not overlay:
        return

    for result in results:
        source = result.get("source")
        if isinstance(source, str) and source in overlay:
            result["learning"] = overlay[source]


def _metadata_path_score(query: str, metadata: Dict[str, Any]) -> float:
    """Return a small generic score boost when query terms match path metadata."""
    query_terms = re.findall(r"[a-z0-9][-a-z0-9]*[a-z0-9]|[a-z0-9]", query.lower())
    if not query_terms:
        return 0.0

    source = str(metadata.get("source", ""))
    filename = str(metadata.get("filename", ""))
    path_text = f"{source} {filename}".lower()
    path_tokens = set(re.findall(r"[a-z0-9][-a-z0-9]*[a-z0-9]|[a-z0-9]", path_text))
    if not path_tokens:
        return 0.0

    score = 0.0
    for term in query_terms:
        if term in path_tokens:
            score += 0.0006
        elif term in path_text:
            score += 0.0003

    query_phrase = query.strip().lower()
    if query_phrase and query_phrase in path_text:
        score += 0.0012

    return min(score, 0.003)


# ╭─╴ Multi-query RRF (A3.4) ╶─────────────────────────────────╮
# │   Second-level rank fusion for the multi-query fan-out.    │
# │   Distinct from the semantic+BM25 fusion in fusion.py —    │
# │   this one fuses N per-variation formatted result lists    │
# │   AFTER each list has been reranked, so it only cares      │
# │   about ranks, not raw scores.                             │
# ╰────────────────────────────────────────────────────────────╯

_MULTI_QUERY_MISSING_RANK = 1000
"""Fallback rank for a doc absent from a variation's top-K.

Matches the historical fallback used by :class:`RRFusion` in fusion.py
so behaviour is consistent across the two RRF passes."""


def _fuse_multi_query_results(
    per_variation_results: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fuse per-variation formatted result lists via multi-query RRF.

    Multi-query RRF formula::

        score(d) = sum over variations v of 1 / (k + rank_v(d))

    where ``rank_v(d)`` is d's 1-based rank in variation v's result
    list, or :data:`_MULTI_QUERY_MISSING_RANK` when d is absent. The
    fallback rank (rather than 0 / ``+inf``) is inherited from
    :class:`RRFusion` in fusion.py so the two RRF passes behave
    consistently — a single-variation call to :func:`_fuse_multi_query_results`
    is byte-close to the original ranking (only the metadata_path_score
    stacking is not re-applied because the per-variation lists already
    include it).

    Documents are keyed by ``(source, chunk_index)`` because the
    formatted result dicts do not carry the internal ``chunk_id``. Two
    variations returning the same physical chunk under different orders
    thus contribute their ranks to the SAME fused doc.

    The first variation's dict wins on tie for metadata / content
    (dedup is order-preserving). This matters because
    :func:`_maybe_multi_query_variations` guarantees position 0 is the
    original query, so the original's content stays authoritative for
    display even when a paraphrased variation surfaces the same chunk.

    Args:
        per_variation_results: List of per-variation result lists. Empty
            variations (LLM returned nothing) contribute no ranks and
            do not affect the fused ranking.
        k: RRF constant. Default 60 matches Cormack et al. 2009 and the
            in-tree :class:`RRFusion` default. Tests can pin a different
            k to isolate ranking behaviour from the constant.
        top_k: Optional cap on the returned list length. ``None``
            (default) returns every fused doc.

    Returns:
        Merged list of formatted result dicts sorted descending by
        multi-query RRF score. Each dict has an additional
        ``multi_query_rrf_score`` field so callers can debug the
        ranking without recomputing it.

    Example:
        >>> a = [{"source": "a.md", "chunk_index": 0, "content": "..."}]
        >>> b = [{"source": "a.md", "chunk_index": 0, "content": "..."}]
        >>> fused = _fuse_multi_query_results([a, b])
        >>> len(fused)
        1
    """
    if not per_variation_results:
        return []

    # Filter out empty variation lists — they contribute nothing to the
    # fused ranking and let us skip the inner loop entirely for them.
    non_empty = [r for r in per_variation_results if r]
    if not non_empty:
        return []
    if len(non_empty) == 1:
        # Degenerate case: only one variation returned results. There is
        # nothing to fuse, so we short-circuit to that variation's own
        # ranking. Still annotate ``multi_query_rrf_score`` so the caller
        # can distinguish "fan-out ran with 1 non-empty branch" from
        # "fan-out never ran".
        results = list(non_empty[0])
        for rank, entry in enumerate(results, start=1):
            entry = dict(entry)  # do not mutate the caller's dicts
            entry["multi_query_rrf_score"] = round(1.0 / (k + rank), 6)
            results[rank - 1] = entry
        return results if top_k is None else results[:top_k]

    # Score accumulator + first-seen data dict per key.
    scores: Dict[Tuple[str, int], float] = {}
    data_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    seen_order: List[Tuple[str, int]] = []

    for variation_results in per_variation_results:
        if not variation_results:
            continue
        for rank, entry in enumerate(variation_results, start=1):
            source = str(entry.get("source", ""))
            chunk_index = entry.get("chunk_index")
            try:
                chunk_index_int = int(chunk_index) if chunk_index is not None else -1
            except (TypeError, ValueError):
                chunk_index_int = -1
            key = (source, chunk_index_int)

            if key not in data_by_key:
                # First-seen wins for the visible payload: preserves the
                # original variation's content / metadata / expansions.
                data_by_key[key] = dict(entry)
                seen_order.append(key)

            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    # Docs that never surfaced in ANY variation are trivially absent —
    # they simply have no entry in ``scores``. Docs that appeared in only
    # some variations still get their RRF contribution from the ones
    # they DID surface in, with no missing-rank penalty. This diverges
    # from the semantic+BM25 RRFusion in fusion.py (which penalises
    # missing branches with ``MISSING_RANK = 1000``) because there we
    # have exactly two branches and the docs' presence pattern is
    # informative; here we may have N variations and penalising every
    # absence would over-weight docs that happen to appear in every
    # variation's noisy tail.

    # Sort by fused score descending, break ties by first-seen order
    # (a doc surfaced by an earlier variation ranks higher than one
    # surfaced by a later variation when their fused scores tie).
    order_index = {key: i for i, key in enumerate(seen_order)}
    sorted_keys = sorted(scores.keys(), key=lambda k_: (-scores[k_], order_index.get(k_, 10**9)))

    fused_results: List[Dict[str, Any]] = []
    for key in sorted_keys:
        entry = data_by_key[key]
        entry["multi_query_rrf_score"] = round(scores[key], 6)
        fused_results.append(entry)

    if top_k is not None:
        fused_results = fused_results[:top_k]
    return fused_results


class KnowledgeOrchestrator:
    """Main orchestrator for knowledge retrieval with semantic search + keyword routing"""

    def __init__(self):
        self.parser = DocumentParser()
        # Provider dispatch (A2.2). Defaults preserve the historical
        # FastEmbed/ChromaDB/cross-encoder path *and* the late-bind hooks
        # (``mcp_server.server.FastEmbedEmbeddings`` etc.) that unit tests
        # patch. Non-default providers go through the registry.
        embedding_provider_name = config.embedding_provider
        if embedding_provider_name == "fastembed":
            FastEmbedEmbeddings = _resolve_embeddings_class()
            self.embed_fn = FastEmbedEmbeddings()
        else:
            if not is_embedding_registered(embedding_provider_name):
                raise ValueError(
                    f"Unknown embedding provider {embedding_provider_name!r}. "
                    "Install the plugin or set models.embedding.provider back to 'fastembed'."
                )
            self.embed_fn = get_embedding(embedding_provider_name)

        # Vector store dispatch. ChromaDB stays inlined so
        # ``_safe_get_collection`` can atomically rebuild ``self.chroma_client``
        # + ``self.collection`` on corruption. Third-party stores go through
        # the registry and expose the raw handles via ``.client`` / ``.collection``
        # when possible so orchestrator internals keep working.
        vector_store_name = config.vector_store
        if vector_store_name == "chromadb":
            self.chroma_client = chromadb.PersistentClient(path=str(config.chroma_dir))
            if config.transport != "stdio":
                _enable_wal_mode(config.chroma_dir)
            self.collection = self._safe_get_collection()
            self.vector_store: Any = None  # set once ChromaVectorStore wrap is stable
        else:
            if not is_vector_store_registered(vector_store_name):
                raise ValueError(
                    f"Unknown vector store {vector_store_name!r}. "
                    "Install the plugin or set models.vector_store.provider back to 'chromadb'."
                )
            self.vector_store = get_vector_store(
                vector_store_name,
                path=config.chroma_dir,
                collection_name=config.collection_name,
                embedding_function=self.embed_fn,
                enable_wal=config.transport != "stdio",
            )
            # Third-party stores that adapt to ChromaDB's client/collection API
            # can expose them here. Everyone else keeps these as ``None`` and
            # relies on the Protocol methods; callers of ``self.collection``
            # inside this class assume ChromaDB semantics and will raise if a
            # non-ChromaDB store is selected. That is intentional — provider
            # extensibility is opt-in and requires code changes here.
            self.chroma_client = getattr(self.vector_store, "client", None)
            self.collection = getattr(self.vector_store, "collection", None)

        # BM25 index for hybrid search
        self.bm25_index = BM25Index()
        self._bm25_initialized = False

        # Cross-encoder reranker (lazy-loaded on first query)
        reranker_name = config.reranker_provider
        if reranker_name == "cross_encoder":
            CrossEncoderRerankerCls = _resolve_reranker_class()
            self.reranker = CrossEncoderRerankerCls()
        else:
            if not is_reranker_registered(reranker_name):
                raise ValueError(
                    f"Unknown reranker {reranker_name!r}. "
                    "Install the plugin or set models.reranker.provider back to 'cross_encoder'."
                )
            self.reranker = get_reranker(reranker_name)

        # Query cache (LRU with TTL)
        self.query_cache = QueryCache(max_size=100, ttl_seconds=300)

        # Index metadata cache
        self._metadata_file = config.data_dir / "index_metadata.json"
        self._indexed_docs: Dict[str, Dict] = self._load_metadata()

        # Reverse lookup: resolved source path → doc_id (for O(1) adjacent chunk expansion)
        self._source_to_docid: Dict[str, str] = self._build_source_lookup()

        # Migration: deferred — checked in main() after full init
        self._needs_rebuild = False

        # Background reindex progress (polled via get_index_stats)
        self._reindex_progress: Dict[str, Any] = {"active": False}

        # Scratch attribute for MMR — ``query()`` writes the current query
        # text here so ``_apply_mmr`` can embed it without changing the
        # method signature (monkey-patched by tests/test_pr98_regression.py).
        self._current_query_text: Optional[str] = None

    def _safe_get_collection(self):
        """
        Get or create ChromaDB collection with auto-recovery.

        Handles:
        - Corrupted SQLite DB (segfault/crash during previous indexing)
        - Embedding function conflict (collection created with different embed fn)
        - Any other ChromaDB initialization error

        Recovery: deletes corrupted data and starts fresh.
        """
        import shutil

        try:
            return self.chroma_client.get_or_create_collection(
                name=config.collection_name,
                embedding_function=self.embed_fn,
                metadata={"description": "Knowledge base for RAG"},
            )
        except (ValueError, Exception) as e:
            error_msg = str(e).lower()
            if "conflict" in error_msg or "embedding" in error_msg:
                log.warning("[RECOVERY] Embedding function conflict detected: %s", e)
                log.warning("[RECOVERY] Deleting old collection and recreating...")
                try:
                    self.chroma_client.delete_collection(config.collection_name)
                except Exception:
                    pass
            else:
                log.warning("[RECOVERY] ChromaDB error: %s", e)
                log.warning("[RECOVERY] Clearing corrupted database...")
                # Nuclear cleanup — delete all ChromaDB data
                chroma_dir = config.chroma_dir
                if chroma_dir.exists():
                    for item in chroma_dir.iterdir():
                        try:
                            if item.is_dir():
                                shutil.rmtree(item)
                            else:
                                item.unlink()
                        except Exception:
                            pass
                # Recreate client
                self.chroma_client = chromadb.PersistentClient(path=str(config.chroma_dir))

            log.info("[RECOVERY] Creating fresh collection...")
            return self.chroma_client.get_or_create_collection(
                name=config.collection_name,
                embedding_function=self.embed_fn,
                metadata={"description": "Knowledge base for RAG"},
            )

    def _check_dimension_mismatch(self) -> bool:
        """Check if stored embeddings have different dimension than current config.

        Uses a test query to detect dimension mismatch (more reliable than
        reading stored embeddings which may not be available in all ChromaDB backends).
        """
        if self.collection.count() == 0:
            return False
        try:
            # Attempt a real query — ChromaDB will throw if dimensions don't match
            self.collection.query(query_texts=["dimension check"], n_results=1, include=["documents"])
            return False  # Query succeeded, dimensions match
        except Exception as e:
            error_msg = str(e).lower()
            if "dimension" in error_msg:
                log.warning("[MIGRATION] Embedding dimension mismatch detected: %s", e)
                log.warning("[MIGRATION] Nuclear rebuild required.")
                return True
            # Other error — don't trigger rebuild
            log.warning("[WARN] Dimension check query failed (non-dimension error): %s", e)
            return False

    _bm25_build_lock = threading.Lock()

    def _ensure_bm25_index(self) -> None:
        """Lazy initialization of BM25 index from existing ChromaDB data.

        Only marks ``_bm25_initialized=True`` after a successful build with
        actual content. Empty-collection bootup or build failures leave the
        flag ``False`` so the next call retries once documents are present.

        Prior behavior set the flag unconditionally at the end of the guarded
        block, which trapped the index in an uninitialized state when the
        server booted against an empty collection (issue #114): subsequent
        ``add_document`` calls populate ``bm25_index.corpus`` but do not
        rebuild the inverted index, and this method — the only path that
        does — would short-circuit forever on the stale flag.
        """
        if self._bm25_initialized:
            return
        with self._bm25_build_lock:
            if self._bm25_initialized:
                return

            try:
                count = self.collection.count()
                if count == 0:
                    # Empty collection — nothing to build. Leave flag False so
                    # a later call retries once documents become available.
                    return

                all_data = self.collection.get(include=["documents"], limit=count)
                if not all_data["ids"] or not all_data["documents"]:
                    return

                self.bm25_index.add_documents(all_data["ids"], all_data["documents"])
                self.bm25_index.build_index()
                log.info("[INFO] BM25 index built with %d documents", len(self.bm25_index))
                self._bm25_initialized = True
            except Exception as e:
                log.warning("[WARN] Failed to build BM25 index: %s", e)
                # Do not mark initialized; retry allowed on next call.

    # =========================================================================
    # Indexing
    # =========================================================================

    _index_lock = threading.Lock()

    def index_all(self, force: bool = False) -> Dict[str, Any]:
        """
        Index documents with incremental change detection.

        Compares file mtime/size against stored metadata to detect changes.
        Only re-indexes files that are new or modified.  Serialized via
        _index_lock so concurrent calls (watcher + MCP tool) don't corrupt
        ChromaDB's SQLite database.
        """
        # A2.6 — traced at the entry point so watcher-initiated and MCP-tool
        # initiated reindexes share the same span name for filtering in APM.
        _tracer = get_tracer()
        _span_ctx = _tracer.start_as_current_span("knowledge_rag.index_all", attributes={"index.force": bool(force)})
        _span_ctx.__enter__()
        try:
            return self._index_all_traced(force)
        finally:
            _span_ctx.__exit__(None, None, None)

    def _index_all_traced(self, force: bool) -> Dict[str, Any]:
        """Inner body of :meth:`index_all` — lock acquisition inside the span."""
        if not self._index_lock.acquire(blocking=False):
            return {
                "total_files": 0,
                "indexed": 0,
                "updated": 0,
                "skipped": 0,
                "deleted": 0,
                "errors": 0,
                "chunks_added": 0,
                "chunks_removed": 0,
                "dedup_skipped": 0,
                "categories": {},
                "skipped_reason": "reindex_already_running",
            }
        try:
            return self._index_all_impl(force)
        finally:
            self._index_lock.release()

    def _index_all_impl(self, force: bool = False) -> Dict[str, Any]:
        """Inner implementation of index_all (caller holds _index_lock)."""
        stats = {
            "total_files": 0,
            "indexed": 0,
            "updated": 0,
            "skipped": 0,
            "deleted": 0,
            "errors": 0,
            "chunks_added": 0,
            "chunks_removed": 0,
            "dedup_skipped": 0,
            "categories": {},
        }

        documents = self.parser.parse_directory()
        stats["total_files"] = len(documents)
        self._reindex_progress["total_files"] = stats["total_files"]
        if stats["total_files"] > 100:
            log.info("[INDEX] Scanning %d documents...", stats["total_files"])

        path_to_docid: Dict[str, str] = {}
        for doc_id, info in list(self._indexed_docs.items()):
            path_to_docid[info.get("source", "")] = doc_id

        current_paths = {str(doc.source) for doc in documents}

        # Clean up orphaned docs BEFORE indexing so that moved files
        # are not blocked by stale content hashes (fixes #90).
        orphan_ids = []
        for doc_id, info in list(self._indexed_docs.items()):
            if info.get("source", "") not in current_paths:
                removed = self._remove_document_chunks(doc_id)
                stats["chunks_removed"] += removed
                stats["deleted"] += 1
                orphan_ids.append(doc_id)

        for doc_id in orphan_ids:
            src = self._indexed_docs[doc_id].get("source", "")
            if src:
                self._source_to_docid.pop(str(Path(src).resolve()), None)
            del self._indexed_docs[doc_id]

        _progress_interval = max(1, stats["total_files"] // 10)

        for idx, doc in enumerate(documents):
            try:
                source_str = str(doc.source)
                existing_doc_id = path_to_docid.get(source_str)

                if not force and existing_doc_id:
                    existing_meta = self._indexed_docs.get(existing_doc_id, {})
                    stored_mtime = existing_meta.get("file_mtime", "")
                    stored_size = existing_meta.get("file_size", 0)

                    try:
                        current_stat = doc.source.stat()
                        current_mtime = datetime.fromtimestamp(current_stat.st_mtime).isoformat()
                        current_size = current_stat.st_size
                    except OSError:
                        current_mtime = ""
                        current_size = 0

                    if stored_mtime == current_mtime and stored_size == current_size:
                        stats["skipped"] += 1
                        continue

                    removed = self._remove_document_chunks(existing_doc_id)
                    stats["chunks_removed"] += removed
                    src = self._indexed_docs[existing_doc_id].get("source", "")
                    if src:
                        self._source_to_docid.pop(str(Path(src).resolve()), None)
                    del self._indexed_docs[existing_doc_id]
                    stats["updated"] += 1
                elif not force and doc.id in self._indexed_docs:
                    stats["skipped"] += 1
                    continue

                chunks_added, dedup_skipped = self._index_document(doc)

                if not (existing_doc_id and not force):
                    stats["indexed"] += 1
                stats["chunks_added"] += chunks_added
                stats["dedup_skipped"] += dedup_skipped
                stats["categories"][doc.category] = stats["categories"].get(doc.category, 0) + 1

                try:
                    file_stat = doc.source.stat()
                    file_mtime = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    file_size = file_stat.st_size
                except OSError:
                    file_mtime = datetime.now().isoformat()
                    file_size = 0

                self._indexed_docs[doc.id] = {
                    "source": str(doc.source),
                    "category": doc.category,
                    "format": doc.format,
                    "chunks": chunks_added,
                    "keywords": doc.keywords,
                    "indexed_at": datetime.now().isoformat(),
                    "file_mtime": file_mtime,
                    "file_size": file_size,
                }
                self._source_to_docid[str(doc.source.resolve())] = doc.id

            except Exception as e:
                stats["errors"] += 1
                log.error("[ERROR] Failed to index %s: %s", doc.source, e)

            self._reindex_progress.update(
                {
                    "processed": idx + 1,
                    "indexed": stats["indexed"],
                    "skipped": stats["skipped"],
                    "errors": stats["errors"],
                }
            )

            if stats["total_files"] > 100 and (idx + 1) % _progress_interval == 0:
                pct = int((idx + 1) / stats["total_files"] * 100)
                log.info(
                    "[INDEX] Progress: %d/%d (%d%%) — %d new, %d skipped",
                    idx + 1,
                    stats["total_files"],
                    pct,
                    stats["indexed"],
                    stats["skipped"],
                )

        self._save_metadata()

        if stats["indexed"] > 0 or stats["updated"] > 0 or stats["deleted"] > 0:
            self.query_cache.invalidate()

        return stats

    _CHROMA_BATCH_SIZE = 500

    def _index_document(self, doc: Document) -> Tuple[int, int]:
        """Index a single document's chunks into ChromaDB and BM25 with dedup.

        Large documents are split into batches of _CHROMA_BATCH_SIZE to
        prevent memory spikes when embedding thousands of chunks at once.

        Pre-computed embeddings (R5.7 — Late Chunking): when EVERY unique
        chunk carries a non-``None`` ``embedding`` attribute, the batch
        add call passes ``embeddings=[...]`` alongside ``documents=[...]``
        so ChromaDB stores the caller-supplied vectors instead of
        invoking its embedding function. Any partial coverage — one
        chunk without a pre-computed vector — falls back to the standard
        embed-on-add path for the WHOLE batch to keep the vector-space
        homogeneous (mixing late-chunked and standard vectors inside one
        collection would silently skew cosine similarity).
        """
        if not doc.chunks:
            return 0, 0

        unique_ids = []
        unique_docs = []
        unique_metas = []
        unique_embeddings: List[Optional[List[float]]] = []
        dedup_skipped = 0
        seen_hashes: set = set()

        for chunk in doc.chunks:
            content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:20]
            chunk_id = f"{doc.id}_{chunk.index}"

            if content_hash in seen_hashes:
                dedup_skipped += 1
                continue

            seen_hashes.add(content_hash)
            unique_ids.append(chunk_id)
            unique_docs.append(chunk.content)
            unique_metas.append(
                {
                    "doc_id": doc.id,
                    "source": str(doc.source),
                    "filename": doc.filename,
                    "category": doc.category,
                    "format": doc.format,
                    "chunk_index": chunk.index,
                    "keywords": ",".join(doc.keywords[:10]),
                    "content_hash": content_hash,
                    **chunk.metadata,
                }
            )
            unique_embeddings.append(getattr(chunk, "embedding", None))

        if unique_ids:
            bs = self._CHROMA_BATCH_SIZE
            # Only use pre-computed embeddings when the entire batch has
            # them — a partial pass would leave ChromaDB mixing caller-
            # supplied vectors with embedder-produced ones, and no
            # invariant guarantees they share a vector space.
            use_precomputed = all(e is not None for e in unique_embeddings)
            for i in range(0, len(unique_ids), bs):
                add_kwargs: Dict[str, Any] = {
                    "ids": unique_ids[i : i + bs],
                    "documents": unique_docs[i : i + bs],
                    "metadatas": unique_metas[i : i + bs],
                }
                if use_precomputed:
                    add_kwargs["embeddings"] = [e for e in unique_embeddings[i : i + bs] if e is not None]
                self.collection.add(**add_kwargs)
            self.bm25_index.add_documents(unique_ids, unique_docs)

        return len(unique_ids), dedup_skipped

    def _remove_document_chunks(self, doc_id: str) -> int:
        """Remove all chunks belonging to a document from ChromaDB and BM25."""
        try:
            results = self.collection.get(where={"doc_id": doc_id}, include=[])

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                self._bm25_initialized = False
                return len(results["ids"])
        except Exception as e:
            log.warning("[WARN] Failed to remove chunks for doc %s: %s", doc_id, e)

        return 0

    def start_reindex_background(self, mode: str) -> Dict[str, Any]:
        """Start reindex in a background thread. Returns immediately."""
        if self._reindex_progress.get("active"):
            return {"status": "already_running", "progress": dict(self._reindex_progress)}

        self._reindex_progress = {
            "active": True,
            "operation": mode,
            "total_files": 0,
            "processed": 0,
            "indexed": 0,
            "skipped": 0,
            "errors": 0,
            "started_at": datetime.now().isoformat(),
        }

        target = {
            "incremental": lambda: self.index_all(force=False),
            "smart_reindex": self.reindex_all,
            "nuclear_rebuild": self.nuclear_rebuild,
        }[mode]

        thread = threading.Thread(target=self._run_reindex, args=(target,), daemon=True)
        thread.start()
        return {"status": "started", "operation": mode}

    def _run_reindex(self, target: Any) -> None:
        """Background thread runner for reindex operations."""
        try:
            result = target()
            self._reindex_progress["result"] = result
        except Exception as e:
            self._reindex_progress["error"] = str(e)
            log.error("[ERROR] Background reindex failed: %s", e)
        finally:
            self._reindex_progress["active"] = False

    def reindex_all(self) -> Dict[str, Any]:
        """Smart reindex: incremental detection + BM25 rebuild + orphan cleanup."""
        import shutil

        _tracer = get_tracer()
        with _tracer.start_as_current_span("knowledge_rag.reindex_all"):
            return self._reindex_all_impl(shutil)

    def _reindex_all_impl(self, shutil: Any) -> Dict[str, Any]:
        """Body of :meth:`reindex_all` — separated so the span wraps everything."""
        log.info("[REINDEX] Starting smart incremental reindex...")
        start_time = time.time()

        stats = self.index_all(force=False)

        log.info("[REINDEX] Rebuilding BM25 index...")
        self.bm25_index.clear()
        self._bm25_initialized = False
        self._ensure_bm25_index()

        chroma_dir = config.chroma_dir
        orphans_cleaned = 0
        if chroma_dir.exists():
            for item in chroma_dir.iterdir():
                if item.is_dir() and len(item.name) == 36 and "-" in item.name:
                    try:
                        if not any(item.iterdir()):
                            shutil.rmtree(item)
                            orphans_cleaned += 1
                    except Exception:
                        pass

        self.query_cache.invalidate()

        elapsed = time.time() - start_time
        stats["orphan_folders_cleaned"] = orphans_cleaned
        stats["elapsed_seconds"] = round(elapsed, 2)
        log.info(
            "[REINDEX] Completed in %.1fs (indexed: %d, updated: %d, skipped: %d, deleted: %d)",
            elapsed,
            stats["indexed"],
            stats["updated"],
            stats["skipped"],
            stats["deleted"],
        )

        return stats

    def nuclear_rebuild(self) -> Dict[str, Any]:
        """Nuclear rebuild: DELETE everything and re-embed ALL documents."""
        import shutil

        _tracer = get_tracer()
        with _tracer.start_as_current_span("knowledge_rag.nuclear_rebuild"):
            return self._nuclear_rebuild_impl(shutil)

    def _nuclear_rebuild_impl(self, shutil: Any) -> Dict[str, Any]:
        """Body of :meth:`nuclear_rebuild` — separated so the span wraps everything."""
        log.info("[NUCLEAR] Starting full rebuild...")
        start_time = time.time()

        try:
            self.chroma_client.delete_collection(config.collection_name)
            log.info("[NUCLEAR] Deleted ChromaDB collection")
        except Exception:
            pass

        chroma_dir = config.chroma_dir
        if chroma_dir.exists():
            for item in chroma_dir.iterdir():
                if item.is_dir() and len(item.name) == 36 and "-" in item.name:
                    try:
                        shutil.rmtree(item)
                    except Exception:
                        pass

        self.collection = self.chroma_client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=self.embed_fn,
            metadata={"description": "Knowledge base for RAG"},
        )

        self._indexed_docs = {}
        self._source_to_docid = {}
        self.bm25_index.clear()
        self._bm25_initialized = False
        self.query_cache.invalidate()

        stats = self.index_all(force=True)

        self.bm25_index.build_index()
        self._bm25_initialized = True

        elapsed = time.time() - start_time
        stats["elapsed_seconds"] = round(elapsed, 2)
        log.info(
            "[NUCLEAR] Full rebuild completed in %.1fs (%d docs, %d chunks)",
            elapsed,
            stats["indexed"],
            stats["chunks_added"],
        )

        return stats

    # =========================================================================
    # Search
    # =========================================================================

    def query(
        self,
        query_text: str,
        max_results: int = None,
        category_filter: Optional[str] = None,
        hybrid_alpha: Optional[float] = None,
        fusion: Optional[str] = None,
        query_rewrite: Optional[bool] = None,
        self_query: Optional[bool] = None,
        hyde: Optional[int] = None,
        multi_query: Optional[int] = None,
        adaptive: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with pluggable fusion + cross-encoder reranking.

        Pipeline: Semantic + BM25 -> Fusion (default RRF) -> Reranker -> Results

        Args:
            query_text: Raw user query.
            max_results: Cap on returned chunks.
            category_filter: Optional category filter.
            hybrid_alpha: Semantic/keyword balance in ``[0.0, 1.0]``.
            fusion: Optional fusion strategy override — one of ``rrf``,
                ``combsum``, ``combmnz``, ``weighted``. ``None`` (default)
                falls back to ``config.fusion_strategy`` which itself defaults
                to ``rrf``. The historical RRF-K=60 path stays byte-identical
                when ``fusion`` is ``None`` and no ``search.fusion.strategy``
                is set in ``config.yaml``.
            query_rewrite: Optional per-call override for
                ``config.query_rewrite_enabled`` (A3.2). ``True`` forces the
                LLM rewrite path, ``False`` disables it for this call, and
                ``None`` (default) inherits from config. Fails open on any
                LLM error — a failed rewrite silently falls back to the raw
                ``query_text`` and retrieval continues.
            self_query: Optional per-call override for
                ``config.self_query_enabled`` (A3.5). ``True`` forces the
                LLM filter-extraction path, ``False`` disables it for this
                call, and ``None`` (default) inherits from config. Only
                fires when the caller did NOT pass an explicit
                ``category_filter`` — explicit intent always wins. Fails
                open on any LLM error (raw query kept, no filters inferred).
            hyde: Optional per-call override for HyDE (A3.3). A positive
                integer forces the LLM path with that many hypothetical
                passages, ``0`` disables HyDE for this call, and ``None``
                (default) inherits ``config.hyde_num_hypotheses`` when
                ``config.hyde_enabled`` is True. Fails open on any LLM /
                embedding error — the semantic branch reverts to the
                default ``query_texts=[query]`` path and retrieval continues.
            multi_query: Optional per-call override for the A3.4 LLM
                multi-query fan-out. Integer ``N > 1`` forces N-way
                retrieval (original query + up to N-1 LLM-generated
                variations, each running the full single-query pipeline;
                the N formatted result lists are then merged via a
                top-level multi-query RRF-K=60). ``0`` / ``1`` disables
                fan-out for this call. ``None`` (default) inherits
                ``config.multi_query_n`` when ``config.multi_query_enabled``
                is True. Fails open on any LLM error — a failed variation
                generation collapses to single-query retrieval.
            adaptive: Optional per-call override for the A3.9 LLM
                adaptive retrieval router. ``True`` forces the router
                to classify the query into one of ``simple`` / ``hybrid``
                / ``multi_hop`` / ``code`` / ``filter`` and apply the
                per-strategy overrides in
                :data:`~mcp_server.retrieval.llm_features.adaptive.ROUTE_TO_FLAGS`
                on top of whatever the caller passed (user-explicit
                values ALWAYS win — router hints only fill in the blanks
                for ``hybrid_alpha`` / ``self_query`` / ``hyde`` /
                ``multi_query``). ``False`` disables the router for this
                call regardless of config. ``None`` (default) inherits
                ``config.adaptive_retrieval_enabled``. Fails open to the
                ``hybrid`` strategy (empty override dict, byte-identical
                to the pre-A3.9 path) on any LLM error.
        """
        # A2.6 — OpenTelemetry span wraps the entire pipeline. Zero-cost when
        # tracing is disabled: the no-op tracer's __enter__/__exit__ do nothing.
        tracer = get_tracer()
        # A2.4 — resolve the effective fusion name once so cache keys and
        # tracing attributes match the strategy that actually runs.
        effective_fusion = fusion or config.fusion_strategy

        # A3.9 — adaptive retrieval router. Runs BEFORE any of the other
        # Fase 3 features so its per-strategy hints can flip them on for
        # THIS query without touching the operator's static config. The
        # router is strictly additive: it only fills in defaults for
        # params the caller left as ``None`` / omitted, so explicit user
        # intent (``self_query=True``, ``hybrid_alpha=0.2``, ...) always
        # beats the router's suggestion. Fails open to ``"hybrid"`` (an
        # empty override dict) on any LLM error — no visible behaviour
        # change vs. the pre-A3.9 path when the router misfires.
        (
            hybrid_alpha,
            self_query,
            hyde,
            multi_query,
            _adaptive_strategy,
        ) = self._maybe_route_query(
            query_text,
            adaptive,
            hybrid_alpha,
            self_query,
            hyde,
            multi_query,
        )

        # A3.5 — optional LLM self-query filter extraction. Runs BEFORE the
        # rewrite so the filter cues ("in the redteam category", "after
        # 2024-01-01", ...) are stripped from the natural-language portion
        # the rewriter sees. Only touches the query when the caller did
        # NOT pin ``category_filter`` explicitly — self-query never
        # overrides explicit user intent. Failure modes collapse to the
        # raw query with all filters None; retrieval never breaks.
        (
            effective_query,
            category_filter,
            _self_query_source_contains,
            _self_query_date_after,
            _self_query_date_before,
        ) = self._maybe_self_query(query_text, category_filter, self_query)

        # A3.2 — optional LLM query rewrite. Runs BEFORE the query_cache
        # lookup so cached rankings key off the rewritten text (same raw
        # query + same rewrite ⇒ same cache slot). The rewrite call is
        # itself semantic-cache-backed so repeat raw queries pay zero LLM
        # cost. Failure modes (no provider, timeout, bad output) collapse
        # into ``effective_query == query_text`` — retrieval never breaks.
        effective_query = self._maybe_rewrite_query(effective_query, query_rewrite)

        # A3.3 — optional HyDE (Hypothetical Document Embeddings). Runs
        # AFTER the rewrite so the hypothetical passages are generated
        # from the LLM-cleaned form. Returns None when disabled OR on
        # any failure; the semantic branch then falls back to the
        # historical ``ChromaDB.query(query_texts=[query])`` path. The
        # BM25 branch is deliberately untouched — sparse retrieval on
        # hallucinated passages hurts exact-term matching (CVE IDs,
        # tool names, file paths).
        hyde_embedding = self._maybe_apply_hyde(effective_query, hyde)

        # A3.4 — optional multi-query fan-out. Runs AFTER rewrite + HyDE
        # so the variations paraphrase the LLM-cleaned form and every
        # variation retrieval reuses the same HyDE embedding (HyDE runs
        # exactly once even in fan-out mode, keeping cost proportional
        # to N x semantic-BM25-fusion rather than N x HyDE-LLM-call).
        # Returns ``[effective_query]`` when disabled OR on any failure.
        variations = self._maybe_multi_query_variations(effective_query, multi_query)

        with tracer.start_as_current_span(
            "knowledge_rag.query",
            attributes={
                "query.length": len(effective_query or ""),
                "query.max_results": max_results or config.default_results,
                "query.category_filter": category_filter or "",
                "query.hybrid_alpha": hybrid_alpha,
                "query.fusion": effective_fusion,
                "query.rewritten": effective_query != query_text,
                "query.self_query_category": category_filter or "",
                "query.hyde_used": hyde_embedding is not None,
                "query.multi_query_n": len(variations),
                # A3.9 — router strategy is empty string when router did
                # not run (disabled OR skipped via ``adaptive=False``);
                # populated with one of simple/hybrid/multi_hop/code/filter
                # when it did.
                "query.adaptive_strategy": _adaptive_strategy or "",
            },
        ) as _root_span:
            max_results = max_results or config.default_results

            # Check cache — fusion name is folded into the cache key so that
            # switching strategies at runtime never returns a stale ranking.
            # The delimiter is ASCII Unit Separator (0x1F) which cannot appear
            # in a normal query text, so we cannot collide with a real query.
            # HyDE flag is folded in too: enabling HyDE changes the ranking
            # even for the same effective query, so a stale non-HyDE cache
            # entry must not shadow a fresh HyDE-augmented one. Multi-query
            # N is folded in for the same reason: a stale N=1 entry must
            # never shadow a fresh N=3 fan-out ranking.
            hyde_key = "hyde" if hyde_embedding is not None else "raw"
            mq_key = f"mq={len(variations)}"
            cache_key = f"{effective_query}\x1ffusion={effective_fusion}\x1femb={hyde_key}\x1f{mq_key}"
            cached = self.query_cache.get(cache_key, max_results, category_filter, hybrid_alpha)
            if cached is not None:
                _root_span.set_attribute("query.cache_hit", True)
                _root_span.set_attribute("query.result_count", len(cached))
                # M4.2 — layer the Work Memory overlay on cache hits too so
                # a flag toggled AFTER the entry was cached still surfaces
                # (or hides) the ``learning`` tag on the next call. Mutates
                # in place: the overlay is idempotent, applying it twice on
                # the same list produces the same result, and the ``learning``
                # key is namespaced so it cannot collide with any pre-existing
                # search-payload field.
                _apply_work_memory_overlay(cached)
                # M4.4 — dashboard query log hook. Zero cost when disabled.
                _maybe_log_query(query_text, cached)
                return cached
            _root_span.set_attribute("query.cache_hit", False)

            if len(variations) > 1:
                results = self._multi_query_impl(
                    variations,
                    max_results,
                    category_filter,
                    hybrid_alpha,
                    effective_fusion,
                    _root_span,
                    query_embedding=hyde_embedding,
                )
            else:
                results = self._query_impl(
                    effective_query,
                    max_results,
                    category_filter,
                    hybrid_alpha,
                    effective_fusion,
                    _root_span,
                    query_embedding=hyde_embedding,
                )

            # M4.2 — Work Memory / Lessons Learned overlay. Runs LAST so
            # every retrieval path (single-query, multi-query fan-out,
            # HyDE, cache-hit) sees the same tagging behaviour. Feature
            # is DEFAULT OFF: with ``config.work_memory_enabled`` false
            # the helper short-circuits before opening the overlay file
            # and the output shape is byte-identical to pre-M4.2.
            _apply_work_memory_overlay(results)
            # M4.4 — dashboard query log hook. Runs AFTER the overlay so a
            # single append reflects the exact payload the caller sees.
            # Best-effort: any failure is swallowed and never breaks the
            # search. Zero cost when ``config.query_log_enabled`` is False.
            _maybe_log_query(query_text, results)
            return results

    def _maybe_route_query(
        self,
        query_text: str,
        override: Optional[bool],
        user_hybrid_alpha: Optional[float],
        user_self_query: Optional[bool],
        user_hyde: Optional[int],
        user_multi_query: Optional[int],
    ) -> Tuple[float, Optional[bool], Optional[int], Optional[int], Optional[str]]:
        """Apply the A3.9 LLM adaptive retrieval router when opted-in.

        Extracted from :meth:`query` for the same reasons as the sibling
        ``_maybe_*`` helpers: keeps :meth:`query` readable and gives tests
        a trivial patch point (mock this method to force specific route
        outputs without touching the LLM registry).

        Precedence for whether to consult the router:
            1. ``override`` when explicitly ``True`` / ``False`` at the
               call site (MCP tool arg, CLI flag).
            2. ``config.adaptive_retrieval_enabled`` otherwise.

        When the router runs, its per-strategy overrides in
        :data:`~mcp_server.retrieval.llm_features.adaptive.ROUTE_TO_FLAGS`
        are applied ONLY to parameters the caller left as ``None`` /
        omitted. Explicit user intent ALWAYS beats the router:

        * ``user_hybrid_alpha`` non-None → router's ``hybrid_alpha`` hint
          is discarded, the caller's value is returned verbatim.
        * ``user_self_query`` non-None → the router's ``self_query`` hint
          is discarded, ``user_self_query`` is returned verbatim.
        * ``user_hyde`` non-None → router's ``hyde`` hint is discarded.
        * ``user_multi_query`` non-None → router's ``multi_query`` hint
          is discarded.

        Fails open: any router failure (missing LLM, network timeout,
        unknown strategy from the LLM) collapses to the ``"hybrid"``
        strategy which emits an empty override dict. Effective values
        then reduce to the caller-supplied ones (or their defaults),
        making adaptive mode byte-identical to the pre-A3.9 path on
        failure.

        Args:
            query_text: Raw user query fed to the router prompt.
            override: Per-call override; ``None`` inherits from config.
            user_hybrid_alpha: Caller-supplied ``hybrid_alpha`` or
                ``None`` when the caller wants the effective default.
            user_self_query: Caller-supplied ``self_query`` toggle or
                ``None``.
            user_hyde: Caller-supplied ``hyde`` count / ``0`` disable /
                ``None`` inherit.
            user_multi_query: Caller-supplied ``multi_query`` fan-out
                count / ``0`` / ``1`` disable / ``None`` inherit.

        Returns:
            Five-tuple ``(effective_hybrid_alpha, effective_self_query,
            effective_hyde, effective_multi_query, strategy_name)``. The
            strategy name is returned for logging / tracing and is
            ``None`` when the router did not run.
        """
        # Effective hybrid_alpha defaults to 0.5 (the historical
        # orchestrator default) when neither caller nor router set it.
        # Kept here so every early-return path hands back the same fallback.
        default_alpha = 0.5

        enabled = override if override is not None else config.adaptive_retrieval_enabled
        if not enabled:
            effective_alpha = user_hybrid_alpha if user_hybrid_alpha is not None else default_alpha
            return (
                effective_alpha,
                user_self_query,
                user_hyde,
                user_multi_query,
                None,
            )

        # Deferred imports so opt-in feature code never runs on the
        # default retrieval path (byte-identical to pre-A3.9).
        from mcp_server.retrieval.llm_features.adaptive import ROUTE_TO_FLAGS, route_query
        from mcp_server.retrieval.semantic_cache import SemanticCache

        cache: Optional[SemanticCache] = None
        if config.semantic_cache_enabled:
            try:
                cache_dir_raw = config.semantic_cache_dir
                cache_path = Path(cache_dir_raw)
                if not cache_path.is_absolute():
                    cache_path = config.data_dir.parent / cache_dir_raw
                ttl_seconds = (
                    None if config.semantic_cache_ttl_days == 0 else config.semantic_cache_ttl_days * 24 * 3600
                )
                cache = SemanticCache(
                    cache_dir=cache_path,
                    ttl_seconds=ttl_seconds,
                    max_entries=config.semantic_cache_max_entries,
                )
            except Exception as exc:  # pragma: no cover — defensive
                log.debug("SemanticCache instantiation failed: %s. Cache disabled.", exc)
                cache = None

        provider = config.llm_provider or None
        try:
            strategy = route_query(query_text, provider_name=provider, cache=cache)
        except Exception as exc:  # pragma: no cover — route_query is fail-open itself
            log.debug("Adaptive router raised unexpectedly: %s. Falling back to 'hybrid'.", exc)
            strategy = "hybrid"

        overrides = ROUTE_TO_FLAGS.get(strategy, {})

        # Apply router overrides only where the caller did NOT pass an
        # explicit value. This is the "user-explicit wins" contract.
        if user_hybrid_alpha is not None:
            effective_alpha = user_hybrid_alpha
        elif "hybrid_alpha" in overrides:
            effective_alpha = float(overrides["hybrid_alpha"])
        else:
            effective_alpha = default_alpha

        if user_self_query is not None:
            effective_self_query = user_self_query
        elif "self_query" in overrides:
            effective_self_query = bool(overrides["self_query"])
        else:
            effective_self_query = None

        # HyDE hint is a bool in ROUTE_TO_FLAGS but the orchestrator
        # param is an int (count). Translate:
        #   True  → config.hyde_num_hypotheses (or 1 if missing / <= 0)
        #   False → 0 (explicit disable)
        # Integer hint from the router (if a future strategy adds one)
        # passes through verbatim.
        if user_hyde is not None:
            effective_hyde = user_hyde
        elif "hyde" in overrides:
            hyde_hint = overrides["hyde"]
            if hyde_hint is True:
                _cfg_n = int(getattr(config, "hyde_num_hypotheses", 1) or 1)
                effective_hyde = _cfg_n if _cfg_n >= 1 else 1
            elif hyde_hint is False:
                effective_hyde = 0
            else:
                try:
                    effective_hyde = int(hyde_hint)
                except (TypeError, ValueError):
                    effective_hyde = None
        else:
            effective_hyde = None

        if user_multi_query is not None:
            effective_multi_query = user_multi_query
        elif "multi_query" in overrides:
            try:
                effective_multi_query = int(overrides["multi_query"])
            except (TypeError, ValueError):
                effective_multi_query = None
        else:
            effective_multi_query = None

        log.info(
            "Adaptive router chose strategy=%r (overrides=%s)",
            strategy,
            {
                k: v
                for k, v in overrides.items()
                if k in ("hybrid_alpha", "self_query", "hyde", "multi_query", "code_aware")
            },
        )

        return (
            effective_alpha,
            effective_self_query,
            effective_hyde,
            effective_multi_query,
            strategy,
        )

    def _maybe_self_query(
        self,
        query_text: str,
        category_filter: Optional[str],
        override: Optional[bool],
    ) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Apply the A3.5 LLM self-query filter extraction when opted-in.

        Extracted from :meth:`query` for the same reasons as
        :meth:`_maybe_rewrite_query`: keeps :meth:`query` readable and
        gives tests a trivial patch point (mock this method to force
        specific extraction results without touching the LLM registry).

        Precedence for whether to attempt extraction:
            1. ``override`` when explicitly ``True`` / ``False`` at the
               call site (MCP tool arg, CLI flag).
            2. ``config.self_query_enabled`` otherwise.

        Explicit intent wins: when the caller passed a non-None
        ``category_filter``, self-query is skipped entirely so the user's
        pinned category is never even hashed into the LLM's context.
        (Skipping the LLM call is a stronger guarantee than running it
        and discarding the ``category`` field — the operator's log stays
        clean of a self-query event they never asked for.)

        Args:
            query_text: Raw user query.
            category_filter: Explicit category from the caller, if any.
                ``None`` unlocks self-query; a truthy value short-circuits
                the whole feature.
            override: Per-call override; ``None`` inherits from config.

        Returns:
            Five-tuple: ``(effective_query, category_filter,
            source_contains, date_after, date_before)``. On skip /
            failure, ``effective_query`` is ``query_text`` unchanged,
            ``category_filter`` is the input passthrough, and every
            other field is ``None``.
        """
        enabled = override if override is not None else config.self_query_enabled
        if not enabled:
            return query_text, category_filter, None, None, None

        # Explicit-wins: caller already pinned a category. Never override.
        if category_filter is not None:
            return query_text, category_filter, None, None, None

        # Deferred imports so opt-in feature code never runs on the
        # default retrieval path (byte-identical to pre-A3.5).
        from mcp_server.retrieval.llm_features.self_query import extract_filters
        from mcp_server.retrieval.semantic_cache import SemanticCache

        cache: Optional[SemanticCache] = None
        if config.semantic_cache_enabled:
            try:
                cache_dir_raw = config.semantic_cache_dir
                cache_path = Path(cache_dir_raw)
                if not cache_path.is_absolute():
                    cache_path = config.data_dir.parent / cache_dir_raw
                ttl_seconds = (
                    None if config.semantic_cache_ttl_days == 0 else config.semantic_cache_ttl_days * 24 * 3600
                )
                cache = SemanticCache(
                    cache_dir=cache_path,
                    ttl_seconds=ttl_seconds,
                    max_entries=config.semantic_cache_max_entries,
                )
            except Exception as exc:  # pragma: no cover — defensive
                log.debug("SemanticCache instantiation failed: %s. Cache disabled.", exc)
                cache = None

        provider = config.llm_provider or None
        # Valid categories drawn from config — ``category_mappings`` values
        # are the canonical names stamped into chunk metadata at index time,
        # so filtering by any other name is guaranteed to miss.
        categories = list({v for v in config.category_mappings.values() if isinstance(v, str)})
        result = extract_filters(
            query_text,
            categories=categories,
            provider_name=provider,
            cache=cache,
        )

        cleaned_query = result.get("cleaned_query") or query_text
        inferred_category = result.get("category")
        if inferred_category:
            log.info(
                "Self-query inferred category=%r for %r (cleaned=%r)",
                inferred_category,
                query_text,
                cleaned_query,
            )

        return (
            cleaned_query,
            inferred_category,
            result.get("source_contains"),
            result.get("date_after"),
            result.get("date_before"),
        )

    def _maybe_rewrite_query(self, query_text: str, override: Optional[bool]) -> str:
        """Apply the A3.2 LLM query rewrite when opted-in.

        Extracted so :meth:`query` stays readable and the wiring is
        trivial to mock in tests (patch this method to force / suppress
        a rewrite without touching the LLM registry).

        Precedence for whether to attempt a rewrite:
            1. ``override`` when explicitly ``True`` / ``False`` at the
               call site (MCP tool arg, CLI flag).
            2. ``config.query_rewrite_enabled`` otherwise.

        Args:
            query_text: Raw user query.
            override: Per-call override; ``None`` inherits from config.

        Returns:
            The rewritten query on success, or ``query_text`` unchanged
            on any failure / when the feature is disabled.
        """
        enabled = override if override is not None else config.query_rewrite_enabled
        if not enabled:
            return query_text

        # Deferred imports so opt-in feature code never runs on the
        # default retrieval path (byte-identical to pre-A3.2).
        from mcp_server.retrieval.llm_features.query_rewrite import rewrite_query
        from mcp_server.retrieval.semantic_cache import SemanticCache

        cache: Optional[SemanticCache] = None
        if config.semantic_cache_enabled:
            try:
                cache_dir_raw = config.semantic_cache_dir
                cache_path = Path(cache_dir_raw)
                if not cache_path.is_absolute():
                    cache_path = config.data_dir.parent / cache_dir_raw
                ttl_seconds = (
                    None if config.semantic_cache_ttl_days == 0 else config.semantic_cache_ttl_days * 24 * 3600
                )
                cache = SemanticCache(
                    cache_dir=cache_path,
                    ttl_seconds=ttl_seconds,
                    max_entries=config.semantic_cache_max_entries,
                )
            except Exception as exc:  # pragma: no cover — defensive
                log.debug("SemanticCache instantiation failed: %s. Cache disabled.", exc)
                cache = None

        provider = config.llm_provider or None
        rewritten = rewrite_query(query_text, provider_name=provider, cache=cache)
        if rewritten != query_text:
            log.info("Query rewritten: %r -> %r", query_text, rewritten)
        return rewritten

    def _maybe_apply_hyde(self, query_text: str, override: Optional[int]) -> Optional[List[float]]:
        """Compute a HyDE-averaged embedding for ``query_text`` when opted-in.

        Extracted from :meth:`query` for the same reasons as
        :meth:`_maybe_rewrite_query` / :meth:`_maybe_self_query`: keeps
        :meth:`query` readable and gives tests a trivial patch point
        (mock this method to force / suppress HyDE without touching the
        LLM registry).

        Precedence for whether to run HyDE and with how many hypotheses:
            1. ``override`` when explicitly set at the call site
               (MCP tool arg, CLI flag). ``0`` disables for this call
               regardless of config; any positive integer forces the
               feature ON with that many hypotheses.
            2. ``config.hyde_enabled`` + ``config.hyde_num_hypotheses``
               otherwise.

        Args:
            query_text: The (already-rewritten, already-self-queried)
                effective query text to seed HyDE with.
            override: Per-call override; ``None`` inherits from config.
                ``0`` disables; positive integers set ``n_hypos``.

        Returns:
            The averaged query embedding on success, or ``None`` when
            HyDE is disabled OR any failure occurred (missing provider,
            LLM error, embedding error). The caller MUST branch on
            ``None`` and fall back to the default text-based semantic
            search path.
        """
        # Resolve the effective ``n_hypos`` for this call. The override
        # semantics are: positive int wins, ``0`` disables, ``None``
        # inherits from config.
        if override is not None:
            if override <= 0:
                return None
            n_hypos = int(override)
        else:
            if not config.hyde_enabled:
                return None
            n_hypos = int(config.hyde_num_hypotheses)
            if n_hypos <= 0:
                return None

        # Deferred imports so opt-in feature code never runs on the
        # default retrieval path (byte-identical to pre-A3.3).
        from mcp_server.retrieval.llm_features.hyde import apply_hyde
        from mcp_server.retrieval.semantic_cache import SemanticCache

        cache: Optional[SemanticCache] = None
        if config.semantic_cache_enabled:
            try:
                cache_dir_raw = config.semantic_cache_dir
                cache_path = Path(cache_dir_raw)
                if not cache_path.is_absolute():
                    cache_path = config.data_dir.parent / cache_dir_raw
                ttl_seconds = (
                    None if config.semantic_cache_ttl_days == 0 else config.semantic_cache_ttl_days * 24 * 3600
                )
                cache = SemanticCache(
                    cache_dir=cache_path,
                    ttl_seconds=ttl_seconds,
                    max_entries=config.semantic_cache_max_entries,
                )
            except Exception as exc:  # pragma: no cover — defensive
                log.debug("SemanticCache instantiation failed: %s. Cache disabled.", exc)
                cache = None

        provider = config.llm_provider or None
        try:
            embedding = apply_hyde(
                query_text,
                self.embed_fn.embed_query,
                n_hypos=n_hypos,
                provider_name=provider,
                cache=cache,
            )
        except Exception as exc:  # pragma: no cover — apply_hyde is fail-open itself
            log.warning("HyDE call raised (%s). Falling back to raw query embedding.", exc)
            return None

        if embedding is not None:
            log.info("HyDE applied: %d hypotheses averaged for %r", n_hypos, query_text[:60])
        return embedding

    def _maybe_multi_query_variations(
        self,
        query_text: str,
        override: Optional[int],
    ) -> List[str]:
        """Generate query variations (A3.4) when multi-query is opted-in.

        Extracted from :meth:`query` for the same reasons as
        :meth:`_maybe_rewrite_query`: keeps :meth:`query` readable and
        gives tests a trivial patch point (mock this method to force
        specific variations without touching the LLM registry).

        Precedence for whether to fan out AND how many queries to run:
            1. ``override`` when explicitly non-``None`` at the call site.
               Values ``<= 1`` disable fan-out for this call regardless of
               ``config.multi_query_enabled``.
            2. ``config.multi_query_n`` when ``config.multi_query_enabled``
               is True. A stale ``multi_query_n <= 1`` in config disables
               the feature the same way an explicit ``0`` at the call
               site does.

        The original ``query_text`` is always the first element of the
        returned list, so a downstream consumer that walks the list in
        order gets an implicit safety net — even if the LLM produces
        useless paraphrases, the original query's retrieval still
        contributes fully to the fused ranking.

        Args:
            query_text: The already-rewritten query (self-query + rewrite
                have already run). Variations paraphrase THIS form so a
                downstream cache hit on the raw + rewrite still lands on
                the same variation-generation slot.
            override: Per-call override; ``None`` inherits from config.

        Returns:
            ``[query_text]`` when the feature is disabled OR any LLM
            failure collapses the fan-out (fail-open). Otherwise
            ``[query_text, variation_1, ..., variation_{n-1}]`` with
            length at most ``n``.
        """
        # ── Resolve N with fail-safe precedence ─────────────────────────
        if override is not None:
            n = override
        elif config.multi_query_enabled:
            n = config.multi_query_n
        else:
            n = 1

        if not isinstance(n, int) or n <= 1:
            return [query_text]

        # Deferred imports so opt-in feature code never runs on the
        # default retrieval path (byte-identical to pre-A3.4).
        from mcp_server.retrieval.llm_features.multi_query import (
            generate_query_variations,
        )
        from mcp_server.retrieval.semantic_cache import SemanticCache

        cache: Optional[SemanticCache] = None
        if config.semantic_cache_enabled:
            try:
                cache_dir_raw = config.semantic_cache_dir
                cache_path = Path(cache_dir_raw)
                if not cache_path.is_absolute():
                    cache_path = config.data_dir.parent / cache_dir_raw
                ttl_seconds = (
                    None if config.semantic_cache_ttl_days == 0 else config.semantic_cache_ttl_days * 24 * 3600
                )
                cache = SemanticCache(
                    cache_dir=cache_path,
                    ttl_seconds=ttl_seconds,
                    max_entries=config.semantic_cache_max_entries,
                )
            except Exception as exc:  # pragma: no cover — defensive
                log.debug("SemanticCache instantiation failed: %s. Cache disabled.", exc)
                cache = None

        provider = config.llm_provider or None
        try:
            variations = generate_query_variations(
                query_text,
                n=n,
                provider_name=provider,
                cache=cache,
            )
        except Exception as exc:  # pragma: no cover — generate_query_variations is fail-open itself
            log.warning("Multi-query call raised (%s). Falling back to single-query.", exc)
            return [query_text]

        # Defensive: generate_query_variations guarantees a non-empty
        # list with the original at position 0, but we double-check here
        # so a future refactor of that contract cannot break the
        # downstream fan-out invariant.
        if not variations:
            return [query_text]
        if variations[0] != query_text:
            log.debug("Multi-query: first variation != original, prepending original.")
            variations = [query_text] + [v for v in variations if v != query_text]

        if len(variations) > 1:
            log.info(
                "Multi-query fan-out active: %d queries (original + %d LLM variations)",
                len(variations),
                len(variations) - 1,
            )
        return variations

    def _multi_query_impl(
        self,
        variations: List[str],
        max_results: int,
        category_filter: Optional[str],
        hybrid_alpha: float,
        fusion_name: str,
        _root_span: Any,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Multi-query fan-out branch of :meth:`query`.

        Runs the full single-query pipeline (semantic + BM25 + fusion +
        rerank + MMR + expansion) once per variation, then fuses the N
        formatted result lists via a top-level multi-query RRF (K=60).

        Design choice — reranker-per-variation vs. reranker-once:

            Two designs were on the table:

            1. **Extract candidate generation**, run semantic+BM25+fusion
               per variation, MERGE via multi-query RRF, then run
               reranker + MMR + expansion ONCE on the merged bag.
            2. **Run the full pipeline per variation**, then fuse the N
               formatted result lists via multi-query RRF.

            Design 2 wins here because:

            * The reranker is cross-encoder; scores are query-dependent.
              Running it once with the ORIGINAL query on merged candidates
              would score every candidate against the original query only,
              throwing away the variation-specific relevance signal.
              Running it per-variation lets each variation's candidates be
              scored against the query they were retrieved for.
            * The result-list rank-fusion literature (Cormack et al. RRF
              2009) is well-supported at ANY pipeline stage. Fusing
              post-rerank formatted lists is not merely "acceptable" —
              it's the canonical multi-query pattern used in TREC.
            * Design 2 lets each variation's :meth:`_query_impl` call
              hit its own ``query_cache`` slot, so repeat identical
              variations across sessions pay zero retrieval cost.

            The N x reranker latency is documented in the config
            comments; multi-query is opt-in specifically because it
            trades latency for recall.

        HyDE composability: ``query_embedding`` (the HyDE hypothetical
        embedding) is computed once from the ORIGINAL rewritten query
        and passed through unchanged to every variation's ``_query_impl``
        call. This keeps HyDE cost bounded at 1 LLM call regardless of
        ``multi_query_n`` while still benefiting every variation's
        semantic branch.

        Args:
            variations: Non-empty list of query strings. Position 0 must
                be the original / primary query — its cache slot is what
                the outer :meth:`query` cache hit is verified against.
            max_results: Cap on returned chunks.
            category_filter: Optional category filter passed through to
                every per-variation call.
            hybrid_alpha: Semantic/keyword balance in ``[0.0, 1.0]``.
            fusion_name: Canonical single-query fusion strategy name used
                INSIDE each variation's semantic/BM25 fusion. The top-level
                multi-query fan-out fusion is always RRF-K=60 regardless.
            _root_span: Parent span; result-count metadata is attached
                once the fan-out completes.
            query_embedding: Optional HyDE embedding shared across all
                variations. ``None`` (default) reverts to the historical
                ``query_texts=[variation]`` semantic branch for every call.

        Returns:
            Merged top-``max_results`` list of formatted result dicts.
            Cache put runs under the original query's cache key so a
            subsequent identical call short-circuits at the outer cache
            check without re-fanning-out.
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("knowledge_rag.query.multi_query_fanout") as _mq_span:
            _mq_span.set_attribute("multi_query.n", len(variations))
            _mq_span.set_attribute("multi_query.top_k", max_results)

            per_variation_results: List[List[Dict[str, Any]]] = []
            for i, variation in enumerate(variations):
                with tracer.start_as_current_span("knowledge_rag.query.multi_query_variation") as _v_span:
                    _v_span.set_attribute("multi_query.variation_index", i)
                    _v_span.set_attribute("multi_query.variation_length", len(variation))
                    # Ask each variation for the full reranker candidate
                    # window so the multi-query fusion sees enough depth
                    # to promote docs that only surfaced deep in one
                    # variation but nowhere in others. Cap defensively:
                    # in the rare case a variation returns fewer than
                    # ``max_results``, we simply have fewer entries to
                    # fuse — no error.
                    v_results = self._query_impl(
                        variation,
                        max_results,
                        category_filter,
                        hybrid_alpha,
                        fusion_name,
                        _v_span,
                        query_embedding=query_embedding,
                    )
                    per_variation_results.append(v_results)

            with tracer.start_as_current_span("knowledge_rag.query.multi_query_rrf") as _fuse_span:
                _fuse_span.set_attribute("multi_query.strategy", "rrf")
                _fuse_span.set_attribute("multi_query.k", 60)
                fused = _fuse_multi_query_results(per_variation_results, k=60, top_k=max_results)
                _fuse_span.set_attribute("multi_query.fused_count", len(fused))

        # Cache put under the ORIGINAL query's cache slot so the outer
        # :meth:`query` cache check finds it on the next identical call
        # without re-running the fan-out. The cache key format MUST match
        # what :meth:`query` computes when ``len(variations) > 1``, or
        # the get / put pair drifts and multi-query never caches.
        put_cache_key = (
            f"{variations[0]}\x1ffusion={fusion_name}"
            f"\x1femb={'hyde' if query_embedding is not None else 'raw'}"
            f"\x1fmq={len(variations)}"
        )
        self.query_cache.put(put_cache_key, max_results, category_filter, hybrid_alpha, fused)

        try:
            _root_span.set_attribute("query.result_count", len(fused))
            _root_span.set_attribute("query.multi_query_fanout", True)
        except Exception:
            pass
        return fused

    def _query_impl(
        self,
        query_text: str,
        max_results: int,
        category_filter: Optional[str],
        hybrid_alpha: float,
        fusion_name: str,
        _root_span: Any,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Cache-miss branch of :meth:`query`.

        Extracted so :meth:`query` stays readable while every sub-stage of the
        pipeline (bm25 init, semantic, bm25, RRF, rerank, MMR, adjacent chunk
        expansion) gets its own OTel span. Attributes on ``_root_span`` are
        upgraded as the pipeline finishes so a trace shows both intent and
        outcome without another span lookup.

        Args:
            query_text: Raw user query.
            max_results: Cap on returned chunks (already normalized).
            category_filter: Optional category filter applied to both
                semantic and BM25 candidates.
            hybrid_alpha: Semantic/keyword balance in ``[0.0, 1.0]``.
            fusion_name: Canonical fusion strategy name resolved by
                :meth:`query` — one of the values in ``fusion.available_strategies``.
            _root_span: Active parent span; result-count metadata is attached
                to it once the pipeline completes.
            query_embedding: Optional precomputed embedding vector for
                the semantic branch (A3.3 HyDE). When ``None`` (default)
                ChromaDB embeds ``query_text`` itself via the configured
                embedding function — this is the historical, byte-identical
                path. When a vector is supplied, ChromaDB skips its own
                embedding step and searches directly against the vector.
                BM25 is unaffected — sparse retrieval always uses
                ``query_text`` as-is regardless of this arg.

        Returns:
            List[Dict[str, Any]]: Formatted results ready for JSON emission.
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("knowledge_rag.query.bm25_init"):
            self._ensure_bm25_index()

        # Keyword routing — informational only.
        # `routed_category` is surfaced via the `routed_by` field for telemetry,
        # but MUST NOT restrict the search when the user did not pass an explicit
        # `category_filter`. Auto-routing to a sparsely-populated category (e.g. one
        # with only a handful of docs) was hiding relevant material that lived under
        # the top-level `security` bucket. Users who want a hard filter still get it
        # by passing `category_filter=...` explicitly.
        routed_category = self._route_by_keywords(query_text)
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        def _matches_category(metadata: Dict[str, Any]) -> bool:
            if not where_filter:
                return True
            expected_category = where_filter.get("category")
            return not expected_category or metadata.get("category") == expected_category

        # Parallel Semantic + BM25 search (threaded for latency reduction)
        semantic_results = {}
        bm25_results = {}

        def _do_semantic():
            r = {}
            if hybrid_alpha > 0:
                try:
                    n_candidates = min(max_results * 3, config.max_results)
                    # A3.3 — HyDE injection point. When ``query_embedding``
                    # is supplied by :meth:`query`, pass it as
                    # ``query_embeddings`` and skip ChromaDB's own embed
                    # step. When it is ``None`` (feature off or fail-open
                    # branch taken) fall back to ``query_texts=[query_text]``
                    # so the historical byte-identical semantic path runs.
                    if query_embedding is not None:
                        results = self.collection.query(
                            query_embeddings=[query_embedding],
                            n_results=n_candidates,
                            where=where_filter,
                            include=["documents", "metadatas", "distances"],
                        )
                    else:
                        results = self.collection.query(
                            query_texts=[query_text],
                            n_results=n_candidates,
                            where=where_filter,
                            include=["documents", "metadatas", "distances"],
                        )
                    if results["ids"] and results["ids"][0]:
                        for i, chunk_id in enumerate(results["ids"][0]):
                            r[chunk_id] = {
                                "rank": i + 1,
                                "distance": results["distances"][0][i] if results["distances"] else 0,
                                "document": results["documents"][0][i] if results["documents"] else "",
                                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            }
                except Exception as e:
                    log.warning("[WARN] Semantic search failed: %s", e)
            return r

        def _do_bm25():
            r = {}
            if hybrid_alpha < 1.0:
                try:
                    bm25_top_k = max_results * (20 if where_filter else 3)
                    bm25_hits = self.bm25_index.search(query_text, top_k=bm25_top_k)

                    if where_filter:
                        chunk_ids = [chunk_id for chunk_id, _ in bm25_hits]
                        metadata_by_id = {}
                        if chunk_ids:
                            fetched = self.collection.get(ids=chunk_ids, include=["metadatas"])
                            metadata_by_id = dict(zip(fetched.get("ids", []), fetched.get("metadatas", [])))

                        bm25_hits = [
                            (chunk_id, bm25_score)
                            for chunk_id, bm25_score in bm25_hits
                            if _matches_category(metadata_by_id.get(chunk_id, {}))
                        ]

                    for rank, (chunk_id, bm25_score) in enumerate(bm25_hits[: max_results * 3]):
                        r[chunk_id] = {"rank": rank + 1, "bm25_score": bm25_score}
                except Exception as e:
                    log.warning("[WARN] BM25 search failed: %s", e)
            return r

        # Run both in parallel when hybrid mode
        if 0 < hybrid_alpha < 1.0:
            with tracer.start_as_current_span("knowledge_rag.query.parallel_retrieval"):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    sem_future = executor.submit(_do_semantic)
                    bm25_future = executor.submit(_do_bm25)
                    semantic_results = sem_future.result()
                    bm25_results = bm25_future.result()
        else:
            with tracer.start_as_current_span("knowledge_rag.query.semantic"):
                semantic_results = _do_semantic()
            with tracer.start_as_current_span("knowledge_rag.query.bm25"):
                bm25_results = _do_bm25()

        # A2.4 — Pluggable fusion. Default ``rrf`` is byte-identical to the
        # previous hardcoded RRF-K=60 path (missing branch = rank 1000 fallback,
        # metadata path bonus stacked on top). ``combsum`` / ``combmnz`` /
        # ``weighted`` are opt-in via ``search.fusion.strategy`` in config.yaml
        # or the ``fusion=`` MCP / CLI override. The downstream ``rrf_score``
        # dict key is preserved regardless of strategy so reranker / MMR / the
        # public ``raw_rrf_score`` payload field never break their contract —
        # renaming it here would ripple into the response schema and every
        # downstream consumer.
        combined_scores: Dict[str, Dict] = {}
        all_chunk_ids = set(semantic_results.keys()) | set(bm25_results.keys())

        with tracer.start_as_current_span("knowledge_rag.query.fusion") as _fusion_span:
            _fusion_span.set_attribute("fusion.semantic_hits", len(semantic_results))
            _fusion_span.set_attribute("fusion.bm25_hits", len(bm25_results))
            _fusion_span.set_attribute("fusion.union_size", len(all_chunk_ids))
            _fusion_span.set_attribute("fusion.strategy", fusion_name)

            # Build per-branch RetrievedResult inputs. Each candidate shows up
            # in exactly one list — the strategy unions by doc_id internally.
            sem_inputs = [
                RetrievedResult(
                    doc_id=cid,
                    semantic_rank=data.get("rank"),
                    bm25_rank=None,
                    semantic_score=1.0 - float(data.get("distance", 0.0)),
                    bm25_score=None,
                    metadata=data.get("metadata", {}),
                )
                for cid, data in semantic_results.items()
            ]
            bm25_inputs = [
                RetrievedResult(
                    doc_id=cid,
                    semantic_rank=None,
                    bm25_rank=data.get("rank"),
                    semantic_score=None,
                    bm25_score=float(data.get("bm25_score", 0.0)),
                    metadata={},
                )
                for cid, data in bm25_results.items()
            ]

            strategy = get_strategy(fusion_name)
            fused_pairs = strategy.fuse(
                sem_inputs,
                bm25_inputs,
                alpha=hybrid_alpha,
                weights=config.fusion_weights or None,
                query=query_text,
            )
            fused_score_by_id: Dict[str, float] = dict(fused_pairs)

            # Materialise combined_scores in fused order, hydrating docs that
            # only came from the BM25 branch (semantic branch already carries
            # document + metadata + distance in its dict).
            for chunk_id in all_chunk_ids:
                if chunk_id in semantic_results:
                    data = semantic_results[chunk_id]
                else:
                    try:
                        fetched = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
                        if (
                            not fetched["documents"]
                            or not fetched["metadatas"]
                            or not fetched["documents"][0]
                            or not fetched["metadatas"][0]
                        ):
                            continue
                        data = {
                            "document": fetched["documents"][0],
                            "metadata": fetched["metadatas"][0],
                            "distance": 0,
                        }
                    except Exception:
                        continue

                if not _matches_category(data.get("metadata", {})):
                    continue

                fused_score = fused_score_by_id.get(chunk_id, 0.0)
                combined_scores[chunk_id] = {
                    # Legacy public field name — kept as ``rrf_score`` even when
                    # a non-RRF strategy is active so the downstream pipeline
                    # and the ``raw_rrf_score`` response field stay stable.
                    "rrf_score": fused_score + _metadata_path_score(query_text, data.get("metadata", {})),
                    "semantic_rank": semantic_results.get(chunk_id, {}).get("rank")
                    if chunk_id in semantic_results
                    else None,
                    "bm25_rank": bm25_results.get(chunk_id, {}).get("rank") if chunk_id in bm25_results else None,
                    "document": data.get("document", ""),
                    "metadata": data.get("metadata", {}),
                    "distance": data.get("distance", 0),
                }

        # Sort by fused score — take extra candidates for reranker
        reranker_k = max_results * config.reranker_top_k_multiplier if config.reranker_enabled else max_results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)[:reranker_k]

        # Cross-encoder reranking
        if config.reranker_enabled and sorted_results:
            with tracer.start_as_current_span("knowledge_rag.query.rerank") as _rerank_span:
                _rerank_span.set_attribute("rerank.candidates", len(sorted_results))
                rerank_input = []
                for chunk_id, data in sorted_results:
                    rerank_input.append(
                        {
                            "chunk_id": chunk_id,
                            "document": data["document"],
                            "metadata": data["metadata"],
                            "rrf_score": data["rrf_score"],
                            "semantic_rank": data["semantic_rank"],
                            "bm25_rank": data["bm25_rank"],
                            "distance": data["distance"],
                        }
                    )
                reranked = self.reranker.rerank(query_text, rerank_input, top_k=max_results)
                sorted_results = [(d["chunk_id"], d) for d in reranked]

        # Normalize scores and format
        if sorted_results:
            raw_scores = [data.get("reranker_score", data.get("rrf_score", 0)) for _, data in sorted_results]
            max_score = max(raw_scores) if raw_scores else 1
            min_score = min(raw_scores) if raw_scores else 0
            score_range = max_score - min_score
        else:
            score_range = 0

        # MMR: Maximal Marginal Relevance — diversify results to reduce redundancy.
        # ``_current_query_text`` is a scratch attribute that ``_apply_mmr`` reads
        # to compute the query embedding. Kept off the method signature so the
        # existing regression tests that monkey-patch ``_apply_mmr`` with a
        # three-arg lambda still work (see tests/test_pr98_regression.py).
        if config.mmr_enabled and len(sorted_results) > max_results:
            self._current_query_text = query_text
            try:
                with tracer.start_as_current_span("knowledge_rag.query.mmr") as _mmr_span:
                    _mmr_span.set_attribute("mmr.input", len(sorted_results))
                    _mmr_span.set_attribute("mmr.top_k", max_results)
                    _mmr_span.set_attribute("mmr.lambda", config.mmr_lambda)
                    sorted_results = self._apply_mmr(sorted_results, max_results, lambda_param=config.mmr_lambda)
            finally:
                self._current_query_text = None

        formatted = []
        for chunk_id, data in sorted_results[:max_results]:
            metadata = data.get("metadata", {})
            s_rank = data.get("semantic_rank")
            b_rank = data.get("bm25_rank")

            if s_rank and b_rank:
                search_method = "hybrid"
            elif s_rank:
                search_method = "semantic"
            else:
                search_method = "keyword"

            raw = data.get("reranker_score", data.get("rrf_score", 0))
            normalized_score = (raw - min_score) / score_range if score_range > 0 else 1.0

            formatted_entry = {
                "content": data.get("document", ""),
                "source": metadata.get("source", ""),
                "filename": metadata.get("filename", ""),
                "category": metadata.get("category", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "score": round(normalized_score, 4),
                "raw_rrf_score": round(data.get("rrf_score", 0), 6),
                "reranker_score": round(data.get("reranker_score", 0), 6) if "reranker_score" in data else None,
                "semantic_rank": s_rank,
                "bm25_rank": b_rank,
                "search_method": search_method,
                "keywords": metadata.get("keywords", "").split(","),
                "routed_by": routed_category if routed_category else "none",
                # Prompt-injection defense, layer 3 — evidence marker.
                # True when this chunk came from an untrusted fetch, so the
                # consuming LLM can weight it as data, never as instruction.
                "external_source": bool(metadata.get("external_source", False)),
                "external_source_uri": metadata.get("external_source_uri", ""),
                "content_hash": metadata.get("content_hash", ""),
                # M4.7 — Confidence label. ``extracted`` (structural boundary),
                # ``inferred`` (fixed-size window), or ``unverified`` (scanned
                # PDF / encoding failure). Explicitly ``None`` when the chunk
                # was indexed before M4.7 so callers can distinguish "unknown"
                # from a real classification.
                "source_confidence": metadata.get("source_confidence") or None,
            }

            # A3.7 — Parent Document Retrieval. Only propagate parent
            # metadata when the chunk actually carries a parent linkage;
            # this keeps the output shape unchanged for corpora indexed
            # under the historical flat chunker.
            parent_id = metadata.get("parent_id")
            if parent_id:
                formatted_entry["parent_id"] = parent_id
                formatted_entry["parent_content"] = metadata.get("parent_content", "")
                formatted_entry["parent_start_char"] = metadata.get("parent_start_char", 0)
                formatted_entry["parent_end_char"] = metadata.get("parent_end_char", 0)

            # A3.6 — Contextual Chunking. When a chunk carries the
            # ``contextual`` marker, expose it on the result so the caller
            # knows the ``content`` field starts with an LLM-generated
            # context sentence (not verbatim source text). Only stamped
            # when true so pre-A3.6 corpora keep an identical output shape.
            if metadata.get("contextual"):
                formatted_entry["contextual"] = True

            formatted.append(formatted_entry)

        # A3.7 — Parent Document Retrieval (Small-to-Big).
        #
        # When the corpus was indexed with hierarchical chunking, swap each
        # small chunk's content for its parent's, giving the caller the
        # broader context window without hurting retrieval precision.
        # Adjacent-chunk expansion is skipped in this mode because the
        # parent already spans multiple children.
        #
        # ``is True`` (not truthy) so tests that patch ``config`` with a
        # bare ``MagicMock`` still get the historical adjacent-chunk path.
        if getattr(config, "parent_document_enabled", False) is True:
            with tracer.start_as_current_span("knowledge_rag.query.parent_expansion"):
                formatted = self._expand_to_parents(formatted)
        else:
            # Adjacent Chunk Retrieval — expand content with surrounding chunks for context
            with tracer.start_as_current_span("knowledge_rag.query.adjacent_chunks"):
                formatted = self._expand_with_adjacent_chunks(formatted)

        # A2.4 — mirror the fusion-aware cache key used on the get() side so
        # entries stored by different strategies never overwrite each other.
        # A3.3 — also fold the HyDE-vs-raw discriminator into the key so a
        # HyDE-augmented ranking never overwrites (or is overwritten by) a
        # raw-embedding ranking for the same effective query.
        hyde_key = "hyde" if query_embedding is not None else "raw"
        put_cache_key = f"{query_text}\x1ffusion={fusion_name}\x1femb={hyde_key}"
        self.query_cache.put(put_cache_key, max_results, category_filter, hybrid_alpha, formatted)
        try:
            _root_span.set_attribute("query.result_count", len(formatted))
        except Exception:
            pass
        return formatted

    def _expand_with_adjacent_chunks(self, results: List[Dict], window: int = 1) -> List[Dict]:
        """
        Expand each result with adjacent chunks for broader context.

        Uses a single batched ChromaDB fetch for all adjacent chunks across all
        results, plus O(1) reverse lookup for doc_id resolution.

        Args:
            results: Formatted search results
            window: Number of adjacent chunks to fetch on each side (default: 1)

        Returns:
            Results with expanded content field
        """
        if not results:
            return results

        all_adj_ids: List[str] = []
        result_adj_map: List[Tuple[int, int, List[str]]] = []

        for i, result in enumerate(results):
            source = result.get("source", "")
            chunk_idx = result.get("chunk_index", 0)
            if not source or chunk_idx is None:
                continue

            doc_id = self._source_to_docid.get(str(Path(source).resolve()))
            if not doc_id:
                continue

            adj_ids: List[str] = []
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                adj_id = f"{doc_id}_{chunk_idx + offset}"
                adj_ids.append(adj_id)
                all_adj_ids.append(adj_id)

            if adj_ids:
                result_adj_map.append((i, chunk_idx, adj_ids))

        if not all_adj_ids:
            return results

        try:
            adj_data = self.collection.get(ids=all_adj_ids, include=["documents"])
            fetched = dict(zip(adj_data["ids"], adj_data["documents"]))
        except Exception:
            return results

        for result_idx, chunk_idx, adj_ids in result_adj_map:
            parts_before: List[str] = []
            parts_after: List[str] = []
            for adj_id in adj_ids:
                doc = fetched.get(adj_id)
                if doc:
                    idx = int(adj_id.split("_")[-1])
                    if idx < chunk_idx:
                        parts_before.append(doc)
                    else:
                        parts_after.append(doc)
            if parts_before or parts_after:
                expanded = "\n\n".join(parts_before + [results[result_idx]["content"]] + parts_after)
                results[result_idx]["content"] = expanded
                results[result_idx]["context_expanded"] = True

        return results

    def _expand_to_parents(self, results: List[Dict]) -> List[Dict]:
        """Swap each result's content for its parent chunk (A3.7).

        Implements Parent Document / Small-to-Big Retrieval. When a
        corpus is indexed via :func:`chunk_hierarchical`, every child
        chunk carries its parent's content inline in the ``parent_content``
        metadata field. This method reads that field and replaces the
        result's ``content`` with the parent's, giving the caller the
        broader context window the parent represents while preserving
        the ranking earned by the precise small-chunk match.

        Parent content lives inline in metadata by design, so this
        expansion is a pure in-memory swap — the historical
        :meth:`_expand_with_adjacent_chunks` needed a batched
        ``collection.get`` round-trip, this one needs none. That keeps
        parent expansion O(1) round-trips regardless of ``len(results)``,
        which is the property the "batch fetch" pattern is meant to
        guarantee.

        Chunks that do not carry ``parent_content`` (pre-A3.7 corpora,
        chunks from documents indexed under the flat chunker, or any
        third-party parser that skipped the hierarchical path) pass
        through untouched. This is what makes the feature safe to enable
        against a partially-migrated index.

        Args:
            results: Formatted search results from :meth:`query`. Modified
                in place — the return value is the same list.

        Returns:
            List[Dict]: Results with ``content`` swapped for parent
            content where available. Each expanded result gains a
            ``parent_expanded`` flag; transient fields
            (``parent_content``, ``parent_start_char``, ``parent_end_char``)
            are popped so the payload does not carry the parent text
            twice.
        """
        if not results:
            return results

        for result in results:
            parent_content = result.pop("parent_content", "")
            # Transient offsets are useful during expansion for potential
            # future callers (e.g. an evaluator that wants to know how far
            # the parent extends), but keeping them in the returned dict
            # duplicates information already implicit in the parent id.
            # Pop them to keep the output shape minimal.
            result.pop("parent_start_char", None)
            result.pop("parent_end_char", None)

            if parent_content:
                result["content"] = parent_content
                result["parent_expanded"] = True

        return results

    def _route_by_keywords(self, query_text: str) -> Optional[str]:
        """Weighted keyword routing with word boundaries."""
        query_lower = query_text.lower()
        category_scores: Dict[str, Tuple[int, List[str]]] = {}

        for category, keywords in config.keyword_routes.items():
            matches = []
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if " " in keyword_lower:
                    if keyword_lower in query_lower:
                        matches.append(keyword)
                else:
                    pattern = r"\b" + re.escape(keyword_lower) + r"\b"
                    if re.search(pattern, query_lower):
                        matches.append(keyword)

            if matches:
                category_scores[category] = (len(matches), matches)

        if not category_scores:
            return None

        best_category = max(category_scores.keys(), key=lambda c: category_scores[c][0])
        return best_category

    def _apply_mmr(
        self, results: List[Tuple[str, Dict]], top_k: int, lambda_param: float = 0.7
    ) -> List[Tuple[str, Dict]]:
        """Diversify the top-k candidates with embedding-based MMR.

        Uses cosine similarity of ChromaDB embeddings for both the relevance
        term (query vs. candidate) and the redundancy penalty (candidate vs.
        already-selected candidates). The signature intentionally does not
        take ``query_text`` — the caller stashes it on
        ``self._current_query_text`` right before invoking this method, so
        existing regression tests that monkey-patch ``_apply_mmr`` with a
        three-arg lambda (see ``tests/test_pr98_regression.py``) still work.

        Fallback: when the query text is unavailable or the embeddings cannot
        be fetched from ChromaDB, the method degrades to the legacy Jaccard
        MMR (``_apply_mmr_jaccard``) so search never dies just because MMR
        cannot access embeddings.

        Args:
            results: Candidates as ``(chunk_id, data)`` pairs, already ordered
                by the caller (reranker or RRF output). Position 0 is treated
                as the top pick and always kept as the first output.
            top_k: Number of items to keep after diversification.
            lambda_param: Relevance/diversity balance in ``[0.0, 1.0]``.
                ``1.0`` = pure relevance, ``0.0`` = pure diversity.

        Returns:
            list[tuple[str, dict]]: Selected candidates in MMR order, with
            their original ``data`` dicts preserved.
        """
        if len(results) <= top_k:
            return results

        query_text = getattr(self, "_current_query_text", None)
        if not query_text:
            # No query text — cannot compute query embedding. Fall back to
            # the legacy Jaccard implementation instead of crashing the search.
            return self._apply_mmr_jaccard(results, top_k, lambda_param)

        chunk_ids = [chunk_id for chunk_id, _ in results]

        try:
            fetched = self.collection.get(ids=chunk_ids, include=["embeddings"])
        except Exception as exc:  # pragma: no cover — chroma failure path
            log.warning("[WARN] MMR embedding fetch failed (%s); falling back to Jaccard", exc)
            return self._apply_mmr_jaccard(results, top_k, lambda_param)

        fetched_ids = fetched.get("ids") or []
        fetched_embs = fetched.get("embeddings")
        # ChromaDB returns ``None`` when the collection has no embeddings and
        # a numpy ndarray otherwise. ``len()`` on both paths tells us whether
        # anything usable came back.
        if fetched_embs is None or len(fetched_embs) == 0 or len(fetched_ids) == 0:
            log.warning("[WARN] MMR embeddings unavailable; falling back to Jaccard")
            return self._apply_mmr_jaccard(results, top_k, lambda_param)

        id_to_emb = {cid: emb for cid, emb in zip(fetched_ids, fetched_embs)}

        candidate_ids: List[str] = []
        candidate_embeddings: List[Any] = []
        candidate_scores: List[float] = []
        data_by_id: Dict[str, Dict] = {}
        for chunk_id, data in results:
            emb = id_to_emb.get(chunk_id)
            if emb is None:
                # Missing embedding for a candidate — skip it for MMR (would
                # otherwise force a shape mismatch). It is still available via
                # the fallback path below when nothing survives.
                continue
            candidate_ids.append(chunk_id)
            candidate_embeddings.append(emb)
            candidate_scores.append(float(data.get("reranker_score", data.get("rrf_score", 0.0)) or 0.0))
            data_by_id[chunk_id] = data

        if not candidate_ids:
            log.warning("[WARN] MMR: no candidate embeddings survived; falling back to Jaccard")
            return self._apply_mmr_jaccard(results, top_k, lambda_param)

        try:
            query_embeddings = self.embed_fn.embed_query(query_text)
            # ``embed_query`` returns list[list[float]] (one row per input).
            if (
                isinstance(query_embeddings, list)
                and query_embeddings
                and isinstance(query_embeddings[0], (list, tuple))
            ):
                query_embedding = query_embeddings[0]
            else:
                query_embedding = query_embeddings
        except Exception as exc:
            log.warning("[WARN] MMR query embed failed (%s); falling back to Jaccard", exc)
            return self._apply_mmr_jaccard(results, top_k, lambda_param)

        try:
            picked = apply_mmr(
                query_embedding=query_embedding,
                candidate_ids=candidate_ids,
                candidate_embeddings=candidate_embeddings,
                candidate_scores=candidate_scores,
                top_k=top_k,
                lambda_param=lambda_param,
            )
        except Exception as exc:
            log.warning("[WARN] MMR reordering failed (%s); falling back to Jaccard", exc)
            return self._apply_mmr_jaccard(results, top_k, lambda_param)

        return [(cid, data_by_id[cid]) for cid, _ in picked]

    def _apply_mmr_jaccard(
        self, results: List[Tuple[str, Dict]], top_k: int, lambda_param: float = 0.7
    ) -> List[Tuple[str, Dict]]:
        """Legacy Jaccard-based MMR — used only as a fallback.

        Kept for the rare case where embeddings cannot be fetched from
        ChromaDB (schema drift, get-with-embeddings raising, embed_query
        failing). The signal is much weaker than embedding cosine and this
        path should not run during normal operation.
        """
        if len(results) <= top_k:
            return results

        def jaccard_sim(a: str, b: str) -> float:
            tokens_a = set(a.lower().split())
            tokens_b = set(b.lower().split())
            if not tokens_a or not tokens_b:
                return 0.0
            return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

        selected = [results[0]]  # first result always kept (highest score)
        remaining = list(results[1:])

        while len(selected) < top_k and remaining:
            best_idx = 0
            best_mmr = -1.0

            for i, (_, data) in enumerate(remaining):
                relevance = data.get("reranker_score", data.get("rrf_score", 0))
                doc_text = data.get("document", "")
                max_sim = max(jaccard_sim(doc_text, sel_data.get("document", "")) for _, sel_data in selected)
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    # =========================================================================
    # Document Retrieval & Management
    # =========================================================================

    def get_document(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Get full document content by filepath.

        The path is containment-checked before the file is opened. Without it
        this method is an arbitrary-file-read primitive for any MCP client:
        every supported suffix (``.json``, ``.md``, ``.py``, ``.csv``, ``.xml``,
        ``.docx``, ...) anywhere on the host would be readable, including MCP
        client configuration files holding API keys (CWE-22).

        Args:
            filepath: Path relative to ``documents_dir``, or an absolute path
                inside it as returned by ``list_documents()``.

        Returns:
            Optional[Dict[str, Any]]: Document payload, or ``None`` when the
            path escapes the corpus or cannot be parsed.
        """
        try:
            filepath = validate_path_within(config.documents_dir, filepath)
        except PathEscapeError as e:
            log.warning("[WARN] Rejected out-of-sandbox document read: %s (%s)", filepath, e)
            return None

        try:
            doc = self.parser.parse_file(filepath)
            if doc:
                return {
                    "content": doc.content,
                    "source": str(doc.source),
                    "filename": doc.filename,
                    "category": doc.category,
                    "format": doc.format,
                    "metadata": doc.metadata,
                    "keywords": doc.keywords,
                    "chunk_count": len(doc.chunks),
                }
        except Exception as e:
            log.error("[ERROR] Failed to read document %s: %s", filepath, e)
        return None

    @_traced_span("knowledge_rag.add_document")
    def add_document_from_content(
        self, content: str, filepath: str, category: str, external_source: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add a new document from raw content string. Saves to disk and indexes.

        Args:
            content: Full text of the document.
            filepath: Destination path relative to ``documents_dir``. Validated
                for containment — ``..`` segments, absolute paths and symlinks
                pointing outside the corpus are rejected (CWE-22).
            category: Category assigned to the document and all of its chunks.
            external_source: Origin URL/path when the content was fetched from
                an untrusted source. Triggers the prompt-injection defense:
                sentinels are neutralized and the body is fenced in a
                provenance marker before it ever reaches disk or the index.

        Returns:
            dict: ``{"chunks_added", "dedup_skipped", "category", "filepath"}``
            on success, or ``{"error": ...}`` on failure.
        """
        try:
            full_path = validate_path_within(config.documents_dir, filepath)
        except PathEscapeError as e:
            return {"error": f"Invalid filepath: {e}"}

        if external_source:
            content = sanitize_external_content(content, external_source)

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return self._index_new_file(full_path, category)

    def add_document_from_file(self, source_path: Any, filepath: str, category: str) -> Dict[str, Any]:
        """
        Copy an existing file into the documents directory and index it.

        Binary-safe counterpart of :meth:`add_document_from_content`: the source
        file is copied byte-for-byte instead of being round-tripped through a text
        decode, so PDF/DOCX/XLSX/PPTX sources survive intact. Used by the
        ``knowledge-rag add <path>`` CLI subcommand.

        Args:
            source_path: Path to the file to ingest (must already exist on disk)
            filepath: Destination path relative to the documents directory
                (e.g. ``"security/incident-report.md"``)
            category: Category assigned to the resulting document

        Returns:
            dict: Same shape as :meth:`add_document_from_content` — either
            ``{"chunks_added", "dedup_skipped", "category", "filepath"}`` on
            success or ``{"error": ...}`` on failure.
        """
        import shutil

        source = Path(source_path)
        if not source.is_file():
            return {"error": f"Source file not found: {source}"}

        try:
            full_path = validate_path_within(config.documents_dir, filepath)
        except PathEscapeError as e:
            return {"error": f"Invalid filepath: {e}"}

        full_path.parent.mkdir(parents=True, exist_ok=True)

        if source.resolve() != full_path.resolve():
            try:
                shutil.copy2(source, full_path)
            except OSError as e:
                return {"error": f"Failed to copy {source} into the documents directory: {e}"}

        return self._index_new_file(full_path, category)

    def _index_new_file(self, full_path: Path, category: str) -> Dict[str, Any]:
        """
        Parse, index and register a file that is already present on disk.

        Shared tail of :meth:`add_document_from_content` and
        :meth:`add_document_from_file`. Updates the on-disk metadata, invalidates
        the query cache and rebuilds the BM25 index so the document is searchable
        immediately.

        Args:
            full_path: Absolute path of the file inside the documents directory
            category: Category to assign to the document and all of its chunks

        Returns:
            dict: ``{"chunks_added", "dedup_skipped", "category", "filepath"}``
            on success, or ``{"error": ...}`` when the file could not be parsed.
        """
        doc = self.parser.parse_file(full_path)
        if not doc:
            return {"error": "Failed to parse document content"}

        doc.category = category
        for chunk in doc.chunks:
            chunk.metadata["category"] = category

        chunks_added, dedup_skipped = self._index_document(doc)

        try:
            file_stat = full_path.stat()
            file_mtime = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            file_size = file_stat.st_size
        except OSError:
            file_mtime = datetime.now().isoformat()
            file_size = 0

        self._indexed_docs[doc.id] = {
            "source": str(full_path),
            "category": category,
            "format": doc.format,
            "chunks": chunks_added,
            "keywords": doc.keywords,
            "indexed_at": datetime.now().isoformat(),
            "file_mtime": file_mtime,
            "file_size": file_size,
        }
        self._source_to_docid[str(full_path.resolve())] = doc.id
        self._save_metadata()
        self.query_cache.invalidate()
        self.bm25_index.build_index()

        return {
            "chunks_added": chunks_added,
            "dedup_skipped": dedup_skipped,
            "category": category,
            "filepath": str(full_path),
        }

    @_traced_span("knowledge_rag.update_document")
    def update_document_content(self, filepath: str, content: str) -> Dict[str, Any]:
        """Update an existing document. Removes old chunks and re-indexes.

        Args:
            filepath: Path to the document. Containment-checked against
                ``documents_dir`` — without it, an MCP client could overwrite
                any writable file on the host (CWE-22).
            content: Replacement text for the whole document.

        Returns:
            dict: Chunk deltas on success, or ``{"error": ...}`` on failure.
        """
        try:
            filepath = validate_path_within(config.documents_dir, filepath)
        except PathEscapeError as e:
            return {"error": f"Invalid filepath: {e}"}

        if not filepath.exists():
            return {"error": f"File not found: {filepath}"}

        # Resolve to absolute for consistent comparison with stored metadata
        filepath_resolved = str(filepath.resolve())

        doc_id = self._source_to_docid.get(filepath_resolved)

        old_chunks_removed = 0
        if doc_id:
            old_chunks_removed = self._remove_document_chunks(doc_id)
            self._source_to_docid.pop(filepath_resolved, None)
            del self._indexed_docs[doc_id]

        filepath.write_text(content, encoding="utf-8")

        doc = self.parser.parse_file(filepath)
        if not doc:
            self._save_metadata()
            return {"error": "Failed to parse updated content", "old_chunks_removed": old_chunks_removed}

        new_chunks_added, dedup_skipped = self._index_document(doc)

        try:
            file_stat = filepath.stat()
            file_mtime = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            file_size = file_stat.st_size
        except OSError:
            file_mtime = datetime.now().isoformat()
            file_size = 0

        self._indexed_docs[doc.id] = {
            "source": str(filepath),
            "category": doc.category,
            "format": doc.format,
            "chunks": new_chunks_added,
            "keywords": doc.keywords,
            "indexed_at": datetime.now().isoformat(),
            "file_mtime": file_mtime,
            "file_size": file_size,
        }
        self._source_to_docid[str(filepath.resolve())] = doc.id
        self._save_metadata()
        self.query_cache.invalidate()
        self.bm25_index.build_index()

        return {
            "old_chunks_removed": old_chunks_removed,
            "new_chunks_added": new_chunks_added,
            "dedup_skipped": dedup_skipped,
            "filepath": str(filepath),
        }

    @_traced_span("knowledge_rag.remove_document")
    def remove_document_by_path(self, filepath: str, delete_file: bool = False) -> Dict[str, Any]:
        """Remove a document from the index. Optionally delete from disk.

        Args:
            filepath: Path to the document. Containment-checked against
                ``documents_dir`` — with ``delete_file=True`` an unchecked path
                would be an arbitrary-file-delete primitive (CWE-22).
            delete_file: Also unlink the file from disk.

        Returns:
            dict: Removal summary, or ``{"error": ...}`` on failure.
        """
        try:
            resolved = validate_path_within(config.documents_dir, filepath)
        except PathEscapeError as e:
            return {"error": f"Invalid filepath: {e}"}

        filepath_resolved = str(resolved)

        doc_id = self._source_to_docid.get(filepath_resolved)

        if not doc_id:
            return {"error": f"Document not found in index: {filepath}"}

        chunks_removed = self._remove_document_chunks(doc_id)
        self._source_to_docid.pop(filepath_resolved, None)
        del self._indexed_docs[doc_id]

        if delete_file:
            try:
                resolved.unlink(missing_ok=True)
            except Exception as e:
                log.warning("[WARN] Failed to delete file %s: %s", filepath, e)

        self._save_metadata()
        self.query_cache.invalidate()

        return {"chunks_removed": chunks_removed, "filepath": filepath_resolved, "file_deleted": delete_file}

    @_traced_span("knowledge_rag.add_from_url")
    def add_from_url(self, url: str, category: str, title: str = None) -> Dict[str, Any]:
        """Fetch URL content, convert to markdown, and add to knowledge base.

        The fetched body is untrusted by definition, so it is handed to
        :meth:`add_document_from_content` with ``external_source`` set. That
        runs the full prompt-injection defense (OWASP LLM01:2025) before the
        text reaches disk or the vector index.

        Args:
            url: ``http://`` or ``https://`` URL to fetch.
            category: Category assigned to the resulting document.
            title: Optional title; auto-detected from ``<title>`` when omitted.

        Returns:
            dict: Indexing summary, or ``{"error": ...}`` on failure.
        """
        import requests
        from bs4 import BeautifulSoup

        # Validate URL scheme (only http/https allowed)
        if not url.startswith(("http://", "https://")):
            return {"error": "Only http:// and https:// URLs are supported"}

        try:
            response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (knowledge-rag-ingester)"})
            response.raise_for_status()
        except Exception as e:
            return {"error": f"Failed to fetch URL: {e}"}

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        if not title:
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-1]

        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = f"# {title}\n\nSource: {url}\n\n" + "\n\n".join(lines)

        safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "-").lower()[:60]
        filename = f"{safe_title}.md"
        filepath = f"{category}/{filename}"

        return self.add_document_from_content(clean_text, filepath, category, external_source=url)

    @_traced_span("knowledge_rag.search_similar")
    def search_similar(self, filepath: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document using embedding similarity.

        Args:
            filepath: Reference document path, containment-checked against
                ``documents_dir``.
            max_results: Maximum number of similar documents to return.

        Returns:
            List[Dict[str, Any]]: Similar documents, or an empty list when the
            path escapes the corpus or is not indexed.
        """
        try:
            filepath_resolved = str(validate_path_within(config.documents_dir, filepath))
        except PathEscapeError:
            return []

        doc_id = self._source_to_docid.get(filepath_resolved)

        if not doc_id:
            return []

        try:
            results = self.collection.get(where={"doc_id": doc_id}, include=["embeddings"], limit=1)
            if not results["ids"] or not results.get("embeddings"):
                return []
            embeddings = results.get("embeddings", [])
            if not embeddings:
                return []
            query_embedding = embeddings[0]
        except Exception:
            return []

        try:
            similar = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results + 20,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        if not similar["ids"] or not similar["ids"][0]:
            return []

        seen_sources = set()
        output = []
        for i, chunk_id in enumerate(similar["ids"][0]):
            meta = similar["metadatas"][0][i]
            source = meta.get("source", "")

            if meta.get("doc_id") == doc_id:
                continue
            if source in seen_sources:
                continue
            seen_sources.add(source)

            distance = similar["distances"][0][i] if similar["distances"] else 0
            similarity = max(0, 1.0 - distance)

            output.append(
                {
                    "source": source,
                    "filename": meta.get("filename", ""),
                    "category": meta.get("category", ""),
                    "similarity": round(similarity, 4),
                    "preview": (similar["documents"][0][i] or "")[:200],
                }
            )

            if len(output) >= max_results:
                break

        return output

    @staticmethod
    def _expected_paths(test_case: Dict[str, Any]) -> List[str]:
        """
        Collect the ground-truth paths declared by a single evaluation case.

        Accepts the historical single-path key and a plural form, so a case can
        declare several relevant documents. Precision@k is only meaningful when
        more than one document may legitimately answer a query.

        Args:
            test_case: One entry from the ``test_cases`` payload. Recognized keys
                are ``expected_filepath`` (str) and ``expected_filepaths`` (list).

        Returns:
            list[str]: Non-empty ground-truth path fragments, order preserved.
        """
        paths: List[str] = []
        single = test_case.get("expected_filepath", "")
        if isinstance(single, str) and single:
            paths.append(single)
        plural = test_case.get("expected_filepaths")
        if isinstance(plural, list):
            for item in plural:
                if isinstance(item, str) and item and item not in paths:
                    paths.append(item)
        return paths

    def evaluate_retrieval(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Measure retrieval quality over a set of query / ground-truth pairs.

        Runs each query through the full hybrid pipeline and scores the top-5:

        - **MRR@5** — mean of ``1 / rank`` of the first relevant hit (0 when absent).
        - **Recall@5** — fraction of queries with at least one relevant hit in the top-5.
        - **Precision@5** — ``|relevant ∩ top-5| / denominator``, where the
          denominator is ``min(5, len(results))``. Dividing by the number of
          results actually returned keeps the metric honest on small indexes: a
          corpus holding only three chunks would otherwise be capped at P@5=0.6
          for reasons that have nothing to do with ranking quality. With a
          normally-sized index the denominator is 5, matching the textbook
          definition. ``precision_denominator`` is reported per query so the
          number is always auditable.

        Args:
            test_cases: Entries with a ``query`` and ground truth given as
                ``expected_filepath`` (str) and/or ``expected_filepaths`` (list).

        Returns:
            dict: ``total_queries``, ``mrr_at_5``, ``recall_at_5``,
            ``precision_at_5`` and a ``per_query`` breakdown.
        """
        per_query = []
        mrr_sum = 0.0
        recall_sum = 0.0
        precision_sum = 0.0
        k = 5

        for tc in test_cases:
            query = tc.get("query", "")
            expected_paths = self._expected_paths(tc)

            results = self.query(query, max_results=k)
            top_k = results[:k]

            found_rank = None
            hits_in_top_k = 0
            for i, r in enumerate(top_k):
                source = r.get("source", "")
                if any(expected in source for expected in expected_paths):
                    hits_in_top_k += 1
                    if found_rank is None:
                        found_rank = i + 1

            rr = 1.0 / found_rank if found_rank else 0.0
            recall = 1.0 if found_rank else 0.0
            denominator = min(k, len(top_k))
            precision = hits_in_top_k / denominator if denominator else 0.0

            mrr_sum += rr
            recall_sum += recall
            precision_sum += precision

            per_query.append(
                {
                    "query": query,
                    "expected": expected_paths[0] if expected_paths else "",
                    "expected_all": expected_paths,
                    "found_at_rank": found_rank,
                    "reciprocal_rank": round(rr, 4),
                    "hits_at_5": hits_in_top_k,
                    "precision_denominator": denominator,
                    "precision_at_5": round(precision, 4),
                    "top_result": results[0]["source"] if results else "none",
                }
            )

        n = len(test_cases) if test_cases else 1
        return {
            "total_queries": len(test_cases),
            "mrr_at_5": round(mrr_sum / n, 4),
            "recall_at_5": round(recall_sum / n, 4),
            "precision_at_5": round(precision_sum / n, 4),
            "per_query": per_query,
        }

    # =========================================================================
    # Stats & Metadata
    # =========================================================================

    def list_categories(self) -> Dict[str, int]:
        """List all categories with document counts"""
        categories = {}
        for doc_info in list(self._indexed_docs.values()):
            cat = doc_info.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        return categories

    def list_documents(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        """List all indexed documents, optionally filtered by category"""
        docs = []
        for doc_id, info in list(self._indexed_docs.items()):
            if category and info.get("category") != category:
                continue
            docs.append(
                {
                    "id": doc_id,
                    "source": info.get("source", ""),
                    "category": info.get("category", ""),
                    "format": info.get("format", ""),
                    "chunks": info.get("chunks", 0),
                    "keywords": info.get("keywords", [])[:5],
                }
            )
        return docs

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics including background reindex progress."""
        stats = {
            "total_documents": len(self._indexed_docs),
            "total_chunks": self.collection.count(),
            "categories": self.list_categories(),
            "supported_formats": config.supported_formats,
            "embedding_model": config.embedding_model,
            "embedding_dim": config.embedding_dim,
            "reranker_model": config.reranker_model if config.reranker_enabled else "disabled",
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "query_cache": self.query_cache.stats(),
        }

        progress = self._reindex_progress
        if progress.get("active"):
            total = max(1, progress.get("total_files", 1))
            processed = progress.get("processed", 0)
            stats["reindex"] = {
                "active": True,
                "operation": progress.get("operation"),
                "progress": f"{processed}/{progress.get('total_files', 0)}",
                "percent": round(processed / total * 100),
                "indexed": progress.get("indexed", 0),
                "errors": progress.get("errors", 0),
                "started_at": progress.get("started_at"),
            }
        else:
            stats["reindex"] = {"active": False}

        return stats

    def get_reindex_status(self) -> Dict[str, Any]:
        """Get background reindex progress without computing full index stats."""
        progress = self._reindex_progress
        if progress.get("active"):
            total = max(1, progress.get("total_files", 1))
            processed = progress.get("processed", 0)
            return {
                "active": True,
                "operation": progress.get("operation"),
                "progress": f"{processed}/{progress.get('total_files', 0)}",
                "percent": round(processed / total * 100),
                "indexed": progress.get("indexed", 0),
                "skipped": progress.get("skipped", 0),
                "errors": progress.get("errors", 0),
                "started_at": progress.get("started_at"),
            }

        result: Dict[str, Any] = {"active": False}
        if "result" in progress:
            result["last_result"] = progress["result"]
        if "error" in progress:
            result["last_error"] = progress["error"]
        return result

    def _load_metadata(self) -> Dict[str, Dict]:
        """Load index metadata from disk"""
        if self._metadata_file.exists():
            try:
                return json.loads(self._metadata_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_metadata(self) -> None:
        """Save index metadata to disk"""
        self._metadata_file.parent.mkdir(parents=True, exist_ok=True)
        snapshot = dict(self._indexed_docs)
        self._metadata_file.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    def _build_source_lookup(self) -> Dict[str, str]:
        """Build reverse lookup from resolved source path to doc_id."""
        lookup: Dict[str, str] = {}
        for doc_id, info in list(self._indexed_docs.items()):
            src = info.get("source", "")
            if src:
                lookup[str(Path(src).resolve())] = doc_id
        return lookup
