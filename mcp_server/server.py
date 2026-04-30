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
    - Query expansion for security term synonyms
    - Incremental indexing (only re-indexes changed files)
    - Query caching with TTL for instant repeat queries
    - Chunk deduplication via content hashing
    - CRUD operations via MCP tools (add, update, remove docs)

Autor:   Lyon (Ailton Rocha)
Versao:  3.5.2
Data:    2026-04-16
"""

import hashlib
import json
import re
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ChromaDB
import chromadb

# FastEmbed for ONNX embeddings + reranker
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

# FastMCP
from mcp.server.fastmcp import FastMCP

# BM25 for keyword search (hybrid search)
from rank_bm25 import BM25Okapi
from watchdog.events import FileSystemEventHandler

# File watcher for auto-reindex
from watchdog.observers import Observer

# Local imports
from .config import config
from .ingestion import Document, DocumentParser

# =============================================================================
# QUERY CACHE
# =============================================================================


class QueryCache:
    """
    LRU cache with TTL for search queries.

    Avoids redundant searches when the same query is executed multiple times.
    Uses OrderedDict for O(1) LRU eviction.

    Args:
        max_size: Maximum number of cached entries (default: 100)
        ttl_seconds: Time-to-live for cache entries in seconds (default: 300)
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, max_results: int, category: Optional[str], hybrid_alpha: float) -> str:
        """Generate cache key from query parameters"""
        raw = f"{query}|{max_results}|{category}|{hybrid_alpha}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, query: str, max_results: int, category: Optional[str], hybrid_alpha: float) -> Optional[Any]:
        """Get cached result if exists and not expired"""
        key = self._make_key(query, max_results, category, hybrid_alpha)

        if key in self._cache:
            timestamp, result = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self._cache.move_to_end(key)
                self._hits += 1
                return result
            else:
                del self._cache[key]

        self._misses += 1
        return None

    def put(self, query: str, max_results: int, category: Optional[str], hybrid_alpha: float, result: Any) -> None:
        """Store result in cache"""
        key = self._make_key(query, max_results, category, hybrid_alpha)
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (time.time(), result)

    def invalidate(self) -> None:
        """Clear entire cache (call after reindex)"""
        self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics"""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{(self._hits / total * 100):.1f}%" if total > 0 else "0%",
        }


# =============================================================================
# EMBEDDINGS (FastEmbed — ONNX in-process)
# =============================================================================


class FastEmbedEmbeddings:
    """
    FastEmbed-based embedding function for ChromaDB (v1.4.0+ compatible).

    Uses ONNX Runtime in-process for embedding generation.
    No external server required (replaces Ollama).
    Model: BAAI/bge-small-en-v1.5 (384-dim, MTEB score 62.x)
    """

    @staticmethod
    def _setup_cuda_dll_paths():
        """Add NVIDIA CUDA 12 pip package DLL paths to os.environ['PATH'].

        When onnxruntime-gpu is installed alongside nvidia-cublas-cu12 etc.,
        the DLLs live under site-packages/nvidia/*/bin/ and onnxruntime can't
        find them unless they're on PATH. This is a no-op if the dirs don't exist.
        """
        import os
        import site

        site_dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
        nvidia_libs = [
            "nvidia/cublas/bin",
            "nvidia/cudnn/bin",
            "nvidia/cuda_runtime/bin",
            "nvidia/cufft/bin",
            "nvidia/curand/bin",
            "nvidia/cusolver/bin",
            "nvidia/cusparse/bin",
            "nvidia/nvjitlink/bin",
            "nvidia/cuda_nvrtc/bin",
        ]
        added = []
        for sp in site_dirs:
            for lib in nvidia_libs:
                p = os.path.join(sp, lib)
                if os.path.isdir(p) and p not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
                    added.append(lib.split("/")[1])
        if added:
            print(f"[INFO] CUDA DLL paths added for: {', '.join(dict.fromkeys(added))}")

    def __init__(self, model: str = None):
        self.model_name = model or config.embedding_model
        self._dim = config.embedding_dim
        kwargs = {"model_name": self.model_name, "cache_dir": str(config.models_cache_dir)}
        if config.gpu_acceleration:
            self._setup_cuda_dll_paths()
            kwargs["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print(f"[INFO] Loading embedding model: {self.model_name} ({self._dim}D) [GPU accelerated]...")
            try:
                self._model = TextEmbedding(**kwargs)
                print("[INFO] Embedding model loaded successfully [GPU]")
            except (ValueError, RuntimeError) as e:
                print(f"[WARN] GPU init failed ({e}), falling back to CPU...")
                kwargs["providers"] = ["CPUExecutionProvider"]
                self._model = TextEmbedding(**kwargs)
                print("[INFO] Embedding model loaded successfully [CPU fallback]")
        else:
            kwargs["providers"] = ["CPUExecutionProvider"]
            print(f"[INFO] Loading embedding model: {self.model_name} ({self._dim}D)...")
            self._model = TextEmbedding(**kwargs)
            print("[INFO] Embedding model loaded successfully")

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        ChromaDB embedding_function interface: __call__(input: List[str]) -> List[List[float]]
        FastEmbed.embed() returns a generator, so we consume it into a list.
        """
        if not input:
            return []

        try:
            embeddings = list(self._model.embed(input))
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            print(f"[WARN] Embedding failed: {e}")
            return [[0.0] * self._dim for _ in input]

    def name(self) -> str:
        """Return embedding function name (required by ChromaDB v1.4.0+)"""
        return f"fastembed-{self.model_name}"

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents (alias for __call__)"""
        return self(documents)

    def embed_query(self, input=None, **kwargs) -> List[List[float]]:
        """Embed query text(s) - returns list of embeddings"""
        if isinstance(input, list):
            texts = input
        elif input is not None:
            texts = [input]
        else:
            texts = [kwargs.get("query", "")]
        return self(texts)


# =============================================================================
# CROSS-ENCODER RERANKER
# =============================================================================


class CrossEncoderReranker:
    """
    Cross-encoder reranker using FastEmbed's TextCrossEncoder.

    Applied after hybrid RRF fusion to re-score the top candidates
    using a cross-encoder model that sees query+document pairs jointly.
    Dramatically improves precision over bi-encoder retrieval alone.

    Model: Xenova/ms-marco-MiniLM-L-6-v2 (ONNX, ~25MB)
    """

    def __init__(self, model: str = None):
        self.model_name = model or config.reranker_model
        self._model = None  # Lazy init
        self._load_failed = False

    def _ensure_model(self) -> bool:
        """Lazy initialization of cross-encoder model"""
        if self._load_failed:
            return False
        if self._model is None:
            print(f"[INFO] Loading reranker model: {self.model_name}...")
            try:
                self._model = TextCrossEncoder(model_name=self.model_name, cache_dir=str(config.models_cache_dir))
                print("[INFO] Reranker model loaded successfully")
            except Exception as e:
                self._load_failed = True
                print(f"[WARN] Reranker unavailable, using RRF order: {e}")
                return False
        return True

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder scores.

        Args:
            query: Original search query
            documents: List of result dicts (must have 'document' key)
            top_k: Number of top results to return after reranking

        Returns:
            Reranked list of documents, sorted by cross-encoder score (top_k)
        """
        if not documents or not config.reranker_enabled:
            return documents[:top_k]

        if not self._ensure_model():
            return documents[:top_k]

        texts = [doc.get("document", "") for doc in documents]

        try:
            scores = list(self._model.rerank(query, texts))
            for doc, score in zip(documents, scores):
                doc["reranker_score"] = float(score)
            documents.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)
        except Exception as e:
            print(f"[WARN] Reranker failed, using RRF order: {e}")

        return documents[:top_k]


# =============================================================================
# BM25 INDEX
# =============================================================================


class BM25Index:
    """
    BM25 keyword index for hybrid search with query expansion.

    Maintains a BM25 index of all document chunks for fast keyword-based retrieval.
    Supports security-term synonym expansion for improved recall.
    """

    def __init__(self):
        self.corpus: List[str] = []
        self.corpus_ids: List[str] = []
        self.bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric, keep hyphens"""
        text_lower = text.lower()
        tokens = re.findall(r"[a-z0-9][-a-z0-9]*[a-z0-9]|[a-z0-9]", text_lower)
        return tokens

    def expand_query(self, query: str) -> str:
        """
        Expand query with security-term synonyms for BM25 search.

        Looks up query tokens against config.query_expansions and appends
        synonym tokens. Improves recall for abbreviated technical terms
        (e.g., "sqli" expands to include "sql injection").

        Args:
            query: Original query string

        Returns:
            Expanded query string with synonyms appended
        """
        query_lower = query.lower().strip()
        expanded_terms = set()

        # Check full query
        if query_lower in config.query_expansions:
            expanded_terms.update(config.query_expansions[query_lower])

        # Check individual tokens
        tokens = self._tokenize(query_lower)
        for token in tokens:
            if token in config.query_expansions:
                expanded_terms.update(config.query_expansions[token])

        # Check bigrams
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            if bigram in config.query_expansions:
                expanded_terms.update(config.query_expansions[bigram])

        if expanded_terms:
            return query_lower + " " + " ".join(expanded_terms)
        return query_lower

    def add_documents(self, chunk_ids: List[str], texts: List[str]) -> None:
        """Add documents to the BM25 index"""
        for chunk_id, text in zip(chunk_ids, texts):
            self.corpus.append(text)
            self.corpus_ids.append(chunk_id)
            self._tokenized_corpus.append(self._tokenize(text))

    def build_index(self) -> None:
        """Build/rebuild the BM25 index from the corpus"""
        if self._tokenized_corpus:
            self.bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search the BM25 index with query expansion.

        Returns list of (chunk_id, score) tuples sorted by score descending.
        """
        if not self.bm25 or not self.corpus:
            return []

        # Expand query with synonyms before tokenizing
        expanded_query = self.expand_query(query)
        tokenized_query = self._tokenize(expanded_query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        results = []
        for idx, score in enumerate(scores):
            if score > 0:
                results.append((self.corpus_ids[idx], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        """Clear the index"""
        self.corpus = []
        self.corpus_ids = []
        self._tokenized_corpus = []
        self.bm25 = None

    def __len__(self) -> int:
        return len(self.corpus)


# =============================================================================
# KNOWLEDGE ORCHESTRATOR
# =============================================================================

# =============================================================================
# FILE WATCHER (auto-reindex on document changes)
# =============================================================================


class DocumentWatcher(FileSystemEventHandler):
    """Watches documents directory and triggers reindex on changes."""

    def __init__(self, orchestrator_getter, debounce_seconds: float = 5.0):
        self._get_orchestrator = orchestrator_getter
        self._debounce = debounce_seconds
        self._timer = None
        self._lock = threading.Lock()

    def _schedule_reindex(self):
        """Debounced reindex — waits for changes to settle before reindexing."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce, self._do_reindex)
            self._timer.daemon = True
            self._timer.start()

    def _do_reindex(self):
        """Perform incremental reindex in background."""
        try:
            orch = self._get_orchestrator()
            stats = orch.index_all(force=False)
            changed = stats.get("indexed", 0) + stats.get("updated", 0) + stats.get("deleted", 0)
            if changed > 0:
                print(
                    f"[WATCHER] Auto-reindexed: {stats['indexed']} new, "
                    f"{stats['updated']} updated, {stats['deleted']} deleted"
                )
        except Exception as e:
            print(f"[WATCHER] Reindex failed: {e}")

    def on_created(self, event):
        if not event.is_directory and Path(event.src_path).suffix in config.supported_formats:
            self._schedule_reindex()

    def on_modified(self, event):
        if not event.is_directory and Path(event.src_path).suffix in config.supported_formats:
            self._schedule_reindex()

    def on_deleted(self, event):
        if not event.is_directory and Path(event.src_path).suffix in config.supported_formats:
            self._schedule_reindex()


# =============================================================================
# KNOWLEDGE ORCHESTRATOR
# =============================================================================


class KnowledgeOrchestrator:
    """Main orchestrator for knowledge retrieval with semantic search + keyword routing"""

    def __init__(self):
        self.parser = DocumentParser()
        self.embed_fn = FastEmbedEmbeddings()

        # Initialize ChromaDB with persistent storage (new API v1.4.0+)
        self.chroma_client = chromadb.PersistentClient(path=str(config.chroma_dir))

        # Get or create collection (with auto-recovery from corruption)
        self.collection = self._safe_get_collection()

        # BM25 index for hybrid search
        self.bm25_index = BM25Index()
        self._bm25_initialized = False

        # Cross-encoder reranker (lazy-loaded on first query)
        self.reranker = CrossEncoderReranker()

        # Query cache (LRU with TTL)
        self.query_cache = QueryCache(max_size=100, ttl_seconds=300)

        # Index metadata cache
        self._metadata_file = config.data_dir / "index_metadata.json"
        self._indexed_docs: Dict[str, Dict] = self._load_metadata()

        # Chunk dedup tracking (content_hash -> chunk_id)
        self._chunk_hashes: Dict[str, str] = self._build_dedup_index()

        # Migration: deferred — checked in main() after full init
        self._needs_rebuild = False

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
                print(f"[RECOVERY] Embedding function conflict detected: {e}")
                print("[RECOVERY] Deleting old collection and recreating...")
                try:
                    self.chroma_client.delete_collection(config.collection_name)
                except Exception:
                    pass
            else:
                print(f"[RECOVERY] ChromaDB error: {e}")
                print("[RECOVERY] Clearing corrupted database...")
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

            print("[RECOVERY] Creating fresh collection...")
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
                print(f"[MIGRATION] Embedding dimension mismatch detected: {e}")
                print("[MIGRATION] Nuclear rebuild required.")
                return True
            # Other error — don't trigger rebuild
            print(f"[WARN] Dimension check query failed (non-dimension error): {e}")
            return False

    def _ensure_bm25_index(self) -> None:
        """Lazy initialization of BM25 index from existing ChromaDB data"""
        if self._bm25_initialized:
            return

        try:
            count = self.collection.count()
            if count > 0:
                all_data = self.collection.get(include=["documents"], limit=count)
                if all_data["ids"] and all_data["documents"]:
                    self.bm25_index.add_documents(all_data["ids"], all_data["documents"])
                    self.bm25_index.build_index()
                    print(f"[INFO] BM25 index built with {len(self.bm25_index)} documents")
        except Exception as e:
            print(f"[WARN] Failed to build BM25 index: {e}")

        self._bm25_initialized = True

    # =========================================================================
    # Indexing
    # =========================================================================

    def index_all(self, force: bool = False) -> Dict[str, Any]:
        """
        Index documents with incremental change detection.

        Compares file mtime/size against stored metadata to detect changes.
        Only re-indexes files that are new or modified.
        """
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

        path_to_docid: Dict[str, str] = {}
        for doc_id, info in self._indexed_docs.items():
            path_to_docid[info.get("source", "")] = doc_id

        current_paths = set()

        for doc in documents:
            current_paths.add(str(doc.source))
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

            except Exception as e:
                stats["errors"] += 1
                print(f"[ERROR] Failed to index {doc.source}: {e}")

        # Clean up orphaned docs
        orphan_ids = []
        for doc_id, info in list(self._indexed_docs.items()):
            if info.get("source", "") not in current_paths:
                removed = self._remove_document_chunks(doc_id)
                stats["chunks_removed"] += removed
                stats["deleted"] += 1
                orphan_ids.append(doc_id)

        for doc_id in orphan_ids:
            del self._indexed_docs[doc_id]

        self._save_metadata()

        if stats["indexed"] > 0 or stats["updated"] > 0 or stats["deleted"] > 0:
            self.query_cache.invalidate()

        return stats

    def _index_document(self, doc: Document) -> Tuple[int, int]:
        """Index a single document's chunks into ChromaDB and BM25 with dedup."""
        if not doc.chunks:
            return 0, 0

        unique_ids = []
        unique_docs = []
        unique_metas = []
        dedup_skipped = 0

        for chunk in doc.chunks:
            content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:20]
            chunk_id = f"{doc.id}_{chunk.index}"

            if content_hash in self._chunk_hashes:
                dedup_skipped += 1
                continue

            self._chunk_hashes[content_hash] = chunk_id
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

        if unique_ids:
            self.collection.add(ids=unique_ids, documents=unique_docs, metadatas=unique_metas)
            self.bm25_index.add_documents(unique_ids, unique_docs)

        return len(unique_ids), dedup_skipped

    def _remove_document_chunks(self, doc_id: str) -> int:
        """Remove all chunks belonging to a document from ChromaDB and BM25."""
        try:
            results = self.collection.get(where={"doc_id": doc_id}, include=["metadatas"])

            if results["ids"]:
                for meta in results["metadatas"]:
                    content_hash = meta.get("content_hash", "")
                    if content_hash and content_hash in self._chunk_hashes:
                        del self._chunk_hashes[content_hash]

                self.collection.delete(ids=results["ids"])
                self._bm25_initialized = False
                return len(results["ids"])
        except Exception as e:
            print(f"[WARN] Failed to remove chunks for doc {doc_id}: {e}")

        return 0

    def _build_dedup_index(self) -> Dict[str, str]:
        """Build deduplication index from existing ChromaDB data"""
        dedup = {}
        try:
            count = self.collection.count()
            if count > 0:
                all_data = self.collection.get(include=["metadatas"], limit=count)
                for chunk_id, meta in zip(all_data["ids"], all_data["metadatas"]):
                    content_hash = meta.get("content_hash", "")
                    if content_hash:
                        dedup[content_hash] = chunk_id
        except Exception as e:
            print(f"[WARN] Failed to build dedup index: {e}")
        return dedup

    def reindex_all(self) -> Dict[str, Any]:
        """Smart reindex: incremental detection + BM25 rebuild + orphan cleanup."""
        import shutil

        print("[REINDEX] Starting smart incremental reindex...")
        start_time = time.time()

        stats = self.index_all(force=False)

        print("[REINDEX] Rebuilding BM25 index...")
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
        print(
            f"[REINDEX] Completed in {elapsed:.1f}s "
            f"(indexed: {stats['indexed']}, updated: {stats['updated']}, "
            f"skipped: {stats['skipped']}, deleted: {stats['deleted']})"
        )

        return stats

    def nuclear_rebuild(self) -> Dict[str, Any]:
        """Nuclear rebuild: DELETE everything and re-embed ALL documents."""
        import shutil

        print("[NUCLEAR] Starting full rebuild...")
        start_time = time.time()

        try:
            self.chroma_client.delete_collection(config.collection_name)
            print("[NUCLEAR] Deleted ChromaDB collection")
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
        self.bm25_index.clear()
        self._bm25_initialized = False
        self._chunk_hashes = {}
        self.query_cache.invalidate()

        stats = self.index_all(force=True)

        self.bm25_index.build_index()
        self._bm25_initialized = True

        elapsed = time.time() - start_time
        stats["elapsed_seconds"] = round(elapsed, 2)
        print(
            f"[NUCLEAR] Full rebuild completed in {elapsed:.1f}s "
            f"({stats['indexed']} docs, {stats['chunks_added']} chunks)"
        )

        return stats

    # =========================================================================
    # Search
    # =========================================================================

    def query(
        self, query_text: str, max_results: int = None, category_filter: Optional[str] = None, hybrid_alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with RRF fusion + cross-encoder reranking.

        Pipeline: Semantic + BM25 -> RRF fusion -> Reranker -> Results
        """
        max_results = max_results or config.default_results

        # Check cache
        cached = self.query_cache.get(query_text, max_results, category_filter, hybrid_alpha)
        if cached is not None:
            return cached

        self._ensure_bm25_index()

        # Keyword routing
        routed_category = self._route_by_keywords(query_text)
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}
        elif routed_category:
            where_filter = {"category": routed_category}

        # Parallel Semantic + BM25 search (threaded for latency reduction)
        from concurrent.futures import ThreadPoolExecutor

        semantic_results = {}
        bm25_results = {}

        def _do_semantic():
            r = {}
            if hybrid_alpha > 0:
                try:
                    n_candidates = min(max_results * 3, config.max_results)
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
                    print(f"[WARN] Semantic search failed: {e}")
            return r

        def _do_bm25():
            r = {}
            if hybrid_alpha < 1.0:
                try:
                    bm25_hits = self.bm25_index.search(query_text, top_k=max_results * 3)
                    for rank, (chunk_id, bm25_score) in enumerate(bm25_hits):
                        r[chunk_id] = {"rank": rank + 1, "bm25_score": bm25_score}
                except Exception as e:
                    print(f"[WARN] BM25 search failed: {e}")
            return r

        # Run both in parallel when hybrid mode
        if 0 < hybrid_alpha < 1.0:
            with ThreadPoolExecutor(max_workers=2) as executor:
                sem_future = executor.submit(_do_semantic)
                bm25_future = executor.submit(_do_bm25)
                semantic_results = sem_future.result()
                bm25_results = bm25_future.result()
        else:
            semantic_results = _do_semantic()
            bm25_results = _do_bm25()

        # RRF Fusion
        RRF_K = 60
        combined_scores: Dict[str, Dict] = {}
        all_chunk_ids = set(semantic_results.keys()) | set(bm25_results.keys())

        for chunk_id in all_chunk_ids:
            semantic_rank = semantic_results.get(chunk_id, {}).get("rank", 1000)
            bm25_rank = bm25_results.get(chunk_id, {}).get("rank", 1000)

            semantic_rrf = hybrid_alpha * (1 / (RRF_K + semantic_rank))
            bm25_rrf = (1 - hybrid_alpha) * (1 / (RRF_K + bm25_rank))
            combined_rrf = semantic_rrf + bm25_rrf

            if chunk_id in semantic_results:
                data = semantic_results[chunk_id]
            else:
                try:
                    fetched = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
                    data = {
                        "document": fetched["documents"][0] if fetched["documents"] else "",
                        "metadata": fetched["metadatas"][0] if fetched["metadatas"] else {},
                        "distance": 0,
                    }
                except Exception:
                    continue

            combined_scores[chunk_id] = {
                "rrf_score": combined_rrf,
                "semantic_rank": semantic_rank if chunk_id in semantic_results else None,
                "bm25_rank": bm25_rank if chunk_id in bm25_results else None,
                "document": data.get("document", ""),
                "metadata": data.get("metadata", {}),
                "distance": data.get("distance", 0),
            }

        # Sort by RRF score — take extra candidates for reranker
        reranker_k = max_results * config.reranker_top_k_multiplier if config.reranker_enabled else max_results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)[:reranker_k]

        # Cross-encoder reranking
        if config.reranker_enabled and sorted_results:
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

        # MMR: Maximal Marginal Relevance — diversify results to reduce redundancy
        if len(sorted_results) > max_results:
            sorted_results = self._apply_mmr(sorted_results, max_results, lambda_param=0.7)

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

            formatted.append(
                {
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
                }
            )

        # Adjacent Chunk Retrieval — expand content with surrounding chunks for context
        formatted = self._expand_with_adjacent_chunks(formatted)

        self.query_cache.put(query_text, max_results, category_filter, hybrid_alpha, formatted)
        return formatted

    def _expand_with_adjacent_chunks(self, results: List[Dict], window: int = 1) -> List[Dict]:
        """
        Expand each result with adjacent chunks for broader context.

        For each matched chunk, fetches the chunks immediately before and after it
        (same document) and prepends/appends their content. This gives the LLM
        surrounding context while maintaining precise retrieval on the matched chunk.

        Args:
            results: Formatted search results
            window: Number of adjacent chunks to fetch on each side (default: 1)

        Returns:
            Results with expanded content field
        """
        if not results:
            return results

        for result in results:
            source = result.get("source", "")
            chunk_idx = result.get("chunk_index", 0)

            if not source or chunk_idx is None:
                continue

            # Find the doc_id from metadata lookup
            doc_id = None
            for did, info in self._indexed_docs.items():
                stored = str(Path(info.get("source", "")).resolve())
                if stored == str(Path(source).resolve()):
                    doc_id = did
                    break

            if not doc_id:
                continue

            # Fetch adjacent chunks from ChromaDB
            adjacent_ids = []
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue  # Skip the matched chunk itself
                adj_id = f"{doc_id}_{chunk_idx + offset}"
                adjacent_ids.append(adj_id)

            if not adjacent_ids:
                continue

            try:
                adj_data = self.collection.get(ids=adjacent_ids, include=["documents"])
                if adj_data["ids"] and adj_data["documents"]:
                    # Build ordered context: prev + matched + next
                    parts_before = []
                    parts_after = []
                    for adj_id, adj_doc in zip(adj_data["ids"], adj_data["documents"]):
                        if adj_doc:
                            idx = int(adj_id.split("_")[-1])
                            if idx < chunk_idx:
                                parts_before.append(adj_doc)
                            else:
                                parts_after.append(adj_doc)

                    if parts_before or parts_after:
                        expanded = "\n\n".join(parts_before + [result["content"]] + parts_after)
                        result["content"] = expanded
                        result["context_expanded"] = True
            except Exception:
                pass  # Adjacent chunk not found — use original content

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
        """
        Maximal Marginal Relevance — diversify results to reduce redundancy.

        Balances relevance (score) vs diversity (dissimilarity to already selected docs).
        lambda=1.0 = pure relevance, lambda=0.0 = pure diversity, default 0.7 = relevance-heavy.
        """
        if len(results) <= top_k:
            return results

        # Use content text for similarity (simple Jaccard on token sets)
        def jaccard_sim(a: str, b: str) -> float:
            tokens_a = set(a.lower().split())
            tokens_b = set(b.lower().split())
            if not tokens_a or not tokens_b:
                return 0.0
            return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

        selected = [results[0]]  # First result always selected (highest score)
        remaining = list(results[1:])

        while len(selected) < top_k and remaining:
            best_idx = 0
            best_mmr = -1.0

            for i, (chunk_id, data) in enumerate(remaining):
                # Relevance score (normalized)
                relevance = data.get("reranker_score", data.get("rrf_score", 0))

                # Max similarity to any already-selected doc
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
        """Get full document content by filepath"""
        filepath = Path(filepath)
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
            print(f"[ERROR] Failed to read document {filepath}: {e}")
        return None

    def add_document_from_content(self, content: str, filepath: str, category: str) -> Dict[str, Any]:
        """Add a new document from raw content string. Saves to disk and indexes."""
        full_path = config.documents_dir / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

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
        self._save_metadata()
        self.query_cache.invalidate()
        self.bm25_index.build_index()

        return {
            "chunks_added": chunks_added,
            "dedup_skipped": dedup_skipped,
            "category": category,
            "filepath": str(full_path),
        }

    def update_document_content(self, filepath: str, content: str) -> Dict[str, Any]:
        """Update an existing document. Removes old chunks and re-indexes."""
        filepath = Path(filepath)
        if not filepath.exists():
            return {"error": f"File not found: {filepath}"}

        # Resolve to absolute for consistent comparison with stored metadata
        filepath_resolved = str(filepath.resolve())

        doc_id = None
        for did, info in self._indexed_docs.items():
            stored = str(Path(info.get("source", "")).resolve())
            if stored == filepath_resolved:
                doc_id = did
                break

        old_chunks_removed = 0
        if doc_id:
            old_chunks_removed = self._remove_document_chunks(doc_id)
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
        self._save_metadata()
        self.query_cache.invalidate()
        self.bm25_index.build_index()

        return {
            "old_chunks_removed": old_chunks_removed,
            "new_chunks_added": new_chunks_added,
            "dedup_skipped": dedup_skipped,
            "filepath": str(filepath),
        }

    def remove_document_by_path(self, filepath: str, delete_file: bool = False) -> Dict[str, Any]:
        """Remove a document from the index. Optionally delete from disk."""
        filepath_resolved = str(Path(filepath).resolve())

        doc_id = None
        for did, info in self._indexed_docs.items():
            stored = str(Path(info.get("source", "")).resolve())
            if stored == filepath_resolved:
                doc_id = did
                break

        if not doc_id:
            return {"error": f"Document not found in index: {filepath}"}

        chunks_removed = self._remove_document_chunks(doc_id)
        del self._indexed_docs[doc_id]

        if delete_file:
            try:
                Path(filepath).unlink(missing_ok=True)
            except Exception as e:
                print(f"[WARN] Failed to delete file {filepath}: {e}")

        self._save_metadata()
        self.query_cache.invalidate()

        return {"chunks_removed": chunks_removed, "filepath": filepath_resolved, "file_deleted": delete_file}

    def add_from_url(self, url: str, category: str, title: str = None) -> Dict[str, Any]:
        """Fetch URL content, convert to markdown, and add to knowledge base."""
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

        return self.add_document_from_content(clean_text, filepath, category)

    def search_similar(self, filepath: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document using embedding similarity."""
        filepath_resolved = str(Path(filepath).resolve())

        doc_id = None
        for did, info in self._indexed_docs.items():
            stored = str(Path(info.get("source", "")).resolve())
            if stored == filepath_resolved:
                doc_id = did
                break

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

    def evaluate_retrieval(self, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluate retrieval quality with test queries. Returns MRR@5, Recall@5, Precision@5."""
        per_query = []
        mrr_sum = 0.0
        recall_sum = 0.0
        k = 5

        for tc in test_cases:
            query = tc.get("query", "")
            expected = tc.get("expected_filepath", "")

            results = self.query(query, max_results=k)

            found_rank = None
            for i, r in enumerate(results):
                if expected in r.get("source", ""):
                    found_rank = i + 1
                    break

            rr = 1.0 / found_rank if found_rank else 0.0
            recall = 1.0 if found_rank else 0.0

            mrr_sum += rr
            recall_sum += recall

            per_query.append(
                {
                    "query": query,
                    "expected": expected,
                    "found_at_rank": found_rank,
                    "reciprocal_rank": round(rr, 4),
                    "top_result": results[0]["source"] if results else "none",
                }
            )

        n = len(test_cases) if test_cases else 1
        return {
            "total_queries": len(test_cases),
            "mrr_at_5": round(mrr_sum / n, 4),
            "recall_at_5": round(recall_sum / n, 4),
            "per_query": per_query,
        }

    # =========================================================================
    # Stats & Metadata
    # =========================================================================

    def list_categories(self) -> Dict[str, int]:
        """List all categories with document counts"""
        categories = {}
        for doc_info in self._indexed_docs.values():
            cat = doc_info.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        return categories

    def list_documents(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        """List all indexed documents, optionally filtered by category"""
        docs = []
        for doc_id, info in self._indexed_docs.items():
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
        """Get index statistics"""
        return {
            "total_documents": len(self._indexed_docs),
            "total_chunks": self.collection.count(),
            "unique_content_hashes": len(self._chunk_hashes),
            "categories": self.list_categories(),
            "supported_formats": config.supported_formats,
            "embedding_model": config.embedding_model,
            "embedding_dim": config.embedding_dim,
            "reranker_model": config.reranker_model if config.reranker_enabled else "disabled",
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "query_cache": self.query_cache.stats(),
        }

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
        self._metadata_file.write_text(json.dumps(self._indexed_docs, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# MCP Server
# =============================================================================

mcp = FastMCP("knowledge-rag")

_orchestrator: Optional[KnowledgeOrchestrator] = None


def get_orchestrator() -> KnowledgeOrchestrator:
    """Get or create the orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = KnowledgeOrchestrator()
    return _orchestrator


# =============================================================================
# MCP Tools — Existing (6)
# =============================================================================


@mcp.tool()
def search_knowledge(query: str, max_results: int = 5, category: str = None, hybrid_alpha: float = 0.3) -> str:
    """
    Hybrid search combining semantic search + BM25 keyword search with cross-encoder reranking.

    Args:
        query: Search query text
        max_results: Maximum number of results (default: 5, max: 20)
        category: Optional category filter (security, ctf, logscale, development, general, redteam, blueteam)
        hybrid_alpha: Balance between semantic and keyword search (0.0 = keyword only, 1.0 = semantic only, default: 0.3)

    Returns:
        JSON string with search results including content, source, relevance score, and search method used
    """
    if not query or not query.strip():
        return json.dumps({"status": "error", "message": "Query cannot be empty"})

    max_results = max(1, min(max_results or 5, config.max_results))
    hybrid_alpha = max(0.0, min(hybrid_alpha if hybrid_alpha is not None else 0.3, 1.0))

    valid_categories = list(config.keyword_routes.keys()) + list(set(config.category_mappings.values()))
    if category and category not in valid_categories:
        return json.dumps(
            {"status": "error", "message": f"Invalid category '{category}'. Valid: {', '.join(valid_categories)}"}
        )

    orchestrator = get_orchestrator()
    results = orchestrator.query(
        query.strip(), max_results=max_results, category_filter=category, hybrid_alpha=hybrid_alpha
    )

    if not results:
        return json.dumps({"status": "no_results", "query": query, "message": "No relevant documents found."})

    return json.dumps(
        {
            "status": "success",
            "query": query,
            "hybrid_alpha": hybrid_alpha,
            "result_count": len(results),
            "cache_hit_rate": orchestrator.query_cache.stats()["hit_rate"],
            "results": results,
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool()
def get_document(filepath: str) -> str:
    """
    Get the full content of a specific document.

    Args:
        filepath: Path to the document file

    Returns:
        JSON string with document content and metadata
    """
    orchestrator = get_orchestrator()
    doc = orchestrator.get_document(filepath)

    if not doc:
        return json.dumps({"status": "error", "message": f"Document not found: {filepath}"})

    return json.dumps({"status": "success", "document": doc}, indent=2, ensure_ascii=False)


@mcp.tool()
def reindex_documents(force: bool = False, full_rebuild: bool = False) -> str:
    """
    Index or reindex all documents in the knowledge base.

    Args:
        force: If True, smart reindex (detects changes + rebuilds BM25). FAST.
        full_rebuild: If True, nuclear rebuild (deletes everything, re-embeds ALL). Use if model changed.

    Returns:
        JSON string with indexing statistics
    """
    orchestrator = get_orchestrator()

    if full_rebuild:
        stats = orchestrator.nuclear_rebuild()
        operation = "nuclear_rebuild"
    elif force:
        stats = orchestrator.reindex_all()
        operation = "smart_reindex"
    else:
        stats = orchestrator.index_all()
        operation = "incremental_index"

    return json.dumps({"status": "success", "operation": operation, "stats": stats}, indent=2, ensure_ascii=False)


@mcp.tool()
def list_categories() -> str:
    """List all document categories with their document counts."""
    orchestrator = get_orchestrator()
    categories = orchestrator.list_categories()
    return json.dumps(
        {"status": "success", "categories": categories, "total_documents": sum(categories.values())}, indent=2
    )


@mcp.tool()
def list_documents(category: str = None) -> str:
    """
    List all indexed documents, optionally filtered by category.

    Args:
        category: Optional category filter
    """
    orchestrator = get_orchestrator()
    docs = orchestrator.list_documents(category=category)
    return json.dumps(
        {"status": "success", "filter": category or "all", "count": len(docs), "documents": docs},
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool()
def get_index_stats() -> str:
    """Get statistics about the knowledge base index."""
    orchestrator = get_orchestrator()
    stats = orchestrator.get_stats()
    return json.dumps({"status": "success", "stats": stats}, indent=2)


# =============================================================================
# MCP Tools — New (6)
# =============================================================================


@mcp.tool()
def add_document(content: str, filepath: str, category: str = "general") -> str:
    """
    Add a new document to the knowledge base from raw content.

    Saves the content to the documents directory and indexes it immediately.

    Args:
        content: Full text content of the document
        filepath: Relative path within documents dir (e.g., "security/new-technique.md")
        category: Document category (security, ctf, logscale, development, general)

    Returns:
        JSON string with indexing results
    """
    if not content or not content.strip():
        return json.dumps({"status": "error", "message": "Content cannot be empty"})
    if not filepath or not filepath.strip():
        return json.dumps({"status": "error", "message": "Filepath cannot be empty"})

    orchestrator = get_orchestrator()
    result = orchestrator.add_document_from_content(content.strip(), filepath.strip(), category)

    if "error" in result:
        return json.dumps({"status": "error", "message": result["error"]})

    return json.dumps({"status": "success", **result}, indent=2)


@mcp.tool()
def update_document(filepath: str, content: str) -> str:
    """
    Update an existing document in the knowledge base.

    Removes old chunks and re-indexes with new content.

    Args:
        filepath: Full path to the document file
        content: New content for the document

    Returns:
        JSON string with update results
    """
    if not filepath:
        return json.dumps({"status": "error", "message": "Filepath required"})
    if not content or not content.strip():
        return json.dumps({"status": "error", "message": "Content cannot be empty"})

    orchestrator = get_orchestrator()
    result = orchestrator.update_document_content(filepath, content.strip())

    if "error" in result:
        return json.dumps({"status": "error", "message": result["error"]})

    return json.dumps({"status": "success", **result}, indent=2)


@mcp.tool()
def remove_document(filepath: str, delete_file: bool = False) -> str:
    """
    Remove a document from the knowledge base index.

    Args:
        filepath: Path to the document file
        delete_file: If True, also delete the file from disk (default: False)

    Returns:
        JSON string with removal results
    """
    if not filepath:
        return json.dumps({"status": "error", "message": "Filepath required"})

    orchestrator = get_orchestrator()
    result = orchestrator.remove_document_by_path(filepath, delete_file=delete_file)

    if "error" in result:
        return json.dumps({"status": "error", "message": result["error"]})

    return json.dumps({"status": "success", **result}, indent=2)


@mcp.tool()
def add_from_url(url: str, category: str = "general", title: str = None) -> str:
    """
    Fetch content from a URL and add it to the knowledge base.

    Fetches the page, strips HTML, converts to markdown, and indexes.

    Args:
        url: URL to fetch content from
        category: Document category (default: general)
        title: Optional title for the document (auto-detected if not provided)

    Returns:
        JSON string with indexing results
    """
    if not url or not url.strip():
        return json.dumps({"status": "error", "message": "URL cannot be empty"})

    orchestrator = get_orchestrator()
    result = orchestrator.add_from_url(url.strip(), category, title)

    if "error" in result:
        return json.dumps({"status": "error", "message": result["error"]})

    return json.dumps({"status": "success", **result}, indent=2)


@mcp.tool()
def search_similar(filepath: str, max_results: int = 5) -> str:
    """
    Find documents similar to a given document.

    Uses the document's embedding to find semantically similar documents.

    Args:
        filepath: Path to the reference document
        max_results: Number of similar documents to return (default: 5)

    Returns:
        JSON string with list of similar documents and similarity scores
    """
    if not filepath:
        return json.dumps({"status": "error", "message": "Filepath required"})

    max_results = max(1, min(max_results or 5, 20))

    orchestrator = get_orchestrator()
    results = orchestrator.search_similar(filepath, max_results=max_results)

    if not results:
        return json.dumps({"status": "no_results", "message": "No similar documents found or document not indexed"})

    return json.dumps(
        {"status": "success", "reference": filepath, "count": len(results), "similar_documents": results},
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool()
def evaluate_retrieval(test_cases: str) -> str:
    """
    Evaluate retrieval quality with test queries.

    Args:
        test_cases: JSON string of test cases. Format: [{"query": "search term", "expected_filepath": "path/to/doc.md"}, ...]

    Returns:
        JSON string with MRR@5, Recall@5, and per-query results
    """
    try:
        cases = json.loads(test_cases) if isinstance(test_cases, str) else test_cases
    except json.JSONDecodeError:
        return json.dumps({"status": "error", "message": "Invalid JSON for test_cases"})

    if not isinstance(cases, list) or not cases:
        return json.dumps({"status": "error", "message": "test_cases must be a non-empty JSON array"})

    orchestrator = get_orchestrator()
    results = orchestrator.evaluate_retrieval(cases)

    return json.dumps({"status": "success", **results}, indent=2)


# =============================================================================
# Entry point
# =============================================================================


def _handle_init():
    """Export config template and presets to current directory."""
    import shutil

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


def main():
    """Run the MCP server"""
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        _handle_init()
        return

    from .preflight import run_preflight

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
    try:
        watcher = DocumentWatcher(get_orchestrator, debounce_seconds=5.0)
        observer = Observer()
        observer.schedule(watcher, str(config.documents_dir), recursive=True)
        observer.daemon = True
        observer.start()
        print(f"[WATCHER] Monitoring {config.documents_dir} for changes")
    except Exception as e:
        print(f"[WARN] Failed to start file watcher: {e}")
        print("[WARN] Auto-reindexing disabled. Use reindex_documents tool manually.")

    # Restore real stdout for MCP JSON-RPC, keep print() going to stderr
    from . import _original_stdout

    sys.stdout = _original_stdout
    mcp.run()


if __name__ == "__main__":
    main()
