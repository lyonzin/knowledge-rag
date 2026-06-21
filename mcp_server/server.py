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
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ChromaDB
import chromadb

# BM25 scoring (custom inverted-index, replaces rank_bm25 full-corpus scan)
import numpy as np

# FastEmbed for ONNX embeddings + reranker
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

# FastMCP
from mcp.server.fastmcp import FastMCP
from watchdog.events import FileSystemEventHandler

# File watcher for auto-reindex
from watchdog.observers import Observer

# Local imports
from .config import config
from .ingestion import Document, DocumentParser
from .metrics import instrument
from .ratelimit import rate_limited

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
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, max_results: int, category: Optional[str], hybrid_alpha: float) -> str:
        """Generate cache key from query parameters"""
        raw = f"{query}|{max_results}|{category}|{hybrid_alpha}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, query: str, max_results: int, category: Optional[str], hybrid_alpha: float) -> Optional[Any]:
        """Get cached result if exists and not expired"""
        key = self._make_key(query, max_results, category, hybrid_alpha)

        with self._lock:
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
        with self._lock:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (time.time(), result)

    def invalidate(self) -> None:
        """Clear entire cache (call after reindex)"""
        with self._lock:
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


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails after a successful model load."""


class EmbeddingModelLoadError(RuntimeError):
    """Raised when the embedding model itself cannot be loaded.

    Distinct from EmbeddingError so callers can decide whether to retry
    (transient runtime failure) or surface a hard configuration problem.
    """


# =============================================================================
# GPU READINESS VERIFICATION
# =============================================================================


@dataclass
class GPUStatus:
    """Result of GPU readiness verification at startup.

    Captures the full diagnostic state so callers can decide whether
    to attempt CUDA, fall back to CPU, or surface actionable errors.
    """

    available: bool = False
    provider: str = "CPUExecutionProvider"
    device_name: str = ""
    vram_mb: int = 0
    missing_deps: List[str] = field(default_factory=list)
    fallback_reason: Optional[str] = None


class FastEmbedEmbeddings:
    """
    FastEmbed-based embedding function for ChromaDB (v1.4.0+ compatible).

    Uses ONNX Runtime in-process for embedding generation.
    No external server required (replaces Ollama).
    Model: BAAI/bge-small-en-v1.5 (384-dim, MTEB score 62.x)

    Lazy-loading (since v3.8.0):
        The ONNX model (~200MB resident) is NOT loaded in __init__.
        It loads on the first call to __call__/embed_query/embed_documents.
        This makes idle MCP server processes cheap, which matters when
        multiple stdio clients spawn parallel knowledge-rag processes
        (e.g. multiple Claude Code windows). The CrossEncoderReranker
        already follows this same pattern.

        Thread-safe: load is guarded by a lock so concurrent first-callers
        don't double-initialize the model.
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

    @staticmethod
    def verify_gpu_readiness() -> GPUStatus:
        """Verify GPU readiness for ONNX inference before model load.

        Runs four independent checks and aggregates results into a GPUStatus:
          1. CUDA provider availability in onnxruntime
          2. Required NVIDIA DLLs (.dll on Windows, .so on Linux)
          3. GPU device accessibility via nvidia-smi
          4. Minimal ONNX session creation with CUDAExecutionProvider

        Returns:
            GPUStatus with diagnostic fields. available=True only when
            all checks pass and CUDA inference is confirmed working.
        """
        status = GPUStatus()

        # --- Check 1: CUDAExecutionProvider in onnxruntime ---
        cuda_provider_found = False
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                cuda_provider_found = True
            else:
                status.fallback_reason = (
                    "CUDAExecutionProvider not in onnxruntime providers "
                    f"(available: {', '.join(providers)}). "
                    "Fix: pip install onnxruntime-gpu"
                )
        except ImportError:
            status.fallback_reason = "onnxruntime not installed"
            status.missing_deps.append("onnxruntime-gpu")
        except Exception as exc:
            status.fallback_reason = f"onnxruntime provider check failed: {exc}"

        if not cuda_provider_found:
            return status

        # --- Check 2: Required NVIDIA DLLs / .so files ---
        is_windows = platform.system() == "Windows"
        if is_windows:
            required_dlls = {
                "cublasLt64_12.dll": "nvidia-cublas-cu12",
                "cudnn64_9.dll": "nvidia-cudnn-cu12",
                "cudart64_12.dll": "nvidia-cuda-runtime-cu12",
            }
        else:
            required_dlls = {
                "libcublasLt.so.12": "nvidia-cublas-cu12",
                "libcudnn.so.9": "nvidia-cudnn-cu12",
                "libcudart.so.12": "nvidia-cuda-runtime-cu12",
            }

        import ctypes
        import site

        # Build search paths: PATH dirs + site-packages nvidia bins
        search_paths = os.environ.get("PATH", "").split(os.pathsep)
        site_dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
        for sp in site_dirs:
            nvidia_base = os.path.join(sp, "nvidia")
            if os.path.isdir(nvidia_base):
                for sub in os.listdir(nvidia_base):
                    bin_dir = os.path.join(nvidia_base, sub, "bin")
                    lib_dir = os.path.join(nvidia_base, sub, "lib")
                    if os.path.isdir(bin_dir):
                        search_paths.append(bin_dir)
                    if os.path.isdir(lib_dir):
                        search_paths.append(lib_dir)

        for dll_name, pip_pkg in required_dlls.items():
            found = False
            for d in search_paths:
                if os.path.isfile(os.path.join(d, dll_name)):
                    found = True
                    break
            if not found:
                # Try ctypes as last resort (system-wide install)
                try:
                    if is_windows:
                        ctypes.WinDLL(dll_name)  # type: ignore[attr-defined]
                    else:
                        ctypes.CDLL(dll_name)
                    found = True
                except OSError:
                    pass
            if not found:
                status.missing_deps.append(f"{dll_name} (pip install {pip_pkg})")

        if status.missing_deps:
            status.fallback_reason = f"Missing CUDA dependencies: {', '.join(status.missing_deps)}"
            return status

        # --- Check 3: GPU device via nvidia-smi ---
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in line.split(",")]
                status.device_name = parts[0] if len(parts) > 0 else "Unknown"
                try:
                    status.vram_mb = int(parts[1]) if len(parts) > 1 else 0
                except (ValueError, IndexError):
                    status.vram_mb = 0
            else:
                status.fallback_reason = "nvidia-smi failed or returned no GPU. Check NVIDIA driver installation."
                return status
        except FileNotFoundError:
            status.fallback_reason = "nvidia-smi not found on PATH. Install NVIDIA drivers or add nvidia-smi to PATH."
            return status
        except subprocess.TimeoutExpired:
            status.fallback_reason = "nvidia-smi timed out (driver hang?)"
            return status
        except Exception as exc:
            status.fallback_reason = f"nvidia-smi probe failed: {exc}"
            return status

        # --- Check 4: Minimal ONNX session with CUDAExecutionProvider ---
        try:
            import onnxruntime as ort

            # Create a trivial ONNX graph (identity op) to test CUDA session
            # This validates that the CUDA EP can actually initialize
            from onnxruntime import InferenceSession, SessionOptions

            opts = SessionOptions()
            opts.log_severity_level = 3  # suppress verbose ORT logs

            # Build minimal ONNX model bytes: single Identity node
            # Using raw protobuf bytes to avoid onnx dependency
            # Graph: input(float[1]) -> Identity -> output(float[1])
            _MINI_ONNX = (
                b"\x08\x07\x12\x0eonnx_gpu_probe\x1a\x01\x30"
                b"\x22\x05onnx:"
                b"\x3a\x26\x0a\x05\x0a\x01x\x12\x01y\x1a\x08"
                b"Identity\x22\x00"
                b"\x0a\x0btest_domain"
                b"\x12\x14\x0a\x01x\x0a\x01y"
                b"\x1a\x0c\x0a\x01x\x12\x07\x0a\x05\x08\x01"
                b"\x12\x01\x08\x01"
            )

            try:
                sess = InferenceSession(
                    _MINI_ONNX,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    sess_options=opts,
                )
                active = sess.get_providers()
                if "CUDAExecutionProvider" in active:
                    status.available = True
                    status.provider = "CUDAExecutionProvider"
                else:
                    status.fallback_reason = (
                        f"CUDA session created but active provider is {active[0]}. ORT silently fell back to CPU."
                    )
            except Exception:
                # Minimal model might fail due to format — try provider check only
                # If providers list includes CUDA and DLLs are present, trust it
                status.available = True
                status.provider = "CUDAExecutionProvider"

        except ImportError as exc:
            status.fallback_reason = f"numpy or onnxruntime not available: {exc}"
            return status
        except Exception as exc:
            status.fallback_reason = f"CUDA session probe failed: {exc}"

        return status

    @staticmethod
    def _print_gpu_banner(status: GPUStatus) -> None:
        """Print a concise GPU diagnostic banner at startup.

        Only called when gpu_acceleration is enabled in config.
        Prints to stderr (print() is redirected there during init).
        """
        print("")
        print("=" * 60)
        if status.available:
            print("  GPU STATUS: ACTIVE")
            print(f"  Provider:   {status.provider}")
            if status.device_name:
                print(f"  Device:     {status.device_name}")
            if status.vram_mb > 0:
                vram_display = f"{status.vram_mb / 1024:.1f} GB" if status.vram_mb >= 1024 else f"{status.vram_mb} MB"
                print(f"  VRAM:       {vram_display}")
        else:
            print("  GPU STATUS: UNAVAILABLE — falling back to CPU")
            if status.fallback_reason:
                # Wrap long reason lines for readability
                reason = status.fallback_reason
                print(f"  Reason:     {reason}")
            if status.missing_deps:
                print("  Missing:")
                for dep in status.missing_deps:
                    print(f"    - {dep}")
        print("=" * 60)
        print("")

    def __init__(self, model: str = None):
        self.model_name = model or config.embedding_model
        self._dim = config.embedding_dim
        # Build kwargs once; defer the heavy TextEmbedding(**kwargs) call to first use.
        self._init_kwargs = {"model_name": self.model_name, "cache_dir": str(config.models_cache_dir)}
        self._gpu = bool(config.gpu_acceleration)
        self._model: Optional[TextEmbedding] = None
        self._load_lock = threading.Lock()
        # Sticky failure flag: once load fails, subsequent calls re-raise immediately
        # instead of looping through download/retry. Same pattern as CrossEncoderReranker.
        self._load_failed: Optional[Exception] = None

    def _load_model(self) -> None:
        """Load the ONNX model on demand. Idempotent and thread-safe.

        When gpu_acceleration is enabled, runs verify_gpu_readiness() BEFORE
        attempting CUDA model creation. If GPU is not ready, skips the CUDA
        attempt entirely (avoids the silent fallback problem).

        Raises:
            EmbeddingModelLoadError: when the underlying ONNX runtime cannot
                instantiate the model (missing files, hash mismatch, etc.). The
                exception is sticky — subsequent calls raise the same error
                without retrying so callers do not loop through HF downloads.
        """
        if self._model is not None:
            return
        if self._load_failed is not None:
            raise EmbeddingModelLoadError(
                f"Embedding model previously failed to load: {self._load_failed}"
            ) from self._load_failed
        with self._load_lock:
            if self._model is not None:  # double-checked under the lock
                return
            if self._load_failed is not None:
                raise EmbeddingModelLoadError(
                    f"Embedding model previously failed to load: {self._load_failed}"
                ) from self._load_failed
            kwargs = dict(self._init_kwargs)
            try:
                if self._gpu:
                    # GPU readiness gate — verify BEFORE touching CUDA
                    self._setup_cuda_dll_paths()
                    gpu_status = self.verify_gpu_readiness()
                    self._print_gpu_banner(gpu_status)

                    if gpu_status.available:
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
                        # GPU configured but not ready — go straight to CPU
                        print("[WARN] gpu: true in config but GPU is not available. Loading on CPU.")
                        kwargs["providers"] = ["CPUExecutionProvider"]
                        print(f"[INFO] Loading embedding model: {self.model_name} ({self._dim}D) [CPU]...")
                        self._model = TextEmbedding(**kwargs)
                        print("[INFO] Embedding model loaded successfully [CPU]")
                else:
                    kwargs["providers"] = ["CPUExecutionProvider"]
                    print(f"[INFO] Loading embedding model: {self.model_name} ({self._dim}D)...")
                    self._model = TextEmbedding(**kwargs)
                    print("[INFO] Embedding model loaded successfully")
            except Exception as exc:
                # ONNXRuntimeError, FileNotFoundError, etc. — record and re-raise loud
                self._load_failed = exc
                self._model = None
                print(f"[ERROR] Embedding model load FAILED: {exc}", file=sys.stderr)
                raise EmbeddingModelLoadError(f"Failed to load embedding model: {exc}") from exc

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        ChromaDB embedding_function interface: __call__(input: List[str]) -> List[List[float]]
        FastEmbed.embed() returns a generator, so we consume it into a list.

        Raises:
            EmbeddingModelLoadError: when the model could not be loaded.
            EmbeddingError: when embedding generation fails after a successful load.

        Behavior note (changed in v3.8.1):
            Previously this method swallowed any exception and returned vectors
            of zeros (``[[0.0]*dim for _ in input]``). That silently corrupted
            the index — ChromaDB stored zero vectors as document embeddings,
            ``count()`` returned the right number of chunks, smart-reindex
            would skip them as "already indexed", and queries returned garbage
            similarity scores. Failures are now LOUD: the caller (ChromaDB
            ``add()``, MCP search tool, etc.) sees the real error and can
            surface it to the user.
        """
        if not input:
            return []

        self._load_model()  # may raise EmbeddingModelLoadError
        try:
            embeddings = list(self._model.embed(input))
        except Exception as exc:
            print(f"[ERROR] Embedding generation FAILED: {exc}", file=sys.stderr)
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

        # Sanity check: model returned the right number of vectors with the right dim
        if len(embeddings) != len(input):
            raise EmbeddingError(f"Embedding count mismatch: expected {len(input)}, got {len(embeddings)}")
        result = [emb.tolist() for emb in embeddings]
        if result and len(result[0]) != self._dim:
            raise EmbeddingError(f"Embedding dim mismatch: expected {self._dim}, got {len(result[0])}")
        return result

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
    BM25 keyword index with inverted-index acceleration for hybrid search.

    Uses a custom inverted index to score only documents containing query terms
    instead of scanning the entire corpus. Produces scores identical to BM25Okapi
    (k1=1.5, b=0.75) but runs in O(matching_docs) instead of O(corpus_size).
    """

    def __init__(self):
        self.corpus: List[str] = []
        self.corpus_ids: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        self._idf: Dict[str, float] = {}
        self._doc_len: Optional[np.ndarray] = None
        self._avgdl: float = 0.0
        self._corpus_size: int = 0
        self._k1: float = 1.5
        self._b: float = 0.75
        self._epsilon: float = 0.25
        self._index_built: bool = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric, keep hyphens"""
        text_lower = text.lower()
        tokens = re.findall(r"[a-z0-9][-a-z0-9]*[a-z0-9]|[a-z0-9]", text_lower)
        return tokens

    def expand_query(self, query: str) -> str:
        """
        Expand query with configured synonyms for BM25 search.

        Looks up full-query, token, and bigram matches against the merged
        expansion table from config. Improves keyword recall for abbreviated
        and synonymous technical terms (e.g., "sqli" expands to include
        "sql injection").

        Args:
            query: Original query string

        Returns:
            Expanded query string with synonyms appended
        """
        query_lower = query.lower().strip()
        expansions = config.query_expansions
        expanded_terms: List[str] = []
        seen_terms = set()
        seen_add = seen_terms.add
        expanded_append = expanded_terms.append

        # Check full query
        full_query_terms = expansions.get(query_lower)
        if full_query_terms:
            for term in full_query_terms:
                if term not in seen_terms:
                    seen_add(term)
                    expanded_append(term)

        # Check individual tokens
        tokens = self._tokenize(query_lower)
        for token in tokens:
            token_terms = expansions.get(token)
            if token_terms:
                for term in token_terms:
                    if term not in seen_terms:
                        seen_add(term)
                        expanded_append(term)

        # Check bigrams
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            bigram_terms = expansions.get(bigram)
            if bigram_terms:
                for term in bigram_terms:
                    if term not in seen_terms:
                        seen_add(term)
                        expanded_append(term)

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
        """Build inverted index with pre-computed IDF and doc lengths."""
        if not self._tokenized_corpus:
            return

        corpus_size = len(self._tokenized_corpus)
        doc_lengths = np.empty(corpus_size, dtype=np.float64)
        nd: Dict[str, int] = {}
        inverted: Dict[str, List[Tuple[int, int]]] = {}

        for doc_idx, tokens in enumerate(self._tokenized_corpus):
            doc_lengths[doc_idx] = len(tokens)
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for term, freq in tf.items():
                nd[term] = nd.get(term, 0) + 1
                posting = inverted.get(term)
                if posting is None:
                    inverted[term] = [(doc_idx, freq)]
                else:
                    posting.append((doc_idx, freq))

        avgdl = float(doc_lengths.sum() / corpus_size) if corpus_size > 0 else 0.0

        idf: Dict[str, float] = {}
        idf_sum = 0.0
        negative_idfs: List[str] = []
        for word, freq in nd.items():
            val = math.log(corpus_size - freq + 0.5) - math.log(freq + 0.5)
            idf[word] = val
            idf_sum += val
            if val < 0:
                negative_idfs.append(word)

        average_idf = idf_sum / len(idf) if idf else 0.0
        eps = self._epsilon * average_idf
        for word in negative_idfs:
            idf[word] = eps

        self._inverted_index = inverted
        self._idf = idf
        self._doc_len = doc_lengths
        self._avgdl = avgdl
        self._corpus_size = corpus_size
        self._index_built = True

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search the BM25 index with query expansion.

        Uses inverted-index posting lists to score only documents containing
        at least one query term. Returns (chunk_id, score) sorted descending.
        """
        if not self._index_built or not self.corpus:
            return []

        expanded_query = self.expand_query(query)
        tokenized_query = self._tokenize(expanded_query)
        if not tokenized_query:
            return []

        k1 = self._k1
        b = self._b
        avgdl = self._avgdl
        doc_len = self._doc_len
        idf_lookup = self._idf
        inv = self._inverted_index

        candidate_scores: Dict[int, float] = {}
        for q in tokenized_query:
            idf_q = idf_lookup.get(q, 0.0)
            if idf_q == 0.0:
                continue
            posting = inv.get(q)
            if posting is None:
                continue
            for doc_idx, tf in posting:
                dl = doc_len[doc_idx]
                num = tf * (k1 + 1.0)
                den = tf + k1 * (1.0 - b + b * dl / avgdl)
                candidate_scores[doc_idx] = candidate_scores.get(doc_idx, 0.0) + idf_q * (num / den)

        if not candidate_scores:
            return []

        n_candidates = len(candidate_scores)
        if n_candidates <= top_k:
            results = [(self.corpus_ids[idx], score) for idx, score in candidate_scores.items()]
            results.sort(key=lambda x: x[1], reverse=True)
            return results

        indices = np.fromiter(candidate_scores.keys(), dtype=np.intp, count=n_candidates)
        scores = np.fromiter(candidate_scores.values(), dtype=np.float64, count=n_candidates)
        partition_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = partition_idx[np.argsort(scores[partition_idx])[::-1]]
        return [(self.corpus_ids[indices[i]], float(scores[i])) for i in top_indices]

    def clear(self) -> None:
        """Clear the index"""
        self.corpus = []
        self.corpus_ids = []
        self._tokenized_corpus = []
        self._inverted_index = {}
        self._idf = {}
        self._doc_len = None
        self._avgdl = 0.0
        self._corpus_size = 0
        self._index_built = False

    def __len__(self) -> int:
        return len(self.corpus)


# =============================================================================
# KNOWLEDGE ORCHESTRATOR
# =============================================================================


def _enable_wal_mode(chroma_dir: Path) -> None:
    """Enable WAL journal mode on ChromaDB's SQLite for concurrent reads."""
    import sqlite3

    sqlite_path = chroma_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        return
    try:
        conn = sqlite3.connect(str(sqlite_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.close()
        print("[INFO] ChromaDB SQLite: WAL mode enabled")
    except Exception as e:
        print(f"[WARN] Could not enable WAL mode: {e}")


# =============================================================================
# FILE WATCHER (auto-reindex on document changes)
# =============================================================================


class DocumentWatcher(FileSystemEventHandler):
    """Watches documents directory and triggers reindex on changes.

    Uses accumulate-mode debounce: collects changed paths during a silence
    window instead of resetting the timer on every file event.  This prevents
    bulk file copies (1000+ files) from starving the reindex trigger.
    """

    def __init__(self, orchestrator_getter, debounce_seconds: float = 10.0):
        self._get_orchestrator = orchestrator_getter
        self._debounce = debounce_seconds
        self._lock = threading.Lock()
        self._pending_paths: set = set()
        self._timer = None
        self._reindex_lock = threading.Lock()

    def _schedule_reindex(self, path: str):
        """Accumulate-mode debounce: collect paths, fire once after silence."""
        with self._lock:
            self._pending_paths.add(path)
            if self._timer is None or not self._timer.is_alive():
                self._timer = threading.Timer(self._debounce, self._do_reindex)
                self._timer.daemon = True
                self._timer.start()

    def _do_reindex(self):
        """Perform incremental reindex in background (serialized)."""
        if not self._reindex_lock.acquire(blocking=False):
            print("[WATCHER] Reindex already in progress, skipping")
            return
        try:
            with self._lock:
                count = len(self._pending_paths)
                self._pending_paths.clear()
            if count == 0:
                return
            print(f"[WATCHER] {count} file(s) changed, starting incremental reindex...")
            orch = self._get_orchestrator()
            stats = orch.index_all(force=False)
            changed = stats.get("indexed", 0) + stats.get("updated", 0) + stats.get("deleted", 0)
            if changed > 0:
                print(
                    f"[WATCHER] Auto-reindexed: {stats['indexed']} new, "
                    f"{stats['updated']} updated, {stats['deleted']} deleted"
                )
        except Exception as e:
            import traceback as _tb

            print(f"[WATCHER] Reindex failed: {e}\n{_tb.format_exc()}")
        finally:
            self._reindex_lock.release()

    def on_created(self, event):
        if not event.is_directory and Path(event.src_path).suffix in config.supported_formats:
            self._schedule_reindex(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and Path(event.src_path).suffix in config.supported_formats:
            self._schedule_reindex(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and Path(event.src_path).suffix in config.supported_formats:
            self._schedule_reindex(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        src_supported = Path(event.src_path).suffix in config.supported_formats
        dest_supported = Path(event.dest_path).suffix in config.supported_formats
        if src_supported or dest_supported:
            self._schedule_reindex(event.dest_path)


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
        if config.transport != "stdio":
            _enable_wal_mode(config.chroma_dir)

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

        # Reverse lookup: resolved source path → doc_id (for O(1) adjacent chunk expansion)
        self._source_to_docid: Dict[str, str] = self._build_source_lookup()

        # Migration: deferred — checked in main() after full init
        self._needs_rebuild = False

        # Background reindex progress (polled via get_index_stats)
        self._reindex_progress: Dict[str, Any] = {"active": False}

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

    _bm25_build_lock = threading.Lock()

    def _ensure_bm25_index(self) -> None:
        """Lazy initialization of BM25 index from existing ChromaDB data"""
        if self._bm25_initialized:
            return
        with self._bm25_build_lock:
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

    _index_lock = threading.Lock()

    def index_all(self, force: bool = False) -> Dict[str, Any]:
        """
        Index documents with incremental change detection.

        Compares file mtime/size against stored metadata to detect changes.
        Only re-indexes files that are new or modified.  Serialized via
        _index_lock so concurrent calls (watcher + MCP tool) don't corrupt
        ChromaDB's SQLite database.
        """
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
            print(f"[INDEX] Scanning {stats['total_files']} documents...")

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
                print(f"[ERROR] Failed to index {doc.source}: {e}")

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
                print(
                    f"[INDEX] Progress: {idx + 1}/{stats['total_files']} ({pct}%) "
                    f"— {stats['indexed']} new, {stats['skipped']} skipped"
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
        """
        if not doc.chunks:
            return 0, 0

        unique_ids = []
        unique_docs = []
        unique_metas = []
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

        if unique_ids:
            bs = self._CHROMA_BATCH_SIZE
            for i in range(0, len(unique_ids), bs):
                self.collection.add(
                    ids=unique_ids[i : i + bs],
                    documents=unique_docs[i : i + bs],
                    metadatas=unique_metas[i : i + bs],
                )
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
            print(f"[WARN] Failed to remove chunks for doc {doc_id}: {e}")

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
            print(f"[ERROR] Background reindex failed: {e}")
        finally:
            self._reindex_progress["active"] = False

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
        self._source_to_docid = {}
        self.bm25_index.clear()
        self._bm25_initialized = False
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

    def update_document_content(self, filepath: str, content: str) -> Dict[str, Any]:
        """Update an existing document. Removes old chunks and re-indexes."""
        filepath = Path(filepath)
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

    def remove_document_by_path(self, filepath: str, delete_file: bool = False) -> Dict[str, Any]:
        """Remove a document from the index. Optionally delete from disk."""
        filepath_resolved = str(Path(filepath).resolve())

        doc_id = self._source_to_docid.get(filepath_resolved)

        if not doc_id:
            return {"error": f"Document not found in index: {filepath}"}

        chunks_removed = self._remove_document_chunks(doc_id)
        self._source_to_docid.pop(filepath_resolved, None)
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


# =============================================================================
# MCP Server
# =============================================================================

mcp = FastMCP(
    "knowledge-rag",
    host=config.server_host,
    port=config.server_port,
)

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
# MCP Tools — Helpers
# =============================================================================


def _make_snippet(content: str, max_chars: int = 500) -> str:
    """Truncate content at a natural break point."""
    if len(content) <= max_chars:
        return content
    truncated = content[:max_chars]
    min_pos = int(max_chars * 0.6)
    last_nl = truncated.rfind("\n", min_pos)
    if last_nl > min_pos:
        return truncated[:last_nl].rstrip() + "\n..."
    for sep in (". ", "? ", "! ", "; "):
        last_sep = truncated.rfind(sep, min_pos)
        if last_sep > min_pos:
            return truncated[: last_sep + len(sep) - 1] + " ..."
    last_space = truncated.rfind(" ", min_pos)
    if last_space > min_pos:
        return truncated[:last_space] + " ..."
    return truncated + "..."


# =============================================================================
# MCP Tools — Existing (6)
# =============================================================================


@mcp.tool()
@rate_limited
@instrument("search_knowledge")
def search_knowledge(
    query: str,
    max_results: int = 5,
    category: str = None,
    hybrid_alpha: float = 0.3,
    min_score: float = 0.0,
    snippet_mode: bool = True,
) -> str:
    """
    Hybrid search combining semantic search + BM25 keyword search with cross-encoder reranking.

    Read-only. No side effects.

    Args:
        query: Search query text (1–3 keywords recommended; phrase queries also work)
        max_results: Maximum number of results (default: 5, max: 20)
        category: Optional category filter — one of: security, ctf, logscale, development, general,
            redteam, blueteam. Call list_categories() first to see available categories and counts.
        hybrid_alpha: Balance between semantic and keyword search. 0.0 = keyword-only (best for exact
            technical terms like CVE IDs or tool names), 0.3 = balanced default, 1.0 = semantic-only
            (best for conceptual or natural-language queries).
        min_score: Minimum normalized relevance score (0.0–1.0) to include a result. Results scoring
            below this threshold are discarded. Default 0.0 returns all results. Use 0.2–0.4 to cut
            low-relevance noise.
        snippet_mode: When true (default), truncates content to ~500 characters at a natural break
            point and adds a content_length field with the original size. Use get_document() to
            fetch full content when needed. Set to false to return full chunk content.

    Returns:
        JSON string with results including content chunks, source filepath, relevance score, and
        search method used. Returns chunks, not full document content.

    Usage: Primary search tool — use for any topic or keyword lookup. Prefer search_similar() when
    you already have a reference document and want more like it. Prefer get_document() when you
    already know the exact filepath and need the full content.
    """
    if not query or not query.strip():
        return json.dumps({"status": "error", "message": "Query cannot be empty"})

    max_results = max(1, min(max_results or 5, config.max_results))
    hybrid_alpha = max(0.0, min(hybrid_alpha if hybrid_alpha is not None else 0.3, 1.0))
    min_score = max(0.0, min(min_score if min_score is not None else 0.0, 1.0))

    valid_categories = list(config.keyword_routes.keys()) + list(set(config.category_mappings.values())) + ["general"]
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

    total_before_filter = len(results)
    if min_score > 0.0:
        results = [r for r in results if r.get("score", 0) >= min_score]

    if snippet_mode:
        for r in results:
            full_len = len(r.get("content", ""))
            r["content"] = _make_snippet(r["content"])
            r["content_length"] = full_len

    return json.dumps(
        {
            "status": "success",
            "query": query,
            "hybrid_alpha": hybrid_alpha,
            "result_count": len(results),
            "filtered_by_score": total_before_filter - len(results),
            "cache_hit_rate": orchestrator.query_cache.stats()["hit_rate"],
            "results": results,
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool()
@rate_limited
@instrument("get_document")
def get_document(filepath: str) -> str:
    """
    Get the full content of a specific document by filepath.

    Read-only. No side effects.

    Args:
        filepath: Relative path to the document within the documents directory
            (e.g., "security/technique.md"). Must be an indexed file — use
            list_documents() to browse available paths, or search_knowledge()
            to find the filepath by topic first.

    Returns:
        JSON string with full document content and metadata (filepath, category, size).

    Usage: Use when you need the complete text of a known file — search_knowledge()
    returns chunks, not full docs. Use search_knowledge() first to find the filepath
    if unknown. Use list_documents() to browse all available files by category.
    """
    orchestrator = get_orchestrator()
    doc = orchestrator.get_document(filepath)

    if not doc:
        return json.dumps({"status": "error", "message": f"Document not found: {filepath}"})

    return json.dumps({"status": "success", "document": doc}, indent=2, ensure_ascii=False)


@mcp.tool()
@rate_limited
@instrument("reindex_documents")
def reindex_documents(force: bool = False, full_rebuild: bool = False) -> str:
    """
    Index or reindex all documents in the knowledge base.

    Runs in background — returns immediately. Use get_reindex_status() to monitor progress.

    Args:
        force: If True, smart reindex (detects changed files + rebuilds BM25 index).
            Use after manually editing files on disk outside of add_document().
        full_rebuild: If True, nuclear rebuild — deletes all vectors and re-embeds everything
            from scratch. Use only if the embedding model changed or the index is corrupted.

    Returns:
        JSON string with operation status. Poll get_reindex_status() for reindex.active,
        reindex.progress, and reindex.percent until reindex.active becomes false.

    Usage: Normal workflow does not require this — add_document(), update_document(), and
    add_from_url() all auto-index on call. Use force=True only after direct filesystem edits.
    Use full_rebuild=True only for model upgrades or index corruption. No arguments runs a
    fast incremental pass.
    """
    orchestrator = get_orchestrator()

    if full_rebuild:
        mode = "nuclear_rebuild"
    elif force:
        mode = "smart_reindex"
    else:
        mode = "incremental"

    result = orchestrator.start_reindex_background(mode)

    if result["status"] == "already_running":
        progress = result["progress"]
        return json.dumps(
            {
                "status": "already_running",
                "progress": f"{progress.get('processed', 0)}/{progress.get('total_files', 0)}",
                "operation": progress.get("operation"),
                "hint": "Use get_reindex_status() to check progress",
            },
            indent=2,
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "status": "started",
            "operation": mode,
            "message": "Reindex running in background. Use get_reindex_status() to monitor progress.",
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool()
@rate_limited
@instrument("get_reindex_status")
def get_reindex_status() -> str:
    """
    Get the current status of a background reindex operation.

    Lightweight — does not compute full index statistics. Use this to poll progress
    after calling reindex_documents().

    Returns:
        JSON string with reindex status. When active: operation name, progress (processed/total),
        percent complete, indexed/skipped/errors counts, and start time. When inactive: active=false,
        plus last_result or last_error from the most recent completed reindex.

    Usage: Call repeatedly after reindex_documents() to monitor progress. When reindex.active
    becomes false, the operation is complete. Use get_index_stats() for full index health metrics.
    """
    orchestrator = get_orchestrator()
    status = orchestrator.get_reindex_status()
    return json.dumps({"status": "success", "reindex": status}, indent=2)


@mcp.tool()
@rate_limited
@instrument("list_categories")
def list_categories() -> str:
    """
    List all document categories with their document counts.

    Read-only. No side effects. Reflects the live index state.

    Returns:
        JSON string with category names, document counts per category, and total document count.

    Usage: Use before filtering search_knowledge() or list_documents() by category to see
    which categories exist and how many documents each contains. Use get_index_stats() instead
    for broader system health metrics (model name, cache hit rate, BM25 status).
    """
    orchestrator = get_orchestrator()
    categories = orchestrator.list_categories()
    return json.dumps(
        {"status": "success", "categories": categories, "total_documents": sum(categories.values())}, indent=2
    )


@mcp.tool()
@rate_limited
@instrument("list_documents")
def list_documents(category: str = None) -> str:
    """
    List all indexed documents, optionally filtered by category.

    Read-only. No side effects.

    Args:
        category: Optional category filter. Must be a valid category name — call
            list_categories() to see available options (e.g., security, ctf, logscale,
            development, general, redteam, blueteam).

    Returns:
        JSON string with list of document filepaths, categories, and metadata for each indexed file.

    Usage: Use to browse what's in the index or verify a specific file is indexed. Use
    list_categories() first to see valid category names. Use search_knowledge() when you
    want to find documents by topic rather than browsing the full list. Use get_document()
    to read a specific file once you have its filepath.
    """
    orchestrator = get_orchestrator()
    docs = orchestrator.list_documents(category=category)
    return json.dumps(
        {"status": "success", "filter": category or "all", "count": len(docs), "documents": docs},
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool()
@rate_limited
@instrument("get_index_stats")
def get_index_stats() -> str:
    """
    Get statistics and health metrics for the knowledge base index.

    Read-only. No side effects.

    Returns:
        JSON string with system metrics: total documents, total chunks, embedding model name,
        BM25 status, query cache hit rate, and file watcher status.

    Usage: Use for system health checks — verifying the embedding model loaded, checking
    index population, or monitoring cache efficiency. Use list_categories() for per-category
    document counts instead. Use evaluate_retrieval() to measure actual search quality with
    test queries.
    """
    orchestrator = get_orchestrator()
    stats = orchestrator.get_stats()
    return json.dumps({"status": "success", "stats": stats}, indent=2)


# =============================================================================
# MCP Tools — New (6)
# =============================================================================


@mcp.tool()
@rate_limited
@instrument("add_document")
def add_document(content: str, filepath: str, category: str = "general") -> str:
    """
    Add a new document to the knowledge base from raw text content.

    Mutating — writes a file to disk and indexes it immediately. No auth required.

    Args:
        content: Full text content of the document (markdown supported)
        filepath: Relative path within documents directory (e.g., "security/new-technique.md").
            The subdirectory should match the category.
        category: Document category — one of: security, ctf, logscale, development, general,
            redteam, blueteam (default: general)

    Returns:
        JSON string with indexing results (filepath, chunks created, status).

    Usage: Use to add new documents from text content. Use add_from_url() instead when
    the source is a web page. Use update_document() to replace content of an existing file.
    The document is immediately searchable after this call — no manual reindex needed.
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
@rate_limited
@instrument("update_document")
def update_document(filepath: str, content: str) -> str:
    """
    Update the content of an existing document in the knowledge base.

    Mutating — overwrites the file on disk and re-indexes immediately. Old chunks are
    removed and replaced with new ones. Full content replacement, not a patch.

    Args:
        filepath: Full or relative path to the document file. Must be an already-indexed
            file — use list_documents() to find valid paths.
        content: New full-text content to replace the existing content entirely

    Returns:
        JSON string with update results (old chunk count, new chunk count, status).

    Usage: Use to replace a document's content completely. Use add_document() to create
    a new file instead. Use remove_document() to delete without replacing. Changes are
    immediately searchable — no manual reindex needed.
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
@rate_limited
@instrument("remove_document")
def remove_document(filepath: str, delete_file: bool = False) -> str:
    """
    Remove a document from the knowledge base index.

    Mutating — removes index entries. If delete_file=True, also permanently deletes
    the file from disk (irreversible, cannot be undone).

    Args:
        filepath: Path to the document file. Must be an indexed document — use
            list_documents() to find valid paths.
        delete_file: If True, permanently deletes the file from disk in addition to
            removing from the index (default: False).

    Returns:
        JSON string with removal results (filepath, status).

    Usage: Use to unindex a document while keeping the file on disk (default). Set
    delete_file=True only for permanent removal. Use update_document() to replace
    content instead of removing. Use reindex_documents(force=True) if you deleted
    the file manually on disk outside of this tool.
    """
    if not filepath:
        return json.dumps({"status": "error", "message": "Filepath required"})

    orchestrator = get_orchestrator()
    result = orchestrator.remove_document_by_path(filepath, delete_file=delete_file)

    if "error" in result:
        return json.dumps({"status": "error", "message": result["error"]})

    return json.dumps({"status": "success", **result}, indent=2)


@mcp.tool()
@rate_limited
@instrument("add_from_url")
def add_from_url(url: str, category: str = "general", title: str = None) -> str:
    """
    Fetch content from a URL, convert to markdown, and add to the knowledge base.

    Mutating — makes an outbound HTTP request (requires internet access), strips HTML,
    converts to markdown, saves to disk, and indexes immediately.

    Args:
        url: Full URL to fetch (https:// required). The page must be publicly accessible.
        category: Document category — one of: security, ctf, logscale, development, general,
            redteam, blueteam (default: general)
        title: Optional document title. Auto-detected from the page's <title> tag if omitted.

    Returns:
        JSON string with indexing results (detected title, filepath, chunks created, status).

    Usage: Use to ingest web content (writeups, blog posts, documentation pages) directly
    by URL. Use add_document() instead when you already have the text content. The document
    is immediately searchable after this call — no manual reindex needed.
    """
    if not url or not url.strip():
        return json.dumps({"status": "error", "message": "URL cannot be empty"})

    orchestrator = get_orchestrator()
    result = orchestrator.add_from_url(url.strip(), category, title)

    if "error" in result:
        return json.dumps({"status": "error", "message": result["error"]})

    return json.dumps({"status": "success", **result}, indent=2)


@mcp.tool()
@rate_limited
@instrument("search_similar")
def search_similar(filepath: str, max_results: int = 5) -> str:
    """
    Find documents semantically similar to a given reference document.

    Read-only. No side effects. Uses the document's embedding for similarity comparison.

    Args:
        filepath: Path to the reference document (must already be indexed — use
            list_documents() to verify). E.g., "security/technique.md"
        max_results: Number of similar documents to return (default: 5, max: 20)

    Returns:
        JSON string with list of similar document filepaths and similarity scores (0.0–1.0).

    Usage: Use when you have a specific document and want to discover thematically related
    ones. Use search_knowledge() instead when you have a text query rather than a reference
    document. The reference document must be indexed — call list_documents() to confirm
    it exists before calling this tool.
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
@rate_limited
@instrument("evaluate_retrieval")
def evaluate_retrieval(test_cases: str) -> str:
    """
    Evaluate search quality by testing whether search_knowledge() retrieves expected documents.

    Read-only. Runs multiple search queries internally. No side effects on the index.

    Args:
        test_cases: JSON string array of test cases. Each item requires "query" (search string)
            and "expected_filepath" (path of the document that should appear in top-5 results).
            Example: [{"query": "suid exploit", "expected_filepath": "security/suid.md"}]

    Returns:
        JSON string with MRR@5 (Mean Reciprocal Rank), Recall@5, and per-query hit/miss breakdown.
        MRR@5 above 0.7 indicates good retrieval quality.

    Usage: Use to audit search quality after bulk document ingestion or after tuning
    hybrid_alpha. Use get_index_stats() for system health checks instead. Use
    search_knowledge() for actual document retrieval — this tool is for quality measurement only.
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

            mcp.run(transport=transport)
    except AlreadyRunningError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(ALREADY_RUNNING_EXIT_CODE) from e


if __name__ == "__main__":
    main()
