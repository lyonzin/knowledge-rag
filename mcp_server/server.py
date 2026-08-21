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
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ChromaDB
import chromadb

# BM25 scoring (custom inverted-index, replaces rank_bm25 full-corpus scan)
import numpy as np

# FastEmbed for ONNX embeddings + reranker
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

# MCP Anthropic Tier 1 SDK v2.x (speaks the MCP spec 2026-07-28 natively).
# FastMCP was renamed to MCPServer in mcp 2.0.0; see docs/migration.md
# upstream. Same `@mcp.tool()` / `@mcp.resource()` / `@mcp.prompt()`
# ergonomics — only the import path + class name change.
from mcp.server import MCPServer
from watchdog.events import FileSystemEventHandler

# File watcher for auto-reindex
from watchdog.observers import Observer

# Local imports
from .config import config
from .fts5_index import Fts5LexicalIndex, Fts5NotReadyError
from .ingestion import Document, DocumentParser
from .metrics import (
    FAST_PATH_ERRORS_TOTAL,
    FAST_PATH_FALLBACK_TOTAL,
    FAST_PATH_HITS_TOTAL,
    FAST_PATH_LATENCY_SECONDS,
    FAST_PATH_MIGRATION_DOCS_INDEXED,
    FAST_PATH_MIGRATION_DOCS_TOTAL,
    FAST_PATH_RERANK_SKIPPED_TOTAL,
    get_metrics,
    instrument,
)
from .query_router import QueryRouter
from .ratelimit import rate_limited
from .security import (
    BearerAuthMiddleware,
    PathEscapeError,
    is_path_within,
    sanitize_external_content,
    validate_path_within,
)

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

    def _make_key(
        self,
        query: str,
        max_results: int,
        category: Optional[str],
        hybrid_alpha: float,
        search_method: str = "auto",
    ) -> str:
        """Generate cache key from query parameters.

        ``search_method`` defaults to ``"auto"`` so calls that omit it hash the
        same as pre-v4.8.2 callers (backward compat for internal callers not
        yet aware of the FTS5 fast-path dispatch).
        """
        raw = f"{query}|{max_results}|{category}|{hybrid_alpha}|{search_method}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(
        self,
        query: str,
        max_results: int,
        category: Optional[str],
        hybrid_alpha: float,
        search_method: str = "auto",
    ) -> Optional[Any]:
        """Get cached result if exists and not expired"""
        key = self._make_key(query, max_results, category, hybrid_alpha, search_method)

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

    def put(
        self,
        query: str,
        max_results: int,
        category: Optional[str],
        hybrid_alpha: float,
        result: Any,
        search_method: str = "auto",
    ) -> None:
        """Store result in cache.

        ``search_method`` is appended (with default ``"auto"``) rather than
        inserted before ``result`` so pre-v4.8.2 positional callers keep
        working — the FTS5 dispatch passes it as a keyword argument.
        """
        key = self._make_key(query, max_results, category, hybrid_alpha, search_method)
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
    def _print_gpu_banner(status: Optional[GPUStatus], mode: str) -> None:
        """Print a concise GPU diagnostic banner at startup.

        Called on EVERY startup path (v4.8.0+), including CPU-only and
        fallback, so operators always see which mode ran and why. Prints
        to stderr (print() is redirected there during init).

        ``status`` is None when gpu_mode="false" (no probe performed).
        ``mode`` is one of forced-cpu | forced-cuda | forced-cuda-fallback
        | auto-cuda | auto-cpu-fallback.
        """
        active = status is not None and status.available and mode in ("auto-cuda", "forced-cuda")
        print("")
        print("=" * 60)
        if active:
            FastEmbedEmbeddings._print_gpu_active(status, mode)
        else:
            FastEmbedEmbeddings._print_gpu_unavailable(status, mode)
        print("=" * 60)
        print("")

    @staticmethod
    def _print_gpu_active(status: GPUStatus, mode: str) -> None:
        """Emit the ACTIVE branch of the GPU banner (probe passed + provider used)."""
        print("  GPU STATUS: ACTIVE")
        print(f"  Provider:   {status.provider}")
        if status.device_name:
            print(f"  Device:     {status.device_name}")
        if status.vram_mb > 0:
            vram_display = f"{status.vram_mb / 1024:.1f} GB" if status.vram_mb >= 1024 else f"{status.vram_mb} MB"
            print(f"  VRAM:       {vram_display}")
        print(f"  Mode:       {mode}")

    @staticmethod
    def _print_gpu_unavailable(status: Optional[GPUStatus], mode: str) -> None:
        """Emit the UNAVAILABLE branch (forced CPU, probe failed, or load fallback)."""
        print("  GPU STATUS: UNAVAILABLE — running on CPU")
        if status is not None and status.fallback_reason:
            print(f"  Reason:     {status.fallback_reason}")
        if status is not None and status.missing_deps:
            print("  Missing:")
            for dep in status.missing_deps:
                print(f"    - {dep}")
        print(f"  Mode:       {mode}")
        if mode != "forced-cpu":
            print("  Hint:       pip install onnxruntime-gpu --extra-index-url \\")
            print(
                "              https://aiinfra.pkgs.visualstudio.com/PublicPackages"
                "/_packaging/onnxruntime-cuda-12/pypi/simple/"
            )
            print("              plus nvidia-cudnn-cu12, nvidia-cublas-cu12, nvidia-cuda-runtime-cu12")

    def __init__(self, model: str = None):
        self.model_name = model or config.embedding_model
        self._dim = config.embedding_dim
        # Build kwargs once; defer the heavy TextEmbedding(**kwargs) call to first use.
        self._init_kwargs = {"model_name": self.model_name, "cache_dir": str(config.models_cache_dir)}
        # v4.8.0+: tri-state mode drives the load routing. Legacy bool alias kept for BC.
        self._gpu_mode = getattr(config, "gpu_mode", "auto")
        self._gpu = bool(config.gpu_acceleration)
        self._model: Optional[TextEmbedding] = None
        self._load_lock = threading.Lock()
        # Sticky failure flag: once load fails, subsequent calls re-raise immediately
        # instead of looping through download/retry. Same pattern as CrossEncoderReranker.
        self._load_failed: Optional[Exception] = None

    def _load_model(self) -> None:
        """Load the ONNX model on demand. Idempotent and thread-safe.

        Routes via config.gpu_mode (v4.8.0+):
          "false" — CPU-only, no probe (zero startup overhead).
          "auto"  — probe verify_gpu_readiness(); CUDA if ready, CPU otherwise.
          "true"  — force CUDA; fall back to CPU only if actual load fails.

        Raises:
            EmbeddingModelLoadError: sticky failure — subsequent calls re-raise
                without retrying so callers don't loop through HF downloads.
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
            try:
                self._route_load()
            except Exception as exc:
                # ONNXRuntimeError, FileNotFoundError, etc. — record and re-raise loud
                self._load_failed = exc
                self._model = None
                print(f"[ERROR] Embedding model load FAILED: {exc}", file=sys.stderr)
                raise EmbeddingModelLoadError(f"Failed to load embedding model: {exc}") from exc

    def _route_load(self) -> None:
        """Route model load per gpu_mode. Called under _load_lock by _load_model."""
        mode = self._gpu_mode
        if mode == "false":
            self._load_forced_cpu()
        elif mode == "auto":
            self._load_auto()
        else:
            self._load_forced_cuda()

    def _load_forced_cpu(self) -> None:
        """gpu_mode='false' path — CPU only, no CUDA probing."""
        self._load_with_providers(["CPUExecutionProvider"], label="CPU")
        self._print_gpu_banner(status=None, mode="forced-cpu")

    def _load_auto(self) -> None:
        """gpu_mode='auto' path — probe GPU, use it if ready else CPU fallback."""
        self._setup_cuda_dll_paths()
        gpu_status = self.verify_gpu_readiness()
        if gpu_status.available:
            try:
                self._load_with_providers(["CUDAExecutionProvider", "CPUExecutionProvider"], label="GPU auto")
                self._print_gpu_banner(status=gpu_status, mode="auto-cuda")
                return
            except (ValueError, RuntimeError) as e:
                print(f"[WARN] GPU probe passed but load failed ({e}); loading on CPU...")
        self._load_with_providers(["CPUExecutionProvider"], label="CPU fallback")
        self._print_gpu_banner(status=gpu_status, mode="auto-cpu-fallback")

    def _load_forced_cuda(self) -> None:
        """gpu_mode='true' path — require CUDA, CPU only as last-resort fallback."""
        self._setup_cuda_dll_paths()
        gpu_status = self.verify_gpu_readiness()
        if gpu_status.available:
            try:
                self._load_with_providers(["CUDAExecutionProvider", "CPUExecutionProvider"], label="GPU forced")
                self._print_gpu_banner(status=gpu_status, mode="forced-cuda")
                return
            except (ValueError, RuntimeError) as e:
                print(f"[WARN] gpu: true but CUDA load failed ({e}); loading on CPU...")
        else:
            print(f"[WARN] gpu: true but GPU not ready ({gpu_status.fallback_reason}); loading on CPU")
        self._load_with_providers(["CPUExecutionProvider"], label="CPU fallback")
        self._print_gpu_banner(status=gpu_status, mode="forced-cuda-fallback")

    def _load_with_providers(self, providers: List[str], label: str) -> None:
        """Instantiate TextEmbedding with the given ONNX providers list."""
        kwargs = dict(self._init_kwargs)
        kwargs["providers"] = providers
        print(f"[INFO] Loading embedding model: {self.model_name} ({self._dim}D) [{label}]...")
        self._model = TextEmbedding(**kwargs)
        print(f"[INFO] Embedding model loaded successfully [{label}]")

    @staticmethod
    def _apply_prefix(texts: List[str], prefix: str) -> List[str]:
        """Prepend ``prefix`` to each string in ``texts``.

        Returns the input list unchanged when ``prefix`` is falsy (empty
        string). This is deliberate: the default profile ``compact`` ships
        no prefix, so most callers should never allocate a new list here.
        """
        if not prefix:
            return texts
        return [f"{prefix}{t}" for t in texts]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Core embedding pipeline — load, run model, validate output.

        No prefix application; callers must prepend the right prefix
        (``query_prefix`` for queries, ``passage_prefix`` for passages)
        before invoking this.

        Raises:
            EmbeddingModelLoadError: when the model could not be loaded.
            EmbeddingError: when embedding generation fails or returns
                the wrong shape.

        Behavior note (changed in v3.8.1):
            Previously the caller (``__call__``) swallowed any exception and
            returned vectors of zeros (``[[0.0]*dim for _ in input]``). That
            silently corrupted the index — ChromaDB stored zero vectors as
            document embeddings, ``count()`` returned the right number of
            chunks, smart-reindex would skip them as "already indexed", and
            queries returned garbage similarity scores. Failures are now LOUD
            so the caller can surface the real error to the user.
        """
        if not texts:
            return []

        self._load_model()  # may raise EmbeddingModelLoadError
        try:
            embeddings = list(self._model.embed(texts))
        except Exception as exc:
            print(f"[ERROR] Embedding generation FAILED: {exc}", file=sys.stderr)
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

        # Sanity check: model returned the right number of vectors with the right dim
        if len(embeddings) != len(texts):
            raise EmbeddingError(f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}")
        result = [emb.tolist() for emb in embeddings]
        if result and len(result[0]) != self._dim:
            raise EmbeddingError(f"Embedding dim mismatch: expected {self._dim}, got {len(result[0])}")
        return result

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts (ChromaDB embedding_function).

        ChromaDB embedding_function interface: ``__call__(input: List[str]) -> List[List[float]]``.
        FastEmbed.embed() returns a generator, so ``_embed`` consumes it.

        Prefix scope (v4.8.0+):
            This method is treated as a **passage** path and applies
            ``config.passage_prefix``. ChromaDB itself uses ``embedding_function``
            for both ``add()`` and ``query()``, but in this codebase the query
            path (via ``orchestrator.query()`` → ``_do_semantic``) calls
            :meth:`embed_query` directly and bypasses ``__call__``. That leaves
            ``__call__`` running exclusively in the ingestion path, so passage
            prefix is the correct default.

        Raises:
            EmbeddingModelLoadError: when the model could not be loaded.
            EmbeddingError: when embedding generation fails after a successful load.
        """
        if not input:
            return []
        return self._embed(self._apply_prefix(input, config.passage_prefix))

    def name(self) -> str:
        """Return embedding function name (required by ChromaDB v1.4.0+)"""
        return f"fastembed-{self.model_name}"

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of passages (alias for ``__call__``).

        Passage prefix is applied via ``__call__`` — do NOT reapply here or
        the prefix will double up (once for the alias, once for ``__call__``).
        """
        return self(documents)

    def embed_query(self, input=None, **kwargs) -> List[List[float]]:
        """Embed query text(s) — applies ``config.query_prefix`` before embedding.

        Bypasses ``__call__`` (which applies ``passage_prefix``) so query and
        passage scopes never collide. Returns a list of embeddings — one per
        input text.
        """
        if isinstance(input, list):
            texts = input
        elif input is not None:
            texts = [input]
        else:
            texts = [kwargs.get("query", "")]
        return self._embed(self._apply_prefix(texts, config.query_prefix))


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


def _metadata_path_score(query: str, metadata: Dict[str, Any]) -> float:
    """Return a small generic score boost when query terms match path metadata."""
    query_terms = re.findall(r"[^\W_]+(?:-[^\W_]+)*", query.lower())
    if not query_terms:
        return 0.0

    source = str(metadata.get("source", ""))
    filename = str(metadata.get("filename", ""))
    path_text = f"{source} {filename}".lower()
    path_tokens = set(re.findall(r"[^\W_]+(?:-[^\W_]+)*", path_text))
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
        """Tokenize: lowercase, split on non-alphanumeric, emit composite + sub-tokens.

        For alphanumeric-code composites (containing at least one digit — e.g.
        "mdr-ad002", "cve-2024-1234", "ms17-010"), emit both the composite token
        AND its sub-parts of length >= 2. This enables fragment queries ("AD002",
        "CVE", "010") to match while IDF preserves exact-match ranking (composite
        is rarer).

        Natural-language hyphenated words (e.g. "pass-the-hash", "state-of-the-art")
        are kept as single tokens — expanding them would flood the inverted index
        with high-frequency stop-word-like sub-parts and hurt query throughput
        without helping recall for typical use cases.
        """
        text_lower = text.lower()
        composite_tokens = re.findall(r"[^\W_]+(?:-[^\W_]+)*", text_lower)
        tokens: List[str] = []
        for tok in composite_tokens:
            tokens.append(tok)
            # Only expand codes (composites containing at least one digit).
            # Skips natural-language phrases like "pass-the-hash".
            if "-" in tok and any(c.isdigit() for c in tok):
                for part in tok.split("-"):
                    if len(part) >= 2:
                        tokens.append(part)
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


# Network filesystems where SQLite WAL is unsafe. WAL coordinates writers via a
# shared-memory (-shm) segment that NFS/SMB/CIFS either emulate poorly or forbid,
# so enabling WAL there risks silent corruption (chroma-core/chroma#7040 caveat).
_NETWORK_FSTYPES = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smbfs",
        "smb",
        "smb2",
        "smb3",
        "9p",
        "ncpfs",
        "afs",
        "afpfs",
        "fuse.sshfs",
        "fuse.glusterfs",
        "glusterfs",
        "lustre",
        "ceph",
        "beegfs",
    }
)


def _unescape_mount(field: str) -> str:
    """Decode the four octal escapes util-linux writes into ``/proc/mounts``.

    Only space (``\\040``), tab (``\\011``), newline (``\\012``) and backslash
    (``\\134``) are escaped; every other byte — including UTF-8 — is literal.
    ``codecs`` ``unicode_escape`` would mangle non-ASCII mountpoints (e.g.
    ``/mnt/données``), so it must not be used here. Backslash is decoded last
    so a real backslash cannot re-trigger the other rules.
    """
    return field.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def _network_fstype_match(target: str, mounts_text: str) -> bool:
    """Return True when ``target`` sits under a network mount in ``mounts_text``.

    ``mounts_text`` is the raw content of ``/proc/mounts`` (field 2 = mountpoint,
    field 3 = fstype). The longest mountpoint that prefixes ``target`` wins, so a
    network mount nested under a local root is still detected. Pure string logic,
    no I/O, so it is unit-testable without a real filesystem.
    """
    best_mount = ""
    best_fstype = ""
    for line in mounts_text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mountpoint = _unescape_mount(parts[1])
        fstype = parts[2]
        under_mount = target == mountpoint or target.startswith(mountpoint.rstrip("/") + "/")
        if under_mount and len(mountpoint) > len(best_mount):
            best_mount = mountpoint
            best_fstype = fstype
    return best_fstype.lower() in _NETWORK_FSTYPES


def _is_network_filesystem(path: Path) -> bool:
    """Best-effort, dependency-free detection of a network filesystem.

    Windows UNC paths (``\\\\server\\share``) are always network. On Linux we
    parse ``/proc/mounts``. Anything undetectable (macOS, no ``/proc``) falls
    through as local so local-disk behaviour is never regressed (fail-open).
    """
    raw = str(path)
    if raw.startswith("\\\\") or raw.startswith("//"):
        return True
    mounts = Path("/proc/mounts")
    if not mounts.exists():
        return False
    try:
        target = os.path.realpath(raw)
        return _network_fstype_match(target, mounts.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False


def _enable_wal_mode(chroma_dir: Path) -> None:
    """Switch ChromaDB's SQLite to WAL for concurrent reads — safely.

    GH #200: this MUST run *before* any ``chromadb.PersistentClient`` opens
    ``chroma_dir``. The Chroma Rust binding keeps a live sqlx connection pool,
    and toggling ``journal_mode`` through a second sqlite3 connection *after*
    that pool exists leaves stale ``-wal``/``-shm`` handles — the first
    ``Collection.add()`` then fails during compaction with SQLITE_NOTADB
    (error 26, "file is not a database"). ``journal_mode`` is sticky in the DB
    header, so one successful switch persists across restarts.

    Skips the switch on network filesystems (NFS/SMB/CIFS) where WAL is unsafe,
    and is idempotent: a database already in WAL is left untouched. When the DB
    does not exist yet it is pre-created as an empty WAL database so a brand-new
    install runs in WAL from its first process (Chroma opens it and runs its
    migrations over the empty WAL file — verified against chromadb 1.5.9).
    """
    import sqlite3

    if _is_network_filesystem(chroma_dir):
        print("[INFO] ChromaDB SQLite: network filesystem detected — keeping default journal mode (WAL unsafe)")
        return
    sqlite_path = chroma_dir / "chroma.sqlite3"
    fresh = not sqlite_path.exists()
    if fresh:
        chroma_dir.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(sqlite_path))
        try:
            current = conn.execute("PRAGMA journal_mode;").fetchone()
            if not fresh and current and str(current[0]).lower() == "wal":
                return  # already WAL (sticky header) — nothing to re-toggle
            conn.execute("PRAGMA journal_mode=WAL;")
            print("[INFO] ChromaDB SQLite: WAL mode enabled")
        finally:
            conn.close()
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


# Sentinel returned by _resolve_existing_or_skip to signal "skip this doc" —
# distinguishes from ``None`` (no existing doc yet, index as fresh).
_SKIP_DOC = object()


class KnowledgeOrchestrator:
    """Main orchestrator for knowledge retrieval with semantic search + keyword routing"""

    def __init__(self):
        # Instance-scoped reindex lock. MUST be per-instance — the previous
        # class-level `_index_lock` was silently shared across staging/main
        # orchestrators created by nuclear_rebuild (task 05 swap). When the
        # staging orch was garbage-collected while holding the lock, the class
        # attribute stayed acquired forever and every subsequent reindex
        # returned `reindex_already_running` despite `active: false`.
        self._index_lock = threading.Lock()

        # GH #161 (v4.8.3): _staging_target holds the staging collection
        # while populate is running. Write helpers dispatch through
        # ``_write_collection`` so writes hit staging but the query path
        # keeps reading ``self.collection`` (production) unchanged.
        # Zero-downtime is now real, not just a docs promise.
        self._staging_target = None

        self.parser = DocumentParser()
        self.embed_fn = FastEmbedEmbeddings()

        # Initialize ChromaDB with persistent storage (new API v1.4.0+).
        # GH #200: WAL is toggled inside _init_chroma_client BEFORE the client
        # opens the DB — never through a second connection afterwards.
        self.chroma_client = self._init_chroma_client()

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

        # v4.8.0 Fase 4: resume checkpoint file (written every 500 docs or 30s
        # during smart_reindex; opt-in loaded via reindex_documents(resume=True)).
        self._checkpoint_file = config.data_dir / "reindex_checkpoint.json"

        # Reverse lookup: resolved source path → doc_id (for O(1) adjacent chunk expansion)
        self._source_to_docid: Dict[str, str] = self._build_source_lookup()

        # Migration: deferred — checked in main() after full init
        self._needs_rebuild = False

        # FTS5 lexical fast-path (Task 03, ADR-006). Opt-in via YAML
        # ``search.lexical_fast_path.enabled: true``. When the toggle is off
        # both handles stay ``None`` and ``query()`` short-circuits before any
        # FTS5 dispatch runs — preserving v4.8.1 behaviour byte-for-byte.
        self.fts5_index: Optional[Fts5LexicalIndex] = None
        self.query_router: Optional[QueryRouter] = None
        if config.fts5_enabled:
            self._initialize_fts5_dispatch()

        # Background reindex progress (polled via get_index_stats).
        # v4.8.0 Fase 4 fields (chunks_processed/chunks_total/throughput_cps/
        # eta_seconds/checkpoint_saved_at) are populated on demand by the
        # background thread; absent when no reindex has ever run in this
        # process. Consumers must treat them as optional.
        self._reindex_progress: Dict[str, Any] = {"active": False}

        # v4.8.0 Fase 5: sweep any staging collections left behind by a
        # crashed rebuild (older than 24h). Idempotent + non-fatal on
        # error; the swap path also calls this before each rebuild.
        try:
            self._cleanup_stale_staging_collections()
        except Exception as e:
            print(f"[STAGING] Startup cleanup skipped (non-fatal): {e}")

    def _init_chroma_client(self):
        """Create the Chroma PersistentClient, switching SQLite to WAL first.

        GH #200: WAL must be toggled while no Rust sqlx handle is live. Doing it
        here — before ``chromadb.PersistentClient`` opens the DB — is the whole
        fix; see ``_enable_wal_mode`` for the SQLITE_NOTADB failure it prevents.
        ``stdio`` transport is single-process, so it keeps Chroma's default
        journal mode.
        """
        if config.transport != "stdio":
            _enable_wal_mode(config.chroma_dir)
        return chromadb.PersistentClient(path=str(config.chroma_dir))

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

        Prefers cached metadata (``self._indexed_docs[*]["embedding_dim"]``) to
        avoid triggering an embed on cold start — the previous behavior fired
        ``collection.query(query_texts=[...])`` which force-loads the model
        (bge-small ~1s CPU, but with GPU probe + CUDA init on v4.8.0 easily
        pushes the MCP handshake past its 30s timeout). Falls back to the
        query path only for legacy chunks indexed before this field existed;
        those chunks get ``embedding_dim`` backfilled on their next re-index.
        """
        if self.collection.count() == 0:
            return False
        # Fast path: check cached metadata (no embed, no model load)
        for entry in self._indexed_docs.values():
            cached_dim = entry.get("embedding_dim")
            if cached_dim is not None:
                if cached_dim != config.embedding_dim:
                    print(
                        f"[MIGRATION] Embedding dim mismatch (cached): "
                        f"stored={cached_dim} config={config.embedding_dim}"
                    )
                    print("[MIGRATION] Nuclear rebuild required.")
                    return True
                return False  # first cached hit is authoritative
        # Legacy fallback: no cached dim → query the collection (embeds once)
        try:
            self.collection.query(query_texts=["dimension check"], n_results=1, include=["documents"])
            return False
        except Exception as e:
            if "dimension" in str(e).lower():
                print(f"[MIGRATION] Embedding dimension mismatch detected: {e}")
                print("[MIGRATION] Nuclear rebuild required.")
                return True
            print(f"[WARN] Dimension check query failed (non-dimension error): {e}")
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

                # Batch to avoid SQLite "too many SQL variables" (chromadb 1.x
                # rebuilds IN(?, ?, ...) per row; single get(limit=count) blows
                # past the 999 default max_variable_number at ~48k chunks).
                batch_size = 500
                all_ids: list = []
                all_docs: list = []
                for offset in range(0, count, batch_size):
                    batch = self.collection.get(include=["documents"], limit=batch_size, offset=offset)
                    if not batch.get("ids"):
                        break
                    all_ids.extend(batch["ids"])
                    all_docs.extend(batch["documents"] or [])
                if not all_ids or not all_docs:
                    return

                self.bm25_index.add_documents(all_ids, all_docs)
                self.bm25_index.build_index()
                print(f"[INFO] BM25 index built with {len(self.bm25_index)} documents")
                self._bm25_initialized = True
            except Exception as e:
                print(f"[WARN] Failed to build BM25 index: {e}")
                # Do not mark initialized; retry allowed on next call.

    # =========================================================================
    # Indexing
    # =========================================================================

    def index_all(
        self,
        force: bool = False,
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Index documents with incremental change detection.

        Compares file mtime/size against stored metadata to detect changes.
        Only re-indexes files that are new or modified.  Serialized via
        _index_lock so concurrent calls (watcher + MCP tool) don't corrupt
        ChromaDB's SQLite database.

        When ``resume_state`` is provided (v4.8.0 Fase 4), docs whose id
        is in ``resume_state['doc_ids']`` are skipped and the chunk
        counter starts at ``resume_state['chunks_processed']`` — used by
        reindex_documents(resume=True) to pick up where the previous
        interrupted run left off.
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
            return self._index_all_impl(force, resume_state=resume_state)
        finally:
            self._index_lock.release()

    def _index_all_impl(
        self,
        force: bool = False,
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Inner implementation of index_all (caller holds _index_lock)."""
        stats = self._init_index_stats()
        documents = self._scan_and_count_documents(stats)
        path_to_docid = self._build_path_to_docid_map()
        self._prune_orphan_documents(documents, stats)

        tracking = self._init_reindex_tracking(resume_state, stats)
        _progress_interval = max(1, stats["total_files"] // 10)

        for idx, doc in enumerate(documents):
            self._process_one_document(idx, doc, force, path_to_docid, stats, tracking)
            self._update_progress_metrics(idx, stats, tracking)
            self._maybe_persist_checkpoint(idx, tracking)
            self._maybe_print_progress(idx, stats, _progress_interval)

        return self._finalize_reindex(stats, tracking)

    @staticmethod
    def _init_index_stats() -> Dict[str, Any]:
        """Fresh zeroed stats dict for a reindex run."""
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
        }

    def _scan_and_count_documents(self, stats: Dict[str, Any]) -> list:
        """Parse docs directory, publish total to progress, print if large."""
        documents = self.parser.parse_directory()
        stats["total_files"] = len(documents)
        self._reindex_progress["total_files"] = stats["total_files"]
        if stats["total_files"] > 100:
            print(f"[INDEX] Scanning {stats['total_files']} documents...")
        return documents

    def _build_path_to_docid_map(self) -> Dict[str, str]:
        """Reverse index: source path → doc_id from currently indexed docs."""
        path_to_docid: Dict[str, str] = {}
        for doc_id, info in list(self._indexed_docs.items()):
            path_to_docid[info.get("source", "")] = doc_id
        return path_to_docid

    def _prune_orphan_documents(self, documents, stats: Dict[str, Any]) -> None:
        """Delete indexed docs whose source path no longer exists (fixes #90).

        Done BEFORE indexing so moved files are not blocked by stale content
        hashes. Mutates ``stats`` (chunks_removed, deleted) and evicts from
        both ``_indexed_docs`` and ``_source_to_docid``.
        """
        current_paths = {str(doc.source) for doc in documents}
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

    def _init_reindex_tracking(self, resume_state: Optional[Dict[str, Any]], stats: Dict[str, Any]) -> Dict[str, Any]:
        """Build the mutable per-run tracking dict (checkpoint + throughput + resume)."""
        op_mode = self._reindex_progress.get("operation")
        resume_ids: set = set(resume_state.get("doc_ids", [])) if resume_state else set()
        chunks_processed = int(resume_state.get("chunks_processed", 0)) if resume_state else 0
        chunks_total_estimate = self._seed_chunks_total_estimate(stats)

        # GH #162 (v4.8.3): committed_this_run tracks only docs successfully
        # processed by this specific run. The previous checkpoint serialized
        # ``list(self._indexed_docs.keys())`` — every doc in the metadata,
        # including ones the current run had not yet reached. Resume then
        # skipped a changed-but-not-yet-processed doc solely by ID membership,
        # leaving stale vectors silently. Now checkpoint only writes IDs the
        # run actually committed, so resume mtime-checks anything unseen.
        # Prior ``resume_ids`` also fold in so the previous run's committed
        # doc list survives across interruptions.
        return {
            "op_mode": op_mode,
            "checkpoint_enabled": op_mode == "smart_reindex",
            "last_checkpoint_ts": time.monotonic(),
            "resume_doc_ids": resume_ids,
            "committed_this_run": set(resume_ids),
            "chunks_processed": chunks_processed,
            "throughput_window": deque(maxlen=100),
            "chunks_total_estimate": chunks_total_estimate,
        }

    def _seed_chunks_total_estimate(self, stats: Dict[str, Any]) -> int:
        """Warm-start estimate so the first status poll is meaningful. Refined per-doc later."""
        if self._indexed_docs:
            avg_chunks = sum(info.get("chunks", 0) for info in self._indexed_docs.values()) / max(
                len(self._indexed_docs), 1
            )
            chunks_total_estimate = int(avg_chunks * stats["total_files"])
        else:
            chunks_total_estimate = 0
        self._reindex_progress["chunks_total"] = chunks_total_estimate
        return chunks_total_estimate

    def _process_one_document(
        self,
        idx: int,
        doc,
        force: bool,
        path_to_docid: Dict[str, str],
        stats: Dict[str, Any],
        tracking: Dict[str, Any],
    ) -> None:
        """Per-doc worker: resume skip / hash check / evict-and-reindex / commit.

        Mutates ``stats`` and ``tracking['chunks_processed']``. Errors are
        caught and logged; the caller keeps iterating.
        """
        try:
            existing_doc_id = self._resolve_existing_or_skip(doc, force, path_to_docid, stats, tracking)
            if existing_doc_id is _SKIP_DOC:
                return

            chunks_added, dedup_skipped = self._index_document(doc)
            self._commit_indexed_doc(doc, chunks_added, dedup_skipped, existing_doc_id, force, stats, tracking)
        except Exception as e:
            stats["errors"] += 1
            print(f"[ERROR] Failed to index {doc.source}: {e}")

    def _resolve_existing_or_skip(
        self,
        doc,
        force: bool,
        path_to_docid: Dict[str, str],
        stats: Dict[str, Any],
        tracking: Dict[str, Any],
    ):
        """Decide the fate of a candidate doc.

        Returns ``_SKIP_DOC`` if the caller should skip (already indexed,
        resume-committed, or unchanged); otherwise returns the existing
        doc_id (may be None for a fresh doc). Evicts stale content when a
        content change is detected.
        """
        if tracking["resume_doc_ids"] and doc.id in tracking["resume_doc_ids"]:
            stats["skipped"] += 1
            return _SKIP_DOC

        existing_doc_id = path_to_docid.get(str(doc.source))
        if not force and existing_doc_id:
            if self._unchanged_since_last_index(doc, existing_doc_id):
                stats["skipped"] += 1
                return _SKIP_DOC
            self._evict_stale_doc(existing_doc_id, stats)
        elif not force and doc.id in self._indexed_docs:
            stats["skipped"] += 1
            return _SKIP_DOC
        return existing_doc_id

    def _commit_indexed_doc(
        self,
        doc,
        chunks_added: int,
        dedup_skipped: int,
        existing_doc_id: Optional[str],
        force: bool,
        stats: Dict[str, Any],
        tracking: Dict[str, Any],
    ) -> None:
        """Post-index bookkeeping: stats bump + throughput sample + metadata write."""
        if not (existing_doc_id and not force):
            stats["indexed"] += 1
        stats["chunks_added"] += chunks_added
        stats["dedup_skipped"] += dedup_skipped
        stats["categories"][doc.category] = stats["categories"].get(doc.category, 0) + 1
        # Only bump chunks_processed on success — chunks_added is
        # meaningful only when _index_document returned normally.
        tracking["chunks_processed"] += chunks_added
        tracking["committed_this_run"].add(doc.id)  # GH #162: only IDs this run committed go into the checkpoint
        self._register_indexed_doc(doc, chunks_added)

    def _unchanged_since_last_index(self, doc, existing_doc_id: str) -> bool:
        """True if the on-disk file matches the stored mtime + size."""
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
        return stored_mtime == current_mtime and stored_size == current_size

    def _evict_stale_doc(self, existing_doc_id: str, stats: Dict[str, Any]) -> None:
        """Remove chunks + metadata for a doc that needs reindexing. Bumps updated."""
        removed = self._remove_document_chunks(existing_doc_id)
        stats["chunks_removed"] += removed
        src = self._indexed_docs[existing_doc_id].get("source", "")
        if src:
            self._source_to_docid.pop(str(Path(src).resolve()), None)
        del self._indexed_docs[existing_doc_id]
        stats["updated"] += 1

    def _register_indexed_doc(self, doc, chunks_added: int) -> None:
        """Persist post-index metadata (mtime/size/chunk count) for a doc."""
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
            "embedding_dim": config.embedding_dim,  # avoids cold-start embed in _check_dimension_mismatch
        }
        self._source_to_docid[str(doc.source.resolve())] = doc.id

    def _update_progress_metrics(self, idx: int, stats: Dict[str, Any], tracking: Dict[str, Any]) -> None:
        """Compute throughput/ETA/refined total, publish to _reindex_progress."""
        chunks_processed = tracking["chunks_processed"]
        throughput_cps = self._sample_throughput(tracking, chunks_processed)
        chunks_total_estimate = self._refine_chunks_total(idx, chunks_processed, stats, tracking)
        eta_seconds = 0
        if throughput_cps > 0 and chunks_total_estimate > chunks_processed:
            eta_seconds = int((chunks_total_estimate - chunks_processed) / throughput_cps)
        self._reindex_progress.update(
            {
                "processed": idx + 1,
                "indexed": stats["indexed"],
                "skipped": stats["skipped"],
                "errors": stats["errors"],
                "chunks_processed": chunks_processed,
                "chunks_total": chunks_total_estimate,
                "throughput_cps": round(throughput_cps, 2),
                "eta_seconds": eta_seconds,
            }
        )

    @staticmethod
    def _sample_throughput(tracking: Dict[str, Any], chunks_processed: int) -> float:
        """Append current sample, prune >30s samples, return chunks/sec estimate.

        Sliding window: 100 samples (deque maxlen) OR last 30s (pruned).
        """
        throughput_window: deque = tracking["throughput_window"]
        now = time.monotonic()
        throughput_window.append((now, chunks_processed))
        while throughput_window and (now - throughput_window[0][0]) > 30:
            throughput_window.popleft()

        if len(throughput_window) < 2:
            return 0.0
        oldest_ts, oldest_cnt = throughput_window[0]
        dt = now - oldest_ts
        if dt <= 0:
            return 0.0
        return (chunks_processed - oldest_cnt) / dt

    @staticmethod
    def _refine_chunks_total(
        idx: int,
        chunks_processed: int,
        stats: Dict[str, Any],
        tracking: Dict[str, Any],
    ) -> int:
        """Refine chunks_total_estimate from rolling per-doc average. Stores back into tracking."""
        chunks_total_estimate = tracking["chunks_total_estimate"]
        if idx + 1 > 0 and chunks_processed > 0:
            running_avg = chunks_processed / (idx + 1)
            chunks_total_estimate = max(
                chunks_processed,
                int(running_avg * stats["total_files"]),
            )
            tracking["chunks_total_estimate"] = chunks_total_estimate
        return chunks_total_estimate

    def _maybe_persist_checkpoint(self, idx: int, tracking: Dict[str, Any]) -> None:
        """Write checkpoint every 500 docs OR 30s, whichever comes first.

        500 covers fast runs (small/cached docs), 30s covers slow runs (huge
        PDFs, network drive). Metadata is flushed alongside so a future
        resume=True sees the same doc_ids in both places — otherwise resume
        would skip docs missing from _indexed_docs and the collection drifts.
        """
        if not tracking["checkpoint_enabled"]:
            return
        due_by_count = (idx + 1) % 500 == 0
        due_by_time = (time.monotonic() - tracking["last_checkpoint_ts"]) >= 30
        if not (due_by_count or due_by_time):
            return
        try:
            self._save_metadata()
            # GH #162 (v4.8.3): serialize only IDs actually committed by this
            # run. Serializing ``list(self._indexed_docs.keys())`` conflated
            # "in metadata" with "processed this run" and hid changed docs on
            # resume.
            self._write_checkpoint(
                operation=tracking["op_mode"],
                indexed_doc_ids=list(tracking["committed_this_run"]),
                chunks_processed=tracking["chunks_processed"],
                started_at=self._reindex_progress.get("started_at"),
            )
            tracking["last_checkpoint_ts"] = time.monotonic()
        except OSError as e:
            # Non-fatal — a lost checkpoint just gives less to resume from.
            print(f"[WARN] Checkpoint write failed (non-fatal): {e}")

    def _maybe_print_progress(self, idx: int, stats: Dict[str, Any], progress_interval: int) -> None:
        """Emit periodic progress line for corpora larger than 100 docs."""
        if stats["total_files"] > 100 and (idx + 1) % progress_interval == 0:
            pct = int((idx + 1) / stats["total_files"] * 100)
            print(
                f"[INDEX] Progress: {idx + 1}/{stats['total_files']} ({pct}%) "
                f"— {stats['indexed']} new, {stats['skipped']} skipped"
            )

    def _finalize_reindex(self, stats: Dict[str, Any], tracking: Dict[str, Any]) -> Dict[str, Any]:
        """Flush metadata, clear checkpoint (if applicable), invalidate query cache."""
        self._save_metadata()

        # Checkpoint is no longer needed after a successful run — clear it so
        # the next reindex(resume=True) does not resume into a stale state.
        if tracking["checkpoint_enabled"]:
            self._clear_checkpoint()

        if stats["indexed"] > 0 or stats["updated"] > 0 or stats["deleted"] > 0:
            self.query_cache.invalidate()

        return stats

    # v4.8.0 Fase 3: kept as fallback constant for tests that instantiate
    # Orchestrator without a full Config (e.g. `object.__new__` mocks in
    # tests/test_search.py). Production reads `config.batch_size` first.
    _CHROMA_BATCH_SIZE = 500

    def _index_document(self, doc: Document) -> Tuple[int, int]:
        """Index a single document's chunks into ChromaDB and BM25 with dedup.

        Batching is controlled by ``config.batch_size`` (see
        ``_add_chunks_batched``). When ``config.parallel_workers > 1``
        the SQLite writes overlap with the NEXT batch's ONNX inference —
        NOT parallel inference (embedding kernel is serial). Default 1
        preserves single-threaded behavior byte-for-byte.
        """
        if not doc.chunks:
            return 0, 0

        unique_ids, unique_docs, unique_metas, dedup_skipped = self._dedup_chunks(doc)

        if unique_ids:
            self._add_chunks_batched(unique_ids, unique_docs, unique_metas)
            self.bm25_index.add_documents(unique_ids, unique_docs)

        return len(unique_ids), dedup_skipped

    @staticmethod
    def _dedup_chunks(doc: Document):
        """Deduplicate chunks by content SHA256 prefix; build parallel id/doc/meta lists."""
        unique_ids = []
        unique_docs = []
        unique_metas = []
        dedup_skipped = 0
        seen_hashes: set = set()

        for chunk in doc.chunks:
            content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:20]
            if content_hash in seen_hashes:
                dedup_skipped += 1
                continue
            seen_hashes.add(content_hash)
            unique_ids.append(f"{doc.id}_{chunk.index}")
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
        return unique_ids, unique_docs, unique_metas, dedup_skipped

    def _add_chunks_batched(self, ids, docs, metas) -> None:
        """Dispatch ChromaDB.add across batches (parallel path when workers > 1)."""
        bs = getattr(config, "batch_size", self._CHROMA_BATCH_SIZE)
        workers = getattr(config, "parallel_workers", 1)

        if workers > 1 and len(ids) > bs:
            self._add_chunks_parallel(ids, docs, metas, bs, workers)
        else:
            # Single-threaded path (default) — byte-identical to prior behavior.
            # GH #161 (v4.8.3): _write_collection routes to staging when a
            # nuclear rebuild is populating, else production. Queries keep
            # reading self.collection so users see zero-downtime.
            target = self._write_collection
            for i in range(0, len(ids), bs):
                target.add(
                    ids=ids[i : i + bs],
                    documents=docs[i : i + bs],
                    metadatas=metas[i : i + bs],
                )

    def _add_chunks_parallel(self, ids, docs, metas, bs: int, workers: int) -> None:
        """Parallel batch add — SQLite writes overlap with NEXT batch's inference.

        ONNX inference itself is serial inside the embedding kernel; the win
        is I/O overlap. First exception from .result() surfaces to the caller
        as an indexing failure (same shape as the sequential path).
        """
        from concurrent.futures import ThreadPoolExecutor

        # GH #161 (v4.8.3): route to staging via _write_collection during
        # nuclear rebuild. Snapshot the target once so all workers agree.
        target = self._write_collection
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    target.add,
                    ids=ids[i : i + bs],
                    documents=docs[i : i + bs],
                    metadatas=metas[i : i + bs],
                )
                for i in range(0, len(ids), bs)
            ]
            for f in futures:
                f.result()

    def _remove_document_chunks(self, doc_id: str) -> int:
        """Remove all chunks belonging to a document from ChromaDB and BM25.

        GH #161 (v4.8.3): routes through ``_write_collection`` so nuclear
        rebuild evictions land on staging, not production. Same doc_id may
        exist in both during populate — that's intentional; production keeps
        serving queries with the pre-rebuild state until swap.
        """
        target = self._write_collection
        try:
            results = target.get(where={"doc_id": doc_id}, include=[])

            if results["ids"]:
                target.delete(ids=results["ids"])
                self._bm25_initialized = False
                return len(results["ids"])
        except Exception as e:
            print(f"[WARN] Failed to remove chunks for doc {doc_id}: {e}")

        return 0

    def start_reindex_background(
        self,
        mode: str,
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Start reindex in a background thread. Returns immediately.

        When ``resume_state`` is provided (v4.8.0 Fase 4), it is forwarded
        to smart_reindex — the background thread skips the doc_ids from
        the previous run's checkpoint and continues chunk counting from
        where it stopped.
        """
        if self._reindex_progress.get("active"):
            return {"status": "already_running", "progress": dict(self._reindex_progress)}

        self._reindex_progress = self._fresh_reindex_progress(mode, resume_state)

        # GH #163 (v4.8.3): propagate force through smart_reindex so
        # reindex_documents(force=True) actually re-embeds files after a
        # prefix/model change (the pre-fix path skipped every unchanged
        # file). Nuclear_rebuild always re-embeds regardless.
        target = {
            "incremental": lambda: self.index_all(force=False),
            "smart_reindex": lambda: self.reindex_all(resume_state=resume_state, force=True),
            "nuclear_rebuild": self.nuclear_rebuild,
        }[mode]

        thread = threading.Thread(target=self._run_reindex, args=(target,), daemon=True)
        thread.start()
        return {"status": "started", "operation": mode}

    @staticmethod
    def _fresh_reindex_progress(mode: str, resume_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the initial _reindex_progress dict for a new background run."""
        return {
            "active": True,
            "operation": mode,
            "total_files": 0,
            "processed": 0,
            "indexed": 0,
            "skipped": 0,
            "errors": 0,
            "started_at": datetime.now().isoformat(),
            # v4.8.0 Fase 4: granular progress + resume checkpoint
            "chunks_processed": (int(resume_state.get("chunks_processed", 0)) if resume_state else 0),
            "chunks_total": 0,
            "throughput_cps": 0.0,
            "eta_seconds": 0,
            "checkpoint_saved_at": None,
            "resumed": bool(resume_state),
        }

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

    def reindex_all(
        self,
        resume_state: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Smart reindex: incremental detection + BM25 rebuild + orphan cleanup.

        When ``resume_state`` is provided (v4.8.0 Fase 4), docs listed in
        it are skipped so the interrupted previous run picks up where it
        stopped.

        ``force`` (v4.8.3, GH #163) propagates the caller's re-embed intent
        down to ``index_all``. The MCP ``reindex_documents(force=True)``
        migration path documented in ``docs/migration-v4.8.0.md`` relies on
        this to re-embed unchanged files after prefix/model changes.
        """
        import shutil

        if resume_state:
            print(
                f"[REINDEX] Resuming smart reindex from checkpoint "
                f"({len(resume_state.get('doc_ids', []))} docs already "
                f"processed, {resume_state.get('chunks_processed', 0)} chunks)"
            )
        elif force:
            print("[REINDEX] Starting FORCED smart reindex (re-embedding all files)...")
        else:
            print("[REINDEX] Starting smart incremental reindex...")
        start_time = time.time()

        stats = self.index_all(force=force, resume_state=resume_state)

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

    # =========================================================================
    # v4.8.0 Fase 5: zero-downtime rebuild via staging collection + atomic swap
    # =========================================================================

    # Sanity queries used by _validate_staging. These are corpus-agnostic
    # tokens that appear in >99% of technical text; a staging collection
    # that fails to return ANY hits for 4/5 of them is almost certainly
    # broken (bad embeddings, empty adds, corrupt writes).
    _STAGING_SANITY_QUERIES = ("readme", "function", "import", "return", "class")

    # Stale staging collections older than this are cleaned up on startup.
    # A rebuild takes minutes to hours; 24h is a very safe upper bound while
    # still preventing orphan accumulation from crashed rebuilds.
    _STAGING_TTL_SECONDS = 24 * 60 * 60

    def _cleanup_stale_staging_collections(self) -> Dict[str, int]:
        """Remove staging collections older than ``_STAGING_TTL_SECONDS``.

        Runs opportunistically at Orchestrator init and before each swap
        rebuild. Idempotent — safe to call repeatedly. Stagings younger than
        the TTL are preserved because they may belong to a rebuild in flight.
        Structured logging reports counts so operators can spot leaks.
        """
        prefix = f"{config.collection_name}__staging_"
        now = int(time.time())
        stats = {"scanned": 0, "removed": 0, "preserved": 0}
        try:
            existing = self.chroma_client.list_collections()
        except Exception as e:
            print(f"[STAGING] list_collections failed (non-fatal): {e}")
            return stats

        for coll in existing:
            self._process_staging_candidate(coll, prefix, now, stats)

        if stats["removed"] > 0 or stats["scanned"] > 0:
            print(
                f"[STAGING] Cleanup: scanned={stats['scanned']} "
                f"removed={stats['removed']} preserved={stats['preserved']}"
            )
        return stats

    def _process_staging_candidate(self, coll, prefix: str, now: int, stats: Dict[str, int]) -> None:
        """Classify one candidate and delete if past TTL. Mutates ``stats``.

        Non-matching names are silent skips. Names with the prefix but a
        non-integer timestamp are preserved — we don't delete anything we
        didn't create. TTL breach triggers a best-effort delete.
        """
        name = getattr(coll, "name", "")
        if not name.startswith(prefix):
            return
        stats["scanned"] += 1
        suffix = name[len(prefix) :]
        try:
            ts = int(suffix)
        except ValueError:
            stats["preserved"] += 1
            return
        if (now - ts) < self._STAGING_TTL_SECONDS:
            stats["preserved"] += 1
            return
        try:
            self.chroma_client.delete_collection(name)
            stats["removed"] += 1
            print(f"[STAGING] Cleaned up stale staging: {name}")
        except Exception as e:
            print(f"[STAGING] Failed to delete {name} (non-fatal): {e}")

    def _create_staging_collection(self, ts: int):
        """Create a fresh staging collection with the current embedding fn.

        Naming ``{collection_name}__staging_{unix_ts}`` — the timestamp
        prevents collisions with concurrent rebuilds and lets the cleanup
        helper age stale ones out. Same embedding_function as production so
        the swap is dimensionally compatible with the query pipeline that
        picks up right after.
        """
        staging_name = f"{config.collection_name}__staging_{ts}"
        return self.chroma_client.get_or_create_collection(
            name=staging_name,
            embedding_function=self.embed_fn,
            metadata={"description": "knowledge-rag v4.8.0 staging rebuild"},
        )

    def _populate_staging(self, staging) -> Dict[str, Any]:
        """Populate ``staging`` while ``self.collection`` keeps serving queries.

        GH #161 (v4.8.3): the previous implementation rebound
        ``self.collection = staging`` before populate, so concurrent queries
        arriving during the hours-long populate phase saw the empty staging
        collection instead of production. Now writes go through
        ``_write_collection`` (which returns ``self._staging_target`` when
        set) while reads on ``self.collection`` continue hitting production.
        Zero-downtime is finally atomic.

        BM25 / _indexed_docs / _source_to_docid are still rebound because
        those in-memory structures have no equivalent read/write split and
        their staging state is only committed on final swap. If populate or
        validate raises, ``_saved`` is restored so production sees zero net
        change.

        Returns the ``index_all`` stats dict on success. Rollback is the
        caller's responsibility on any exception raised out of here.
        """
        _saved = {
            "bm25_index": self.bm25_index,
            "_bm25_initialized": self._bm25_initialized,
            "_indexed_docs": dict(self._indexed_docs),
            "_source_to_docid": dict(self._source_to_docid),
        }
        self._staging_target = staging  # write dispatch redirects here
        self.bm25_index = BM25Index()
        self._bm25_initialized = True  # suppress lazy rebuild on staging
        self._indexed_docs = {}
        self._source_to_docid = {}
        try:
            return self.index_all(force=True)
        except Exception:
            self._staging_target = None
            for k, v in _saved.items():
                setattr(self, k, v)
            raise

    @property
    def _write_collection(self):
        """Return the collection writes should hit — staging if active, else prod.

        ``getattr`` default handles legacy code paths (tests that bypass
        ``__init__`` via ``patch.object(__init__)``) so old fixtures don't
        break — attribute is initialized to None in ``__init__`` for real
        instances.
        """
        target = getattr(self, "_staging_target", None)
        return target if target is not None else self.collection

    def _rollback_staging_state(self, saved: Dict[str, Any]) -> None:
        """Restore orchestrator state from a snapshot taken pre-populate."""
        for k, v in saved.items():
            setattr(self, k, v)

    def _snapshot_pre_staging(self) -> Dict[str, Any]:
        """Snapshot the orchestrator fields that ``_populate_staging`` mutates.

        Returned dict is opaque — feed it back to ``_rollback_staging_state``
        if validate/swap fails so production sees zero net change.
        """
        return {
            "collection": self.collection,
            "bm25_index": self.bm25_index,
            "_bm25_initialized": self._bm25_initialized,
            "_indexed_docs": dict(self._indexed_docs),
            "_source_to_docid": dict(self._source_to_docid),
        }

    def _validate_staging(self, staging, baseline_count: int) -> Dict[str, Any]:
        """Sanity-check the staging collection before swap via three gates.

        Gate 1 (count): within 10% of baseline — see ``_validate_staging_count``.
        Gates 2+3 (queries): canonical hits + backend integrity — see
        ``_validate_staging_canonical_queries``.

        Returns a dict with ``ok: bool`` plus per-gate stats for logging.
        """
        result: Dict[str, Any] = {
            "ok": False,
            "count": 0,
            "baseline": baseline_count,
            "min_expected": 0,
            "canonical_hits": 0,
            "query_error": None,
        }
        if not self._validate_staging_count(staging, baseline_count, result):
            return result
        if not self._validate_staging_canonical_queries(staging, baseline_count, result):
            return result

        result["ok"] = True
        return result

    def _validate_staging_count(self, staging, baseline_count: int, result: Dict[str, Any]) -> bool:
        """Gate 1 — count within 10% of baseline. Baseline 0 skips size check."""
        try:
            result["count"] = staging.count()
        except Exception as e:
            result["query_error"] = f"count failed: {e}"
            return False

        min_expected = int(baseline_count * 0.9) if baseline_count > 0 else 0
        result["min_expected"] = min_expected
        return result["count"] >= min_expected

    def _validate_staging_canonical_queries(self, staging, baseline_count: int, result: Dict[str, Any]) -> bool:
        """Gates 2 + 3 — canonical query sanity + backend integrity.

        Runs each canonical query; any raise fails gate 3 (backend corruption).
        Empty corpus (baseline 0) legitimately fails canonical queries and is
        allowed to pass since gate 1 already covered the size dimension.
        """
        hits = 0
        for q in self._STAGING_SANITY_QUERIES:
            try:
                r = staging.query(query_texts=[q], n_results=1, include=[])
                if r and r.get("ids") and r["ids"][0]:
                    hits += 1
            except Exception as e:
                result["query_error"] = f"query '{q}' failed: {e}"
                return False
        result["canonical_hits"] = hits

        if baseline_count > 0 and hits < 4:
            return False
        return True

    def _swap_collections_atomic(self, staging, prod_name: str, ts: int) -> None:
        """Two-step rename: prod → old, staging → prod, then delete old.

        ChromaDB's ``Collection.modify(name=)`` renames in place — no data
        movement. Race window between the two modifies is one Python
        statement (~microseconds); queries funnel through ``self.query``
        which reads the rebound ``self.collection`` after this returns.

        Rollback contract lives in ``_promote_staging_or_rollback``. This
        raises on any unrecoverable state so the caller aborts loudly.
        """
        old_name = f"{prod_name}__old_{ts}"
        prod = self.chroma_client.get_collection(prod_name)

        # Step 1: free the production name.
        prod.modify(name=old_name)

        # Step 2: staging assumes the production name (with rollback).
        self._promote_staging_or_rollback(staging, prod, prod_name, old_name)

        # Step 3: cleanup the old prod. Non-fatal — cleanup helper ages it out.
        try:
            self.chroma_client.delete_collection(old_name)
        except Exception as e:
            print(f"[SWAP] Failed to delete post-swap old prod (non-fatal): {e}")

    def _promote_staging_or_rollback(self, staging, prod, prod_name: str, old_name: str) -> None:
        """Rename staging to production; on failure, restore previous prod name.

        If both the rename AND the rollback fail, prod is left at ``old_name``
        and the caller aborts loudly (raise) so operators can recover manually.
        """
        try:
            staging.modify(name=prod_name)
        except Exception:
            try:
                prod.modify(name=prod_name)
            except Exception as inner:
                print(
                    f"[SWAP] CRITICAL: staging rename failed AND rollback "
                    f"failed. Prod is at '{old_name}'. Inner: {inner}"
                )
            raise

    def _rebuild_bm25_post_swap(self, prod_name: str) -> None:
        """Reconnect self.collection to the swapped prod + rebuild BM25.

        Must be called AFTER ``_swap_collections_atomic`` so the resolved
        collection has the freshly-written vectors. BM25 rebuild is done
        here rather than inside populate so production BM25 keeps serving
        queries throughout the rebuild window.
        """
        self.collection = self.chroma_client.get_collection(
            name=prod_name,
            embedding_function=self.embed_fn,
        )
        # GH #161 (v4.8.3): populate finished + swap done, so _staging_target
        # is no longer needed. Writes now hit the swapped-in production
        # collection through self.collection directly.
        self._staging_target = None
        self.bm25_index.clear()
        self._bm25_initialized = False
        self.query_cache.invalidate()
        # Trigger a fresh build from the swapped-in ChromaDB contents.
        self._ensure_bm25_index()
        # ADR-008 ``on_reindex_complete`` hook — same rationale as the
        # destructive path: FTS5 is derived from Chroma and must be rebuilt.
        self._fts5_reset_and_rebuild()

    def _rebuild_destructive(self) -> Dict[str, Any]:
        """Legacy destructive rebuild — DELETE everything and re-embed.

        Preserved as ``nuclear_rebuild(swap=False)`` for backwards compat
        and for tests that need a byte-identical baseline. Production paths
        default to the zero-downtime swap workflow (v4.8.0+).
        """
        import shutil

        print("[NUCLEAR] Starting destructive rebuild (swap=False)...")
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
        # ADR-008: FTS5 is a derived index; drop and repopulate from the
        # freshly-rebuilt Chroma corpus so the lexical fast-path stays
        # consistent with the vector store.
        self._fts5_reset_and_rebuild()

        elapsed = time.time() - start_time
        stats["elapsed_seconds"] = round(elapsed, 2)
        print(
            f"[NUCLEAR] Destructive rebuild completed in {elapsed:.1f}s "
            f"({stats['indexed']} docs, {stats['chunks_added']} chunks)"
        )

        return stats

    def _rebuild_via_swap(self) -> Dict[str, Any]:
        """Zero-downtime rebuild: staging collection + validate + atomic swap.

        Production collection keeps serving queries until swap completes.
        If any earlier step fails, staging is deleted and production state
        is restored via snapshot so callers see exactly the pre-call state.
        """
        print("[NUCLEAR] Starting zero-downtime rebuild (swap=True)...")
        start_time = time.time()
        prod_name = config.collection_name
        baseline_count = self._read_baseline_count()

        self._cleanup_stale_staging_collections()

        ts = int(time.time())
        _saved = self._snapshot_pre_staging()
        staging = self._create_staging_collection(ts)

        try:
            stats = self._populate_staging(staging)
            self._enforce_staging_validation(staging, baseline_count)
            self._swap_collections_atomic(staging, prod_name, ts)
            self._rebuild_bm25_post_swap(prod_name)
            self._save_metadata()
        except Exception:
            self._rollback_and_cleanup_staging(prod_name, ts, _saved)
            raise

        return self._finalize_swap_stats(stats, start_time)

    def _finalize_swap_stats(self, stats: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Stamp elapsed_seconds and emit the completion banner."""
        elapsed = time.time() - start_time
        stats["elapsed_seconds"] = round(elapsed, 2)
        print(
            f"[NUCLEAR] Zero-downtime rebuild completed in {elapsed:.1f}s "
            f"({stats['indexed']} docs, {stats['chunks_added']} chunks)"
        )
        return stats

    def _read_baseline_count(self) -> int:
        """Best-effort baseline read for the size gate. Zero on read failure."""
        try:
            return self.collection.count()
        except Exception as e:
            print(f"[SWAP] Could not read baseline count (assume 0): {e}")
            return 0

    def _enforce_staging_validation(self, staging, baseline_count: int) -> None:
        """Run gates + log the outcome. Raises RuntimeError on failure."""
        validation = self._validate_staging(staging, baseline_count)
        if not validation["ok"]:
            print(
                f"[SWAP] Validation FAILED — count={validation['count']} "
                f"min={validation['min_expected']} "
                f"canonical_hits={validation['canonical_hits']}/5 "
                f"err={validation['query_error']}"
            )
            raise RuntimeError(f"Staging validation failed: {validation}")
        print(f"[SWAP] Validation OK — count={validation['count']} canonical_hits={validation['canonical_hits']}/5")

    def _rollback_and_cleanup_staging(self, prod_name: str, ts: int, saved) -> None:
        """Restore pre-staging prod state + delete the staging orphan."""
        # GH #161 (v4.8.3): populate may have set _staging_target — clear
        # it before restoring so future writes hit production, not the
        # collection we're about to delete.
        self._staging_target = None
        self._rollback_staging_state(saved)
        # GH #173: populate persisted the staging map to index_metadata.json
        # (index_all -> _finalize_reindex -> _save_metadata) before validation.
        # Restoring only in-memory fields would leave the file describing
        # staging while memory holds production; re-persist so durable metadata
        # matches the restored state and a restart won't load stale staging.
        self._save_metadata()
        try:
            self.chroma_client.delete_collection(f"{prod_name}__staging_{ts}")
        except Exception:
            pass  # cleanup helper will age it out

    def nuclear_rebuild(self, swap: bool = True) -> Dict[str, Any]:
        """Rebuild the collection from scratch.

        Behavior:
            swap=True (default, v4.8.0+):
                Creates a staging collection, populates it, validates the
                result (count within 10% of baseline + 4 of 5 canonical
                sanity queries return hits), then atomically swaps to
                production via two ``Collection.modify(name=)`` calls.
                Queries continue serving from the previous collection
                until the swap completes. Zero downtime.

            swap=False (legacy):
                Deletes the production collection first, then rebuilds.
                Queries return empty results during the rebuild window
                (minutes to hours depending on corpus size + hardware).
                Preserved for backwards-compat and forced-cleanup cases.
        """
        if swap:
            return self._rebuild_via_swap()
        return self._rebuild_destructive()

    # =========================================================================
    # Search
    # =========================================================================

    # =========================================================================
    # FTS5 lexical fast-path helpers (Task 03; ADR-002/003/006).
    # =========================================================================

    def _initialize_fts5_dispatch(self) -> None:
        """Instantiate ``Fts5LexicalIndex`` + ``QueryRouter`` under the toggle.

        Kept separate from ``__init__`` so failures surface with a clear source
        (config typo vs Chroma issue vs FTS5 issue) and callers can retry after
        fixing config without rebuilding the whole orchestrator.

        Task 05: after instantiation, inspect the marker file. When status is
        anything other than ``complete``, dispatch the lazy background
        migration so first-time users (fresh install / zero-touch upgrade)
        get a populated index without editing config or running a script.
        Interrupted migrations (``in_progress`` with a partial
        ``docs_indexed`` count) resume from the last checkpointed batch.
        """
        db_path = config.data_dir / "fts5_index.db"
        state_path = config.data_dir / "fts5_migration.state"
        self.fts5_index = Fts5LexicalIndex(db_path=db_path, state_path=state_path)
        # QueryRouter raises re.error at construction if any pattern is
        # malformed — treat that as a fatal startup error so the operator
        # sees the broken pattern immediately instead of on the first query.
        self.query_router = QueryRouter(config.fts5_patterns)
        self._maybe_start_fts5_migration()

    def _maybe_start_fts5_migration(self) -> None:
        """Dispatch the lazy FTS5 rebuild when the marker isn't ``complete``.

        v4.8.3 sanity check: even when the marker says ``complete``, cross-
        check FTS5 row count against Chroma. A stale marker with an empty
        FTS5 index (post-swap orphan cleanup, manual truncate, disk
        corruption) previously left the fast-path permanently silent —
        queries returned no_results forever without any error surfaced.
        """
        if self.fts5_index is None:
            return
        state_payload = self.fts5_index.state.read() or {}
        status = state_payload.get("status")
        if status == "complete":
            if not self._fts5_marker_matches_reality():
                print("[FTS5] stale complete marker detected — forcing rebuild")
                # Fall through to dispatch a fresh migration below.
            else:
                return
        resume_from = int(state_payload.get("docs_indexed", 0)) if status == "in_progress" else 0
        try:
            docs_total = int(self.collection.count())
        except Exception as exc:  # noqa: BLE001 — Chroma error must not kill startup
            print(f"[FTS5] migration skipped — cannot count corpus: {exc}")
            return
        if docs_total <= 0:
            # Empty corpus: nothing to rebuild. Mark as complete so the
            # fast-path becomes ready immediately for future writes.
            self.fts5_index._write_state(  # noqa: SLF001 — trusted internal
                "complete",
                0,
                0,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                None,
            )
            with self.fts5_index._fts5_lock:  # noqa: SLF001
                self.fts5_index._ready = True  # noqa: SLF001
            return
        print(f"[FTS5] migration starting (resume_from={resume_from}, docs_total={docs_total})")
        get_metrics().set_gauge(FAST_PATH_MIGRATION_DOCS_TOTAL, float(docs_total))
        get_metrics().set_gauge(FAST_PATH_MIGRATION_DOCS_INDEXED, float(resume_from))
        self.fts5_index.start_migration_background(
            self._iter_chroma_chunks_for_fts5,
            docs_total,
            resume_from=resume_from,
            on_progress=self._fts5_migration_progress,
        )

    def _fts5_marker_matches_reality(self) -> bool:
        """Return True when the FTS5 marker is credible vs actual state.

        Rejects markers claiming ``complete`` while the FTS5 index holds far
        fewer rows than Chroma. Threshold is 10% because a small drift is
        normal (chunk-level dedup, deletes), but a >90% deficit means the
        marker is lying about a rebuild that didn't actually populate.
        """
        try:
            fts5_count = self.fts5_index.count()
            chroma_count = self.collection.count()
        except Exception:
            return True  # can't check → trust the marker (fail-safe)
        if chroma_count == 0:
            return True  # empty corpus, marker complete is legitimate
        return fts5_count >= chroma_count * 0.1

    @staticmethod
    def _fts5_migration_progress(docs_indexed: int, docs_total: int) -> None:
        """Push the migration checkpoint into Prometheus gauges."""
        metrics = get_metrics()
        metrics.set_gauge(FAST_PATH_MIGRATION_DOCS_INDEXED, float(docs_indexed))
        metrics.set_gauge(FAST_PATH_MIGRATION_DOCS_TOTAL, float(docs_total))

    def _iter_chroma_chunks_for_fts5(self) -> Iterable[Tuple[str, str, str, str]]:
        """Yield ``(chunk_id, content, filename, category)`` rows in stable order.

        Batches via offset+limit to avoid the SQLite ``too many SQL variables``
        error (chromadb 1.x rebuilds an ``IN (?, ?, ...)`` clause for every
        returned row; a single ``limit=48184`` call blows past the 999 default
        max_variable_number and the whole migration fails). Deterministic order
        is preserved by sorting each batch by ``chunk_id`` — good enough for
        resume-from-index semantics since chunk_ids are UUIDs.
        """
        try:
            count = self.collection.count()
        except Exception as exc:  # noqa: BLE001
            print(f"[FTS5] chunk iterator aborted — Chroma count failed: {exc}")
            return
        if count == 0:
            return
        batch_size = 500  # SQLite default max_variable_number is 999
        offset = 0
        while offset < count:
            try:
                fetched = self.collection.get(
                    include=["documents", "metadatas"],
                    limit=batch_size,
                    offset=offset,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[FTS5] chunk batch failed at offset={offset}: {exc}")
                return
            ids = fetched.get("ids") or []
            docs = fetched.get("documents") or []
            metas = fetched.get("metadatas") or []
            if not ids:
                break
            order = sorted(range(len(ids)), key=lambda i: ids[i])
            for i in order:
                meta = metas[i] or {}
                yield (
                    str(ids[i]),
                    str(docs[i] or ""),
                    str(meta.get("filename", "") or ""),
                    str(meta.get("category", "") or ""),
                )
            offset += len(ids)

    def _fts5_sync_add(self, ids: Sequence[str], docs: Sequence[str], metas: Sequence[Dict[str, Any]]) -> None:
        """Best-effort CRUD sync hook after a successful ChromaDB write.

        Errors are caught + logged + tallied on the FTS5 error counter
        (``error_class="Fts5CrudSyncError"``) but NEVER raised — upstream
        CRUD must keep succeeding even when the FTS5 secondary index is
        temporarily unwritable (disk full, permission drift). See ADR-008.
        """
        if not (config.fts5_enabled and self.fts5_index is not None):
            return
        for chunk_id, content, meta in zip(ids, docs, metas):
            try:
                self.fts5_index.add_document(
                    str(chunk_id),
                    str(content or ""),
                    str((meta or {}).get("filename", "")),
                    str((meta or {}).get("category", "")),
                )
            except Exception as exc:  # noqa: BLE001 — see docstring; NEVER raise
                get_metrics().inc(FAST_PATH_ERRORS_TOTAL, '{error_class="Fts5CrudSyncError"}')
                print(f"[FTS5] add sync failed for chunk_id={chunk_id}: {exc}")

    def _fts5_sync_remove_by_doc_id(self, doc_id: str) -> None:
        """Remove every chunk whose id begins with ``<doc_id>_`` (see ``_dedup_chunks``)."""
        if not (config.fts5_enabled and self.fts5_index is not None):
            return
        try:
            results = self.collection.get(where={"doc_id": doc_id}, include=[])
        except Exception as exc:  # noqa: BLE001
            get_metrics().inc(FAST_PATH_ERRORS_TOTAL, '{error_class="Fts5CrudSyncError"}')
            print(f"[FTS5] remove sync fetch failed for doc_id={doc_id}: {exc}")
            return
        for chunk_id in results.get("ids") or []:
            try:
                self.fts5_index.remove_document(str(chunk_id))
            except Exception as exc:  # noqa: BLE001
                get_metrics().inc(FAST_PATH_ERRORS_TOTAL, '{error_class="Fts5CrudSyncError"}')
                print(f"[FTS5] remove sync failed for chunk_id={chunk_id}: {exc}")

    def _fts5_reset_and_rebuild(self) -> None:
        """Drop the FTS5 database + marker file then start a fresh migration.

        Called from ``nuclear_rebuild`` and swap-based rebuilds — the corpus
        was recreated from scratch so the derived FTS5 index MUST be dropped
        and repopulated from the swapped-in ChromaDB contents.
        """
        if not (config.fts5_enabled and self.fts5_index is not None):
            return
        try:
            self.fts5_index.close()
        except Exception:  # noqa: BLE001 — best-effort close
            pass
        db_path = config.data_dir / "fts5_index.db"
        state_path = config.data_dir / "fts5_migration.state"
        for path in (db_path, state_path, db_path.with_suffix(".db-wal"), db_path.with_suffix(".db-shm")):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                print(f"[FTS5] could not remove {path}: {exc}")
        self.fts5_index = Fts5LexicalIndex(db_path=db_path, state_path=state_path)
        self._maybe_start_fts5_migration()

    def _maybe_dispatch_fts5(
        self,
        query_text: str,
        max_results: int,
        category_filter: Optional[str],
        search_method: str,
    ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        """Decide whether to serve the query from the FTS5 fast-path.

        Returns ``(result, path)``:
        - ``(list, "fts5")`` — fast-path succeeded, caller returns ``result``.
        - ``(None, "hybrid")`` — router said semantic OR explicit ``"hybrid"``
          override; caller runs the existing hybrid pipeline.
        - ``(None, "fallback")`` — FTS5 attempted but failed (not ready /
          low_hits / error); caller runs hybrid as recovery. Fallback and
          error counters are already incremented here.
        Raises ``Fts5NotReadyError`` when ``search_method="fts5"`` is explicit
        and the index is not ready — surfaces to the MCP wrapper.
        """
        metrics = get_metrics()
        if search_method == "fts5":
            if self.fts5_index is None or not self.fts5_index.is_ready():
                raise Fts5NotReadyError(
                    "FTS5 index is not ready (migration in progress). "
                    "Suggestion: use search_method='auto' to fallback gracefully."
                )
            return self._run_fts5_search(query_text, max_results, category_filter, skip_min_hits=True), "fts5"
        if search_method == "hybrid":
            return None, "hybrid"
        if self.query_router is None or self.query_router.classify(query_text) != "lexical":
            return None, "hybrid"
        if self.fts5_index is None or not self.fts5_index.is_ready():
            metrics.inc(FAST_PATH_FALLBACK_TOTAL, '{reason="disabled"}')
            return None, "fallback"
        try:
            result = self._run_fts5_search(query_text, max_results, category_filter, skip_min_hits=False)
        except Fts5NotReadyError:
            raise
        except Exception as exc:  # noqa: BLE001 — every FTS5 failure must fall back
            metrics.inc(FAST_PATH_ERRORS_TOTAL, f'{{error_class="{exc.__class__.__name__}"}}')
            metrics.inc(FAST_PATH_FALLBACK_TOTAL, '{reason="error"}')
            return None, "fallback"
        # Empty list means category_filter or hydration reduced effective hits
        # to zero — semantically same as low_hits, must trigger fallback.
        if not result:
            metrics.inc(FAST_PATH_FALLBACK_TOTAL, '{reason="low_hits"}')
            return None, "fallback"
        return result, "fts5"

    def _run_fts5_search(
        self,
        query_text: str,
        max_results: int,
        category_filter: Optional[str],
        *,
        skip_min_hits: bool,
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute the FTS5 search + result formatting, tracking latency.

        Returns ``None`` when the hit count is below ``config.fts5_min_hits``
        (only when ``skip_min_hits=False`` — explicit ``search_method="fts5"``
        bypasses the threshold so debug/testing paths always see raw output).
        Rerank defaults to OFF on the fast-path (ADR-003 — cross-encoder cost
        ≈30-70ms per 100 docs obliterates the <10ms lexical budget). Enable
        with ``config.fts5_rerank_enabled=True`` when recall matters more
        than latency; the ``FAST_PATH_RERANK_SKIPPED_TOTAL`` counter only
        moves when rerank was skipped.
        """
        assert self.fts5_index is not None  # narrowed by caller
        metrics = get_metrics()
        candidates = max(max_results * 3, 20)
        start = time.monotonic()
        try:
            hits = self.fts5_index.search(query_text, top_k=candidates)
        finally:
            metrics.observe(FAST_PATH_LATENCY_SECONDS, time.monotonic() - start)
        if not skip_min_hits and len(hits) < config.fts5_min_hits:
            return None
        formatted = self._format_fts5_results(hits, max_results, category_filter)
        if config.fts5_rerank_enabled and formatted:
            formatted = self._rerank_fts5_results(query_text, formatted, max_results)
        else:
            metrics.inc(FAST_PATH_RERANK_SKIPPED_TOTAL)
        formatted = self._expand_with_adjacent_chunks(formatted)
        # If category_filter or hydration wiped all hits, treat as no useful
        # fast-path result — return None so caller records low_hits fallback
        # instead of incrementing a misleading FAST_PATH_HITS_TOTAL{path=fts5}.
        if not skip_min_hits and not formatted:
            return None
        metrics.inc(FAST_PATH_HITS_TOTAL, '{path="fts5"}')
        return formatted

    def _rerank_fts5_results(
        self,
        query_text: str,
        formatted: List[Dict[str, Any]],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Apply cross-encoder rerank to fast-path results, preserving schema.

        The reranker reads ``doc["document"]`` and writes ``doc["reranker_score"]``.
        Fast-path items use ``"content"`` as the text field, so alias it in and
        pop it out. ``search_method`` stays ``"fts5"`` — rerank is a layered
        rescorer, not a path change.
        """
        for item in formatted:
            item["document"] = item.get("content", "")
        reranked = self.reranker.rerank(query_text, formatted, top_k=max_results)
        for item in reranked:
            item.pop("document", None)
        return reranked

    def _format_fts5_results(
        self,
        hits: List[Tuple[str, float]],
        max_results: int,
        category_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Hydrate FTS5 ``(chunk_id, score)`` hits into full result dicts."""
        if not hits:
            return []
        chunk_ids = [chunk_id for chunk_id, _ in hits]
        try:
            fetched = self.collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        except Exception as exc:  # noqa: BLE001 — degrade gracefully; caller falls back
            print(f"[WARN] FTS5 metadata fetch failed: {exc}")
            return []
        documents_by_id = dict(zip(fetched.get("ids", []), fetched.get("documents", [])))
        metadata_by_id = dict(zip(fetched.get("ids", []), fetched.get("metadatas", [])))
        formatted: List[Dict[str, Any]] = []
        raw_scores = [float(score) for _, score in hits if chunk_ids]
        max_score = max(raw_scores) if raw_scores else 1.0
        min_score = min(raw_scores) if raw_scores else 0.0
        score_range = max_score - min_score
        for chunk_id, raw_score in hits:
            # v4.8.3: skip orphan hits where FTS5 has the chunk_id but Chroma
            # doesn't (nuclear rebuild residue, manual delete race). Previous
            # behavior emitted empty result items with source="" that broke
            # downstream reranking + polluted result lists.
            document = documents_by_id.get(chunk_id)
            if not document:
                continue
            metadata = metadata_by_id.get(chunk_id) or {}
            if category_filter and metadata.get("category") != category_filter:
                continue
            normalized = (float(raw_score) - min_score) / score_range if score_range > 0 else 1.0
            formatted.append(
                {
                    "content": document,
                    "source": metadata.get("source", ""),
                    "filename": metadata.get("filename", ""),
                    "category": metadata.get("category", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "score": round(normalized, 4),
                    "raw_rrf_score": None,
                    "reranker_score": None,
                    "semantic_rank": None,
                    "bm25_rank": None,
                    "search_method": "fts5",
                    "keywords": (metadata.get("keywords") or "").split(","),
                    "routed_by": "fts5_router",
                }
            )
            if len(formatted) >= max_results:
                break
        return formatted

    # =========================================================================
    # Query — hybrid pipeline (with optional FTS5 fast-path dispatch on top).
    # =========================================================================

    def query(
        self,
        query_text: str,
        max_results: int = None,
        category_filter: Optional[str] = None,
        hybrid_alpha: float = 0.5,
        search_method: str = "auto",
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with RRF fusion + cross-encoder reranking.

        Pipeline: Semantic + BM25 -> RRF fusion -> Reranker -> Results.

        ``search_method`` (v4.8.2+) selects the dispatch:
        - ``"auto"`` (default): router classifies the query. Lexical goes to
          the FTS5 fast-path when ``config.fts5_enabled`` and the index is
          ready; semantic goes to the hybrid pipeline.
        - ``"hybrid"``: skip the router; force the hybrid pipeline (kill
          switch for suspected router misclassification).
        - ``"fts5"``: skip the router; force the FTS5 fast-path. Raises
          ``Fts5NotReadyError`` when the feature is disabled or the index is
          not ready — the MCP wrapper surfaces the error to the caller.
        """
        max_results = max_results or config.default_results

        # Cache lookup (5-tuple key includes search_method — different paths
        # produce different result sets, they MUST NOT share cache entries).
        cached = self.query_cache.get(query_text, max_results, category_filter, hybrid_alpha, search_method)
        if cached is not None:
            return cached

        # FTS5 dispatch (feature-gated). When the toggle is off, this whole
        # block short-circuits — zero cost on the hybrid path.
        hybrid_path_label = "hybrid"
        if config.fts5_enabled:
            fast_path_result, hybrid_path_label = self._maybe_dispatch_fts5(
                query_text, max_results, category_filter, search_method
            )
            if fast_path_result is not None:
                self.query_cache.put(
                    query_text,
                    max_results,
                    category_filter,
                    hybrid_alpha,
                    fast_path_result,
                    search_method=search_method,
                )
                return fast_path_result
        elif search_method == "fts5":
            raise Fts5NotReadyError("FTS5 fast-path is disabled in config; set search.lexical_fast_path.enabled=true")

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

            if not _matches_category(data.get("metadata", {})):
                continue

            combined_scores[chunk_id] = {
                "rrf_score": combined_rrf + _metadata_path_score(query_text, data.get("metadata", {})),
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

        self.query_cache.put(
            query_text,
            max_results,
            category_filter,
            hybrid_alpha,
            formatted,
            search_method=search_method,
        )
        if config.fts5_enabled:
            get_metrics().inc(FAST_PATH_HITS_TOTAL, f'{{path="{hybrid_path_label}"}}')
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
        """Get full document content by filepath.

        Rejects paths that resolve outside ``config.documents_dir`` to keep
        the endpoint from becoming an arbitrary-file-read primitive.
        Returns ``None`` on rejection so the failure mode matches the
        existing "not found" case exposed to MCP clients.
        """
        try:
            resolved = validate_path_within(config.documents_dir, filepath)
        except PathEscapeError as exc:
            print(f"[SECURITY] get_document refused escaping path {filepath!r}: {exc}")
            return None
        try:
            doc = self.parser.parse_file(resolved)
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
            print(f"[ERROR] Failed to read document {resolved}: {e}")
        return None

    def add_document_from_content(
        self,
        content: str,
        filepath: str,
        category: str,
        external_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a new document from raw content string. Saves to disk and indexes.

        When ``external_source`` is provided the content is treated as
        attacker-influenced: sanitized against known prompt-injection
        sentinels and wrapped in a provenance fence before ever touching
        disk (see ``mcp_server.security.sanitize_external_content``).

        The destination ``filepath`` is resolved inside
        ``config.documents_dir``; escapes via ``..`` or absolute paths are
        rejected with a structured error and never write to disk.
        """
        try:
            full_path = validate_path_within(config.documents_dir, filepath)
        except PathEscapeError as exc:
            return {"error": f"Filepath rejected: {exc}"}

        if external_source:
            content = sanitize_external_content(content, external_source)

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
            "embedding_dim": config.embedding_dim,
        }
        self._source_to_docid[str(full_path.resolve())] = doc.id
        self._save_metadata()
        self.query_cache.invalidate()
        self.bm25_index.build_index()
        self._fts5_sync_add_from_doc(doc)

        return {
            "chunks_added": chunks_added,
            "dedup_skipped": dedup_skipped,
            "category": category,
            "filepath": str(full_path),
        }

    def _fts5_sync_add_from_doc(self, doc: Document) -> None:
        """Insert every chunk of ``doc`` into FTS5. Best-effort (ADR-008)."""
        if not (config.fts5_enabled and self.fts5_index is not None and doc.chunks):
            return
        ids = [f"{doc.id}_{chunk.index}" for chunk in doc.chunks]
        docs_content = [chunk.content for chunk in doc.chunks]
        metas = [{"filename": doc.filename, "category": doc.category} for _ in doc.chunks]
        self._fts5_sync_add(ids, docs_content, metas)

    def update_document_content(self, filepath: str, content: str) -> Dict[str, Any]:
        """Update an existing document. Removes old chunks and re-indexes.

        Rejects paths that resolve outside ``config.documents_dir`` so
        this endpoint cannot be used to overwrite arbitrary host files.
        """
        try:
            filepath = validate_path_within(config.documents_dir, filepath)
        except PathEscapeError as exc:
            return {"error": f"Filepath rejected: {exc}"}

        if not filepath.exists():
            return {"error": f"File not found: {filepath}"}

        # Resolve to absolute for consistent comparison with stored metadata
        filepath_resolved = str(filepath.resolve())

        doc_id = self._source_to_docid.get(filepath_resolved)

        old_chunks_removed = 0
        if doc_id:
            self._fts5_sync_remove_by_doc_id(doc_id)
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
            "embedding_dim": config.embedding_dim,
        }
        self._source_to_docid[str(filepath.resolve())] = doc.id
        self._save_metadata()
        self.query_cache.invalidate()
        self.bm25_index.build_index()
        self._fts5_sync_add_from_doc(doc)

        return {
            "old_chunks_removed": old_chunks_removed,
            "new_chunks_added": new_chunks_added,
            "dedup_skipped": dedup_skipped,
            "filepath": str(filepath),
        }

    def remove_document_by_path(self, filepath: str, delete_file: bool = False) -> Dict[str, Any]:
        """Remove a document from the index. Optionally delete from disk.

        Guarded by ``validate_path_within(documents_dir, filepath)`` so a
        client cannot use ``delete_file=True`` as an arbitrary-file-delete
        primitive against paths outside the corpus.
        """
        try:
            resolved_path = validate_path_within(config.documents_dir, filepath)
        except PathEscapeError as exc:
            return {"error": f"Filepath rejected: {exc}"}

        filepath_resolved = str(resolved_path)

        doc_id = self._source_to_docid.get(filepath_resolved)

        if not doc_id:
            return {"error": f"Document not found in index: {filepath}"}

        self._fts5_sync_remove_by_doc_id(doc_id)
        chunks_removed = self._remove_document_chunks(doc_id)
        self._source_to_docid.pop(filepath_resolved, None)
        del self._indexed_docs[doc_id]

        if delete_file:
            try:
                resolved_path.unlink(missing_ok=True)
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

        # The body was fetched from an untrusted URL; the sanitizer inserts a
        # provenance fence and defuses model control tokens before the text
        # ever reaches the parser or the LLM that will consume the RAG output.
        return self.add_document_from_content(clean_text, filepath, category, external_source=url)

    def search_similar(self, filepath: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document using embedding similarity.

        Paths that resolve outside ``config.documents_dir`` return an empty
        list rather than probing index state — an attacker must not be able
        to use this endpoint to test for the existence of arbitrary files.
        """
        if not is_path_within(config.documents_dir, filepath):
            return []

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
        """Get background reindex progress without computing full index stats.

        Active runs return doc-level counters + v4.8.0 Fase 4 granular fields
        (chunks_processed, chunks_total, throughput_cps, eta_seconds,
        checkpoint_saved_at, resumed). Idle returns ``active=False`` plus
        ``last_result`` or ``last_error`` from the most recent completed run.
        """
        progress = self._reindex_progress
        if progress.get("active"):
            return self._active_reindex_status(progress)
        return self._idle_reindex_status(progress)

    @staticmethod
    def _active_reindex_status(progress: Dict[str, Any]) -> Dict[str, Any]:
        """Payload shape for an in-flight background reindex."""
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
            # v4.8.0 Fase 4: granular progress + resume checkpoint fields
            "chunks_processed": progress.get("chunks_processed", 0),
            "chunks_total": progress.get("chunks_total", 0),
            "throughput_cps": progress.get("throughput_cps", 0.0),
            "eta_seconds": progress.get("eta_seconds", 0),
            "checkpoint_saved_at": progress.get("checkpoint_saved_at"),
            "resumed": progress.get("resumed", False),
        }

    @staticmethod
    def _idle_reindex_status(progress: Dict[str, Any]) -> Dict[str, Any]:
        """Payload shape for idle state — surfaces last_result or last_error if present."""
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

    # =========================================================================
    # v4.8.0 Fase 4: reindex checkpoint (resume support)
    # =========================================================================

    CHECKPOINT_VERSION = 1

    def _compute_config_signature(self) -> str:
        """SHA256 of embedding + chunking config.

        Any drift invalidates the checkpoint — resuming with a different
        embedding model or chunk size would produce a mixed collection
        (partial old + partial new), which is worse than a full restart.
        """
        payload = f"{config.embedding_model}|{config.embedding_dim}|{config.chunk_size}|{config.chunk_overlap}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _write_checkpoint(
        self,
        operation: str,
        indexed_doc_ids: List[str],
        chunks_processed: int,
        started_at: Optional[str] = None,
    ) -> None:
        """Atomically persist reindex progress to reindex_checkpoint.json.

        Writes to a sibling `.tmp` then os.replace() — guarantees the
        consumer sees either the old file or the new one, never a
        partially-written file (Windows-safe: os.replace is atomic on
        NTFS in-directory).
        """
        self._checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.CHECKPOINT_VERSION,
            "started_at": started_at or datetime.now().isoformat(),
            "checkpoint_at": datetime.now().isoformat(),
            "operation": operation,
            "indexed_doc_ids": list(indexed_doc_ids),
            "chunks_processed": chunks_processed,
            "config_signature": self._compute_config_signature(),
        }
        tmp_path = self._checkpoint_file.with_suffix(self._checkpoint_file.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, self._checkpoint_file)
        # Reflect in progress dict for status polling.
        self._reindex_progress["checkpoint_saved_at"] = payload["checkpoint_at"]

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load and validate a checkpoint from disk.

        Returns None (with a WARN) when the file is missing, corrupt,
        from a future schema version, or when the config signature
        differs from the current config — any of those means the
        checkpoint cannot be trusted and the caller should restart from
        scratch.
        """
        if not self._checkpoint_file.exists():
            return None

        try:
            data = json.loads(self._checkpoint_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Corrupt checkpoint file, ignoring: {e}")
            return None

        if not isinstance(data, dict):
            print("[WARN] Checkpoint payload is not a dict, ignoring")
            return None

        version = data.get("version")
        if version != self.CHECKPOINT_VERSION:
            print(f"[WARN] Checkpoint version {version} != current {self.CHECKPOINT_VERSION}, ignoring")
            return None

        stored_sig = data.get("config_signature")
        current_sig = self._compute_config_signature()
        if stored_sig != current_sig:
            print("[WARN] Checkpoint config_signature mismatch (embedding model or chunking changed) — starting fresh")
            return None

        return data

    def _clear_checkpoint(self) -> None:
        """Remove the checkpoint file (call after successful reindex)."""
        try:
            if self._checkpoint_file.exists():
                self._checkpoint_file.unlink()
        except OSError as e:
            print(f"[WARN] Failed to remove checkpoint file: {e}")

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

mcp = MCPServer(
    "knowledge-rag",
    version="4.6.0",
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
    search_method: str = "auto",
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
        search_method: Dispatch selector (v4.8.2+). One of ``"auto"`` (router picks FTS5 fast-path
            for lexical queries when enabled, hybrid otherwise), ``"hybrid"`` (force hybrid path —
            kill switch for suspected router misclassification), or ``"fts5"`` (force FTS5 fast-path
            — debug/testing; errors out when the feature is disabled or the index is not ready).
            Default ``"auto"`` preserves pre-v4.8.2 behavior byte-for-byte when the fast-path is
            disabled in config.

    Returns:
        JSON string with results including content chunks, source filepath, relevance score, and
        search method used. Returns chunks, not full document content.

    Usage: Primary search tool — use for any topic or keyword lookup. Prefer search_similar() when
    you already have a reference document and want more like it. Prefer get_document() when you
    already know the exact filepath and need the full content.
    """
    if not query or not query.strip():
        return json.dumps({"status": "error", "message": "Query cannot be empty"})

    if search_method not in ("auto", "hybrid", "fts5"):
        return json.dumps(
            {
                "status": "error",
                "message": f"Invalid search_method '{search_method}'. Valid: auto, hybrid, fts5",
            }
        )

    max_results = max(1, min(max_results or 5, config.max_results))
    hybrid_alpha = max(0.0, min(hybrid_alpha if hybrid_alpha is not None else 0.3, 1.0))
    min_score = max(0.0, min(min_score if min_score is not None else 0.0, 1.0))

    valid_categories = list(config.keyword_routes.keys()) + list(set(config.category_mappings.values())) + ["general"]
    if category and category not in valid_categories:
        return json.dumps(
            {"status": "error", "message": f"Invalid category '{category}'. Valid: {', '.join(valid_categories)}"}
        )

    orchestrator = get_orchestrator()
    try:
        results = orchestrator.query(
            query.strip(),
            max_results=max_results,
            category_filter=category,
            hybrid_alpha=hybrid_alpha,
            search_method=search_method,
        )
    except Fts5NotReadyError as exc:
        # Surface the fast-path error verbatim + always add the auto-fallback
        # suggestion so debug users can recover without hunting docs.
        return json.dumps(
            {
                "status": "error",
                "error": str(exc),
                "suggestion": "search_method='auto' fallback gracefully",
            }
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


def _reindex_error_response(message: str) -> str:
    """JSON error envelope for reindex_documents pre-flight validation."""
    return json.dumps(
        {"status": "error", "error": message},
        indent=2,
        ensure_ascii=False,
    )


def _resolve_reindex_mode(force: bool, full_rebuild: bool, resume: bool) -> str:
    """Pick the reindex mode. Resume forces smart_reindex regardless of force flag.

    Rationale: an interrupted smart run must be resumed with smart — an
    incremental pass would ignore the checkpoint entirely.
    """
    if resume:
        return "smart_reindex"
    if full_rebuild:
        return "nuclear_rebuild"
    if force:
        return "smart_reindex"
    return "incremental"


def _load_reindex_resume_state(orchestrator) -> Optional[Dict[str, Any]]:
    """v4.8.0 Fase 4 — hydrate resume_state from the on-disk checkpoint.

    Returns None (silent fresh reindex) if no valid checkpoint exists —
    corrupt/missing checkpoints must not block a legitimate run.
    """
    cp = orchestrator._load_checkpoint()
    if cp is None:
        print("[INFO] resume=True but no valid checkpoint — starting fresh smart reindex")
        return None
    return {
        "doc_ids": cp.get("indexed_doc_ids", []),
        "chunks_processed": cp.get("chunks_processed", 0),
    }


def _format_reindex_response(mode: str, result: Dict[str, Any]) -> str:
    """Serialize the orchestrator result — already-running vs started envelopes."""
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
@instrument("reindex_documents")
def reindex_documents(
    force: bool = False,
    full_rebuild: bool = False,
    resume: bool = False,
) -> str:
    """Index or reindex all documents in the knowledge base (runs in background).

    ``force`` — smart reindex (detect changed files + rebuild BM25). Use after
    filesystem edits outside add_document/update_document.
    ``full_rebuild`` — nuclear rebuild (delete + re-embed). Use only after
    embedding-model change or index corruption. Mutually exclusive with resume.
    ``resume`` — pick up an interrupted smart reindex from
    ``data/reindex_checkpoint.json``. Falls back to a fresh smart run silently
    if the checkpoint is missing/corrupt/drifted (v4.8.0 Fase 4).

    Returns a JSON envelope. Poll ``get_reindex_status()`` until
    ``reindex.active`` becomes false. Add/update/URL tools already auto-index —
    use these flags only for the recovery/rebuild scenarios above.
    """
    orchestrator = get_orchestrator()

    if resume and full_rebuild:
        return _reindex_error_response("resume=True is only valid for smart reindex; use full_rebuild=False")

    mode = _resolve_reindex_mode(force, full_rebuild, resume)
    resume_state = _load_reindex_resume_state(orchestrator) if resume else None
    result = orchestrator.start_reindex_background(mode, resume_state=resume_state)
    return _format_reindex_response(mode, result)


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


#: Transports for which ``mcp.run()`` speaks HTTP. When the operator has
#: configured a bearer token these need the auth middleware in front, which
#: means we cannot rely on ``mcp.run()`` alone — the middleware wrapper is
#: installed via uvicorn.
_HTTP_TRANSPORTS: Tuple[str, ...] = ("sse", "streamable-http")


def _http_app_factory(transport: str):
    """Return the FastMCP ASGI factory matching ``transport``."""
    if transport == "streamable-http":
        return mcp.streamable_http_app
    if transport == "sse":
        return mcp.sse_app
    raise ValueError(f"Unknown HTTP transport: {transport!r}")


def _run_transport(transport: str) -> None:
    """Boot the requested MCP transport, applying auth when configured.

    * ``stdio`` — the pipe carries no HTTP metadata; auth is not applicable
      and the middleware is not installed.
    * HTTP transports (``sse``, ``streamable-http``) — when
      ``config.auth_bearer_token`` is set, the FastMCP ASGI app is wrapped
      in :class:`BearerAuthMiddleware` and served through uvicorn so no
      unauthenticated request can reach the MCP dispatcher. When the token
      is unset we print a one-line warning and fall back to ``mcp.run``
      so the current, unauth open-port behaviour is preserved for
      backwards compatibility.
    * Anything else is refused loudly rather than silently starting an
      unguarded server.
    """
    if transport == "stdio":
        mcp.run(transport=transport)
        return

    if transport not in _HTTP_TRANSPORTS:
        raise ValueError(f"Unknown transport: {transport!r}")

    token = getattr(config, "auth_bearer_token", "") or ""

    import uvicorn

    from .health import HealthMiddleware

    app_factory = _http_app_factory(transport)
    # Build the app with the configured host. MCP 2.0.0's factories default to
    # host="127.0.0.1", which turns on a localhost-only DNS-rebinding allowlist;
    # without this, a non-loopback server_host is served by uvicorn but the app
    # rejects the configured host with 421 before auth runs (GH #174).
    app = app_factory(host=config.server_host)

    if not token:
        print(
            f"[WARN] Bearer auth disabled on {transport} transport — "
            "set server.auth.bearer_token in config.yaml to require credentials.",
            file=sys.stderr,
        )
        served = HealthMiddleware(app, get_orchestrator)
    else:
        guarded = BearerAuthMiddleware(app, token)
        # HealthMiddleware wraps the bearer app so /health probes bypass auth
        # (matches BearerAuthMiddleware.exempt_paths). Order matters: health
        # first, then bearer — a probe never reaches the auth layer.
        served = HealthMiddleware(guarded, get_orchestrator)
        print(
            f"[SECURITY] Bearer auth enforced on {transport} transport ({config.server_host}:{config.server_port})",
            file=sys.stderr,
        )

    print(
        f"[HEALTH] Probe endpoint at http://{config.server_host}:{config.server_port}/health",
        file=sys.stderr,
    )
    uvicorn.run(served, host=config.server_host, port=config.server_port)


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
    from .logging_config import setup_logging
    from .preflight import run_preflight

    # Opt-in JSON logging. Default ("text") is a no-op — existing print()
    # sites remain the source of stderr output. When operators set
    # server.logging.format: "json" in config.yaml, structured records
    # emitted via ``get_logger(__name__).info(...)`` land on stderr as
    # single-line JSON ready for ELK/Loki/Datadog ingestion.
    setup_logging(fmt=config.log_format, level=config.log_level)

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

            _run_transport(transport)
    except AlreadyRunningError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(ALREADY_RUNNING_EXIT_CODE) from e


if __name__ == "__main__":
    main()
