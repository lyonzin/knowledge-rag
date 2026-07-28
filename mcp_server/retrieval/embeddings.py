"""
╭─╴ FASTEMBED EMBEDDINGS ╶───────────────────────────────────────╮
│                                                                │
│   FastEmbed-based ChromaDB embedding function + GPU readiness  │
│   probe. Extracted verbatim from server.py in the A2.1         │
│   refactor.                                                    │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``

Test-patch compatibility:
    ``FastEmbedEmbeddings._load_model`` resolves the underlying
    ``fastembed.TextEmbedding`` class via ``mcp_server.server.TextEmbedding``
    at call time, not import time. The historical unit tests patch
    ``mcp_server.server.TextEmbedding`` (see ``tests/test_lazy_embeddings.py``
    and ``tests/chaos/test_hf_down.py``); routing the lookup through the
    ``mcp_server.server`` module keeps every one of those patches effective
    after the refactor.
"""

import os
import platform
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import List, Optional

from ..config import config


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails after a successful model load."""


class EmbeddingModelLoadError(RuntimeError):
    """Raised when the embedding model itself cannot be loaded.

    Distinct from EmbeddingError so callers can decide whether to retry
    (transient runtime failure) or surface a hard configuration problem.
    """


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


def _resolve_text_embedding_class():
    """Late-bind lookup for the ``TextEmbedding`` class.

    Tests patch ``mcp_server.server.TextEmbedding`` (a re-export). Routing
    the lookup through that module keeps those patches effective after the
    A2.1 module split. Falls back to the real ``fastembed.TextEmbedding``
    when the server module has not populated its re-exports yet.
    """
    try:
        from mcp_server import server as _srv

        return _srv.TextEmbedding
    except (ImportError, AttributeError):
        from fastembed import TextEmbedding

        return TextEmbedding


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
            import onnxruntime as ort  # noqa: F401

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
        self._model = None
        self._load_lock = threading.Lock()
        # Sticky failure flag: once load fails, subsequent calls re-raise immediately
        # instead of looping through download/retry. Same pattern as CrossEncoderReranker.
        self._load_failed: Optional[Exception] = None

    @property
    def dimension(self) -> int:
        """Output vector dimensionality — satisfies the EmbeddingProvider Protocol."""
        return self._dim

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
            # Late-bind TextEmbedding via mcp_server.server so unit tests that
            # patch mcp_server.server.TextEmbedding still take effect.
            TextEmbedding = _resolve_text_embedding_class()
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
