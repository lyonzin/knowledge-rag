"""Public offline corpus and measurements shared by the opt-in retrieval audit."""

from __future__ import annotations

import hashlib
import shutil
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "README.md",
    "docs/INSTALLATION.md",
    "docs/gpu-setup.md",
    "docs/CONFIGURATION.md",
    "docs/runbooks/fts5_migration.md",
    "docs/reindex-operations.md",
)
CASES = (
    ("en", "How do I install knowledge-rag on Windows with PowerShell?", "docs/INSTALLATION.md"),
    ("en", "How does CPU fallback work when the CUDA provider cannot initialize?", "docs/gpu-setup.md"),
    ("en", "How can I override the embedding batch size?", "docs/CONFIGURATION.md"),
    ("en", "How do I resume or rebuild the FTS5 lexical index migration?", "docs/runbooks/fts5_migration.md"),
    ("en", "How can I resume an interrupted document reindex from a checkpoint?", "docs/reindex-operations.md"),
    ("en", "How can multiple MCP clients share one HTTP knowledge-rag server?", "README.md"),
    ("pt", "Como instalar o knowledge-rag no Windows usando PowerShell?", "docs/INSTALLATION.md"),
    ("pt", "Como funciona o fallback para CPU quando CUDA falha?", "docs/gpu-setup.md"),
    ("pt", "Como configurar o tamanho do lote de embeddings?", "docs/CONFIGURATION.md"),
    ("pt", "Como reconstruir ou retomar a migração do índice lexical FTS5?", "docs/runbooks/fts5_migration.md"),
    ("pt", "Como retomar uma reindexação interrompida usando o checkpoint?", "docs/reindex-operations.md"),
    ("pt", "Como vários clientes MCP podem compartilhar um servidor HTTP?", "README.md"),
)
UNICODE_DOCUMENT = "guia-ação-索引.md"
EDIT_MARKER = "AUDIT-2026-1001"


def copy_corpus(destination: Path) -> list[dict[str, Any]]:
    """Copy only versioned public repository documents, recording exact hashes."""
    manifest = []
    for source_name in (*SOURCES, "CODE_OF_CONDUCT.md"):
        source = ROOT / source_name
        relative = UNICODE_DOCUMENT if source_name == "CODE_OF_CONDUCT.md" else source_name
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest.append(
            {
                "repository_source": source_name,
                "indexed_path": relative,
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    return manifest


def configure(args: Any, directory: Path) -> tuple[Any, Path]:
    """Set every writable path before constructing the real orchestrator."""
    from mcp_server import server
    from mcp_server.config import config

    snapshots = sorted(args.cache.glob("models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/*/model_optimized.onnx"))
    if len(snapshots) != 1:
        raise ValueError(f"Expected one copied BGE-small snapshot in {args.cache}; found {len(snapshots)}")
    values = {
        "base_dir": directory,
        "documents_dir": directory / "documents",
        "data_dir": directory / "data",
        "chroma_dir": directory / "data" / "chroma_db",
        "models_cache_dir": args.cache,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dim": 384,
        "gpu_mode": {"cpu": "false", "cuda": "true", "directml": "directml"}[args.backend],
        "gpu_acceleration": args.backend != "cpu",
        "gpu_device_id": args.device_id,
        "embed_batch_size": args.batch,
        "batch_size": 128,
        "parallel_workers": 1,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "reranker_enabled": False,
        "fts5_enabled": args.fts5,
        "fts5_rerank_enabled": False,
        "collection_name": "retrieval_audit",
        "query_prefix": "",
        "passage_prefix": "",
        "transport": "stdio",
        "metrics_enabled": True,
    }
    for name, value in values.items():
        setattr(config, name, value)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    return server, snapshots[0]


def open_orchestrator(server: Any, model_path: Path, threads: int | None) -> Any:
    """Keep the real embedding adapter while requiring the copied local model."""
    orchestrator = server.KnowledgeOrchestrator()
    options = {"specific_model_path": str(model_path.parent), "local_files_only": True}
    if threads is not None:
        options["threads"] = threads
    orchestrator.embed_fn._init_kwargs.update(options)
    await_fts5(orchestrator)
    return orchestrator


def await_fts5(orchestrator: Any) -> None:
    """Wait for the optional derived index without leaving background writers."""
    index = orchestrator.fts5_index
    if index is None:
        return
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = index.state.read() or {}
        if state.get("status") == "failed":
            raise RuntimeError(f"FTS5 migration failed: {state.get('error')}")
        if state.get("status") == "complete" and index.is_ready():
            return
        time.sleep(0.02)
    raise TimeoutError("FTS5 migration did not complete within 60 seconds")


def close_orchestrator(orchestrator: Any) -> None:
    """Release handles so Windows can reopen and remove the disposable index."""
    if orchestrator is None:
        return
    await_fts5(orchestrator)
    if orchestrator.fts5_index is not None:
        orchestrator.fts5_index.close()
    orchestrator.chroma_client.close()


class RssSampler:
    """Sample actual resident bytes; this is separate from wall-clock timing."""

    def __init__(self) -> None:
        import psutil

        self.process = psutil.Process()
        self.start = self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self) -> None:
        while not self.stop.wait(0.02):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def finish(self) -> dict[str, int]:
        self.stop.set()
        self.thread.join(timeout=2)
        retained = self.process.memory_info().rss
        return {
            "start_bytes": self.start,
            "peak_bytes": max(self.peak, retained),
            "retained_bytes": retained,
            "retained_delta_bytes": retained - self.start,
            "sampling_interval_ms": 20,
        }


def timed_index(orchestrator: Any, *, force: bool = False) -> dict[str, Any]:
    """Count actual parser calls without replacing parser behavior or retaining calls."""
    original = orchestrator.parser.parse_file
    count = 0
    lock = threading.Lock()

    def parse(*args: Any, **kwargs: Any) -> Any:
        nonlocal count
        with lock:
            count += 1
        return original(*args, **kwargs)

    orchestrator.parser.parse_file = parse
    start = time.perf_counter()
    try:
        statistics = orchestrator.index_all(force=force)
    finally:
        elapsed = time.perf_counter() - start
        orchestrator.parser.parse_file = original
    if statistics.get("errors"):
        raise RuntimeError(f"Indexing failed: {statistics}")
    await_fts5(orchestrator)
    return {"seconds": elapsed, "parse_file_calls": count, "statistics": statistics}
