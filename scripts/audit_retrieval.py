"""Opt-in offline CPU/CUDA/DirectML E2E audit using a disposable index.

Example: python scripts/audit_retrieval.py --backend cpu --cache .cache/models
         --output .cache/audit/e2e-cpu.json

Run CPU and CUDA separately in the same interpreter for a useful comparison.
English BGE-small is used; Portuguese retrieval quality is measured, not assumed.
The default pytest suite does not run this workload or download any model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_workload import (  # noqa: E402
    CASES,
    EDIT_MARKER,
    UNICODE_DOCUMENT,
    RssSampler,
    close_orchestrator,
    configure,
    copy_corpus,
    open_orchestrator,
    timed_index,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda", "directml"), required=True)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache" / "models")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--fts5", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.rounds < 1 or (args.batch is not None and not 1 <= args.batch <= 512):
        parser.error("rounds must be positive and batch must be between 1 and 512")
    if args.backend == "directml" and (args.device_id is None or not 0 <= args.device_id <= 0x7FFFFFFF):
        parser.error("DirectML requires an explicit non-negative DXGI --device-id")
    args.cache, args.output = args.cache.resolve(), args.output.resolve()
    return args


def validate_index(orchestrator: Any) -> dict[str, Any]:
    """Check stored IDs, source references and real vector finiteness in pages."""
    import numpy as np

    count = orchestrator.collection.count()
    ids: set[str] = set()
    sources: set[str] = set()
    for offset in range(0, count, 128):
        page = orchestrator.collection.get(limit=128, offset=offset, include=["embeddings", "metadatas"])
        vectors = np.asarray(page["embeddings"])
        assert vectors.shape == (len(page["ids"]), 384) and np.isfinite(vectors).all(), vectors.shape
        assert not ids.intersection(page["ids"]), "Duplicate stored chunk IDs"
        ids.update(page["ids"])
        for metadata in page["metadatas"]:
            source = Path(metadata["source"])
            assert source.is_file(), f"Orphan source: {source}"
            sources.add(str(source))
    assert len(ids) == count and count > 0
    if orchestrator.fts5_index is not None:
        assert orchestrator.fts5_index.count() == count, "FTS5/Chroma chunk counts differ"
    return {
        "chunks": count,
        "sources": len(sources),
        "ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        "vectors_shape": [count, 384],
        "all_vectors_finite": True,
    }


def measure_queries(orchestrator: Any, rounds: int) -> dict[str, Any]:
    """Separate uncached hybrid latency, cache latency, and source-based relevance."""
    import numpy as np

    uncached: list[float] = []
    cached: list[float] = []
    quality: list[dict[str, Any]] = []
    for round_index in range(rounds):
        for language, query, expected in CASES:
            orchestrator.query_cache.invalidate()
            start = time.perf_counter()
            hits = orchestrator.query(query, max_results=5, search_method="hybrid")
            uncached.append((time.perf_counter() - start) * 1000)
            assert hits, f"No results for {query!r}"
            start = time.perf_counter()
            assert orchestrator.query(query, max_results=5, search_method="hybrid") == hits
            cached.append((time.perf_counter() - start) * 1000)
            if round_index == 0:
                sources = [hit["source"].replace("\\", "/") for hit in hits]
                rank = next((index + 1 for index, source in enumerate(sources) if source.endswith(expected)), None)
                quality.append(
                    {
                        "language": language,
                        "query": query,
                        "expected": expected,
                        "rank": rank,
                        "top_sources": [source.split("/documents/")[-1] for source in sources],
                    }
                )
    metrics = {
        name: {
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "count": len(values),
            "samples_ms": values,
        }
        for name, values in (("uncached", uncached), ("cached", cached))
    }
    metrics["quality_by_language"] = {
        language: {
            "recall_at_5": sum(case["rank"] is not None for case in quality if case["language"] == language) / 6,
            "mrr_at_5": sum(1 / case["rank"] if case["rank"] else 0 for case in quality if case["language"] == language)
            / 6,
        }
        for language in ("en", "pt")
    }
    return {**metrics, "cases": quality}


def concurrent_reads(orchestrator: Any) -> dict[str, Any]:
    orchestrator.query_cache.invalidate()
    queries = [case[1] for case in CASES[:6]]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda query: orchestrator.query(query, max_results=5), queries))
    assert all(results), "Concurrent read returned no results"
    return {"workers": 3, "queries": len(queries), "seconds": time.perf_counter() - start}


def mutate_corpus(orchestrator: Any, documents: Path) -> dict[str, Any]:
    unicode_path = documents / UNICODE_DOCUMENT
    with unicode_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n## Audit revision\n{EDIT_MARKER}: the revised Unicode document remains searchable.\n")
    edited = timed_index(orchestrator)
    assert edited["parse_file_calls"] == 1, edited
    if orchestrator.fts5_index is not None:
        hits = orchestrator.query(EDIT_MARKER, max_results=5, search_method="fts5")
        assert any(Path(hit["source"]).name == UNICODE_DOCUMENT for hit in hits)
    deleted_path = documents / "docs/reindex-operations.md"
    deleted_path.unlink()
    deleted = timed_index(orchestrator)
    assert deleted["parse_file_calls"] == 0, deleted
    assert orchestrator.collection.get(where={"source": str(deleted_path)}, include=[])["ids"] == []
    before_force = validate_index(orchestrator)
    forced = timed_index(orchestrator, force=True)
    after_force = validate_index(orchestrator)
    assert before_force == after_force, (before_force, after_force)
    assert forced["parse_file_calls"] == 6, forced
    return {
        "edited_one_document": edited,
        "deleted_one_document": deleted,
        "forced_reindex": forced,
        "integrity_after_mutations": after_force,
    }


def versions() -> dict[str, str]:
    result = {}
    for package in (
        "fastembed",
        "onnxruntime",
        "onnxruntime-gpu",
        "onnxruntime-directml",
        "chromadb",
        "numpy",
        "psutil",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def workload(args: Any, directory: Path, report: dict[str, Any]) -> None:
    server, model_path = configure(args, directory)
    report["corpus"] = copy_corpus(server.config.documents_dir)
    report["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    orchestrator = None
    try:
        start = time.perf_counter()
        orchestrator = open_orchestrator(server, model_path, args.threads)
        report["initialization_seconds"] = time.perf_counter() - start
        report["initial_index"] = timed_index(orchestrator)
        report["integrity_initial"] = validate_index(orchestrator)
        provider = orchestrator.embed_fn._active_provider
        report["active_provider"] = provider
        report["session_providers"] = orchestrator.embed_fn._model.model.model.get_providers()
        report["embedding_batch_size"] = orchestrator.embed_fn._embedding_batch_size()
        expected = {"cuda": "CUDA", "cpu": "CPU", "directml": "DirectML"}[args.backend]
        assert provider == expected, f"Requested {args.backend}, actual provider is {provider}"
        report["unchanged_index"] = timed_index(orchestrator)
        assert report["unchanged_index"]["parse_file_calls"] == 0, report["unchanged_index"]
        report["retrieval"] = measure_queries(orchestrator, args.rounds)
        report["concurrent_reads"] = concurrent_reads(orchestrator)
        report["mutations"] = mutate_corpus(orchestrator, server.config.documents_dir)
        expected_integrity = validate_index(orchestrator)
        close_orchestrator(orchestrator)
        orchestrator = None
        gc.collect()
        start = time.perf_counter()
        orchestrator = open_orchestrator(server, model_path, args.threads)
        report["reopen_seconds"] = time.perf_counter() - start
        assert validate_index(orchestrator) == expected_integrity
        report["reopened_unchanged_index"] = timed_index(orchestrator)
        assert report["reopened_unchanged_index"]["parse_file_calls"] == 0
        report["reopened_query_sources"] = [
            hit["source"].split("documents")[-1] for hit in orchestrator.query(CASES[0][1])
        ]
        assert report["reopened_query_sources"]
        report["checks_passed"] = True
    finally:
        close_orchestrator(orchestrator)


def main() -> int:
    args = arguments()
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "KNOWLEDGE_RAG_WATCHER_DISABLED": "1",
        }
    )
    os.environ.pop("KNOWLEDGE_RAG_EMBED_BATCH_SIZE", None)
    report = {
        "requested_backend": args.backend,
        "requested_device_id": args.device_id,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions(),
        "rounds": args.rounds,
        "fts5": args.fts5,
        "checks_passed": False,
        "caveats": [
            "English embedding model; Portuguese recall is measured without a guaranteed threshold.",
            "Session provider presence alone does not prove every graph node executes on GPU.",
            "Reranker disabled to isolate indexing and hybrid retrieval.",
        ],
    }
    memory = RssSampler()
    try:
        with tempfile.TemporaryDirectory(prefix="knowledge-rag-e2e-audit-") as directory:
            os.environ["KNOWLEDGE_RAG_DIR"] = directory
            workload(args, Path(directory), report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        gc.collect()
        report["rss"] = memory.finish()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["checks_passed"] and "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
