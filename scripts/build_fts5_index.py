#!/usr/bin/env python3
"""Standalone rebuilder for the FTS5 lexical index (Task 05 / ADR-008).

Use when the daemon's lazy migration is inconvenient — e.g. suspected
corruption, migration stall, or a maintenance window when the operator
wants to block until the rebuild is done.

Reads the corpus straight from ChromaDB (source of truth) and repopulates
``<data_dir>/fts5_index.db``. Always operates against the same paths the
running daemon would use; if you point it at a hot data_dir make sure the
daemon is stopped or the CRUD hooks are quiet, otherwise WAL contention is
harmless but rebuild progress becomes fuzzy.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running directly from source checkout ``python scripts/build_fts5_index.py``.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the FTS5 lexical index from ChromaDB.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to the knowledge-rag data directory (defaults to config.data_dir).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop the existing fts5_index.db + marker file before rebuilding.",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Block until migration completes (default). Retained for symmetry.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Emit per-batch progress.")
    return parser.parse_args(argv)


def _resolve_data_dir(cli_arg: Path | None) -> Path:
    if cli_arg is not None:
        cli_arg.mkdir(parents=True, exist_ok=True)
        os.environ["KRAG_DATA_DIR"] = str(cli_arg)
        return cli_arg
    from mcp_server.config import config

    return Path(config.data_dir)


def _drop_existing(data_dir: Path) -> None:
    db = data_dir / "fts5_index.db"
    state = data_dir / "fts5_migration.state"
    for path in (db, state, db.with_suffix(".db-wal"), db.with_suffix(".db-shm")):
        try:
            path.unlink(missing_ok=True)
            print(f"[BUILD-FTS5] removed {path}")
        except OSError as exc:
            print(f"[BUILD-FTS5] could not remove {path}: {exc}")


def _iter_chroma_chunks(collection) -> list[tuple[str, str, str, str]]:
    count = collection.count()
    if count == 0:
        return []
    fetched = collection.get(include=["documents", "metadatas"], limit=count)
    ids = fetched.get("ids") or []
    docs = fetched.get("documents") or []
    metas = fetched.get("metadatas") or []
    rows: list[tuple[str, str, str, str]] = []
    for chunk_id, content, meta in zip(sorted(ids), docs, metas):
        # Preserve alignment with sorted ids
        idx = ids.index(chunk_id)
        meta_i = metas[idx] or {}
        rows.append(
            (
                str(chunk_id),
                str(docs[idx] or ""),
                str(meta_i.get("filename", "")),
                str(meta_i.get("category", "")),
            )
        )
    return rows


def _open_index(data_dir: Path):
    from mcp_server.fts5_index import Fts5LexicalIndex

    db = data_dir / "fts5_index.db"
    state = data_dir / "fts5_migration.state"
    return Fts5LexicalIndex(db_path=db, state_path=state)


def _open_collection(data_dir: Path):
    import chromadb

    from mcp_server.config import config

    client = chromadb.PersistentClient(path=str(config.chroma_dir))
    return client.get_or_create_collection(name=config.collection_name)


def _run_migration_sync(index, rows: list[tuple[str, str, str, str]], verbose: bool) -> None:
    total = len(rows)
    started_at = datetime.now(timezone.utc).isoformat()
    index._write_state("in_progress", total, 0, started_at, None, None)  # noqa: SLF001
    docs_indexed = 0
    batch: list[tuple[str, str, str, str]] = []
    last_percent = -10
    for row in rows:
        batch.append(row)
        if len(batch) >= 100:
            index._populate_batch(batch)  # noqa: SLF001
            docs_indexed += len(batch)
            batch = []
            index._write_state("in_progress", total, docs_indexed, started_at, None, None)  # noqa: SLF001
            percent = int(100 * docs_indexed / max(1, total))
            if verbose or percent >= last_percent + 10:
                print(f"[BUILD-FTS5] {percent}% ({docs_indexed}/{total})")
                last_percent = percent
    if batch:
        index._populate_batch(batch)  # noqa: SLF001
        docs_indexed += len(batch)
    completed_at = datetime.now(timezone.utc).isoformat()
    index._write_state("complete", total, docs_indexed, started_at, completed_at, None)  # noqa: SLF001
    print(f"[BUILD-FTS5] complete: {docs_indexed} docs indexed")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_dir = _resolve_data_dir(args.data_dir)
    print(f"[BUILD-FTS5] data_dir={data_dir} force={args.force}")

    if args.force:
        _drop_existing(data_dir)

    start = time.time()
    collection = _open_collection(data_dir)
    rows = _iter_chroma_chunks(collection)
    if not rows:
        print("[BUILD-FTS5] corpus is empty — nothing to index")
        return 0

    index = _open_index(data_dir)
    try:
        _run_migration_sync(index, rows, args.verbose)
    finally:
        index.close()

    elapsed = time.time() - start
    print(f"[BUILD-FTS5] elapsed_seconds={elapsed:.1f}")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entrypoint
    raise SystemExit(main())
