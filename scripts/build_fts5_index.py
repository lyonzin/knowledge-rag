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
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp_server.fts5_index import Fts5LexicalIndex

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
        return cli_arg.expanduser().resolve()
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
            raise RuntimeError(f"Could not remove {path}: {exc}") from exc


def _iter_chroma_chunks(collection: Any) -> Iterator[tuple[str, str, str, str]]:
    """Keep documents and metadata aligned while streaming bounded pages."""
    batch_size = 500
    for offset in range(0, collection.count(), batch_size):
        fetched = collection.get(include=["documents", "metadatas"], limit=batch_size, offset=offset)
        for chunk_id, content, meta in zip(
            fetched.get("ids") or [], fetched.get("documents") or [], fetched.get("metadatas") or []
        ):
            metadata = meta or {}
            yield (
                str(chunk_id),
                str(content or ""),
                str(metadata.get("filename", "")),
                str(metadata.get("category", "")),
            )


def _open_index(data_dir: Path) -> "Fts5LexicalIndex":
    from mcp_server.fts5_index import Fts5LexicalIndex

    db = data_dir / "fts5_index.db"
    state = data_dir / "fts5_migration.state"
    return Fts5LexicalIndex(db_path=db, state_path=state)


def _open_collection(data_dir: Path | None) -> Any:
    """Honor the configured Chroma path unless a data-directory override is explicit."""
    import chromadb

    from mcp_server.config import config

    chroma_dir = Path(config.chroma_dir) if data_dir is None else data_dir / "chroma_db"
    if not (chroma_dir / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"No existing ChromaDB database at {chroma_dir}")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_collection(name=config.collection_name, embedding_function=None)


def _run_migration_sync(index: "Fts5LexicalIndex", collection: Any, verbose: bool) -> None:
    """Reuse transactional migration and its failure/checkpoint handling."""
    from mcp_server.fts5_index import Fts5MigrationError

    def report(done: int, total: int) -> None:
        if verbose:
            print(f"[BUILD-FTS5] {done}/{total}")

    index._migration_worker(lambda: _iter_chroma_chunks(collection), collection.count(), 0, report)  # noqa: SLF001
    state = index.state.read() or {}
    if state.get("status") != "complete":
        raise Fts5MigrationError(str(state.get("error", "FTS5 rebuild did not complete")))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_dir = _resolve_data_dir(args.data_dir)
    print(f"[BUILD-FTS5] data_dir={data_dir} force={args.force}")

    start = time.time()
    collection = _open_collection(None if args.data_dir is None else data_dir)
    if args.force:
        _drop_existing(data_dir)

    index = _open_index(data_dir)
    try:
        _run_migration_sync(index, collection, args.verbose)
    finally:
        index.close()

    elapsed = time.time() - start
    print(f"[BUILD-FTS5] elapsed_seconds={elapsed:.1f}")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entrypoint
    raise SystemExit(main())
