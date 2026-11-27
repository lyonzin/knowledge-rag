"""The child probe's wire contract. This module and its guard must not import config."""

from __future__ import annotations

import json
import signal
import stat
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from . import _index_access

_READING = "counting"
_HNSW = "hnsw-load-failed"
_BUSY = "busy"
_MISSING = "missing"
_DAMAGED_EXIT = 65  # EX_DATAERR; only meaningful after the exact target and read markers
_SEGFAULT_CODES = {-signal.SIGSEGV, 0xC0000005, -1073741819}  # POSIX signal / Windows access violation
if hasattr(signal, "SIGBUS"):
    _SEGFAULT_CODES.add(-signal.SIGBUS)
_HNSW_ERROR = "Error loading hnsw index"
# Chroma 1.5.9, from a persisted segment opened in a fresh process. Unknown wording fails closed.
_HNSW_ERRORS = {
    _HNSW_ERROR,
    "Error executing plan: Error sending backfill request to compactor: "
    f"Error constructing hnsw segment reader: Error creating hnsw segment reader: {_HNSW_ERROR}",
}


def _file_id(path: Path, kind: int) -> tuple[int, int] | None:
    try:
        record = path.lstat()
    except FileNotFoundError:
        return None
    reparse = getattr(record, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    if stat.S_ISLNK(record.st_mode) or reparse or (kind == stat.S_IFREG and record.st_nlink > 1):
        raise RuntimeError(f"Refusing index alias at {path}")
    if stat.S_IFMT(record.st_mode) != kind or not record.st_ino:
        raise RuntimeError(f"Cannot establish the filesystem identity of {path}")
    return record.st_dev, record.st_ino


_IndexId = tuple[tuple[int, int] | None, tuple[int, int] | None]


def _index_id(directory: Path) -> _IndexId:
    # Check ancestors from the root: lstat of a leaf alone follows a junction or symlink in its parent path.
    if ".." in directory.parts:
        raise RuntimeError(f"Refusing ambiguous index alias at {directory}")
    for parent in reversed(directory.parents):
        if _file_id(parent, stat.S_IFDIR) is None:
            return None, None
    return _file_id(directory, stat.S_IFDIR), _file_id(directory / "chroma.sqlite3", stat.S_IFREG)


@dataclass(frozen=True)
class _Target:
    data_dir: Path
    chroma_dir: Path
    collection_name: str
    index_id: _IndexId

    @property
    def access_path(self) -> Path:
        return self.data_dir / _index_access.ACCESS_FILENAME

    def wire(self) -> str:
        return json.dumps(
            [str(self.data_dir), str(self.chroma_dir), self.collection_name, str(self.access_path), self.index_id]
        )

    def check_identity(self) -> None:
        if _index_id(self.chroma_dir) != self.index_id:
            raise RuntimeError(f"Index changed since the preflight snapshot: {self.chroma_dir}; refusing to use it.")


class _Outcome(Enum):
    VERIFIED = "verified"
    BUSY = "busy"
    UNVERIFIED = "unverified"
    HNSW = "hnsw"
    NATIVE = "segfault"


def _enter(raw_target: str) -> _Target:
    """Echo the actual arguments; the sidecar stays held until this child exits."""
    data_dir, chroma_dir, collection_name, access_path, _ = json.loads(raw_target)
    try:
        _index_access.enter_or_exit(Path(access_path))
    except SystemExit as exc:
        if exc.code == _index_access.BUSY_EXIT_CODE:
            print(raw_target, _BUSY, sep="\n", flush=True)
        raise
    directory = Path(chroma_dir)
    target = _Target(Path(data_dir), directory, collection_name, _index_id(directory))
    if target.wire() != raw_target:
        raise RuntimeError(f"Index changed before the child could verify {directory}")
    print(target.wire(), flush=True)
    return target


def _read_index(target: _Target) -> None:
    import chromadb
    from chromadb.errors import InternalError, NotFoundError

    if target.index_id[1] is None:
        if target.index_id[0] is not None and any(target.chroma_dir.iterdir()):
            raise RuntimeError("Non-empty index directory without chroma.sqlite3")
        print(_MISSING, flush=True)
        return
    client = chromadb.PersistentClient(path=str(target.chroma_dir))
    try:
        # GH #248/#252: never create a collection or persist a default embedding function.
        collection = client.get_collection(name=target.collection_name, embedding_function=None)
    except NotFoundError:
        print(_MISSING, flush=True)
        return
    count = collection.count
    target.check_identity()
    print(_READING, flush=True)  # No imports, bootstrap or client/collection lookup inside this window.
    try:
        size = count()
    except BaseException as exc:
        if (
            type(exc) is InternalError
            and exc.__cause__ is None
            and exc.__context__ is None
            and str(exc) in _HNSW_ERRORS
        ):
            print(_HNSW, flush=True)
            print(exc, file=sys.stderr)
            raise SystemExit(_DAMAGED_EXIT) from exc
        print("read-error", flush=True)  # A crash during unwinding is not a crash in count().
        raise
    print(size, flush=True)


def _classify(target: _Target, done: subprocess.CompletedProcess[str]) -> _Outcome:
    """Only a complete, target-bound transcript or an interrupted native read carries evidence."""
    lines = (done.stdout or "").splitlines()
    if not lines or lines[0] != target.wire():
        return _Outcome.UNVERIFIED
    events = lines[1:]
    if done.returncode == _index_access.BUSY_EXIT_CODE and events == [_BUSY]:
        return _Outcome.BUSY
    if done.returncode == _DAMAGED_EXIT and events == [_READING, _HNSW]:
        return _Outcome.HNSW
    if done.returncode in _SEGFAULT_CODES and events == [_READING]:
        return _Outcome.NATIVE
    counted = len(events) == 2 and events[0] == _READING and events[1].isascii() and events[1].isdigit()
    if done.returncode == 0 and (events == [_MISSING] or counted):
        return _Outcome.VERIFIED
    return _Outcome.UNVERIFIED
