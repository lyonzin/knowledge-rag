"""Who may touch the index files: the owner, or one of its preflight probes, never both at once.

The preflight probe reads the index in a child process, and a reader that
opens Chroma while it is written can crash or leave HNSW segments unloadable.
The writer guard (``knowledge-rag.lock.guard``) belongs to the owner process
alone, so it does not cover a probe that outlives its owner, starts late or
runs past its timeout. This access guard, an OS lock on the stable
``knowledge-rag.lock.access`` sidecar, does. A probe takes it before it
touches the index and lets it go only by exiting; the owner takes it after
its probes, before it moves, repairs or opens the index, and keeps it until
it dies. A probe that finds it taken exits with BUSY_EXIT_CODE without
touching anything, which the owner reads as contention, not damage. The
sidecar is never unlinked, for the reason given in ``instance_lock``.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from ._sidecar_lock import lock_sidecar

ACCESS_FILENAME = "knowledge-rag.lock.access"
BUSY_EXIT_CODE = 75  # EX_TEMPFAIL: another process is inside the index; try again later
WAIT_SECONDS = 120.0  # past a probe's own timeout, so only a process stuck inside runs it out
_FIRST_POLL_SECONDS = 0.1
_LONGEST_POLL_SECONDS = 2.0
_BUSY = (
    "Another process kept the index files of this data directory for more than {seconds} s, so this session "
    "did not open them. It is most likely a startup check left behind by a knowledge-rag session that ended "
    "abruptly: a Python process started with -c that imports mcp_server. End it, then restart this client."
)

# The owner's hold (descriptor, sidecar path): only this process's death releases it.
_held: Optional[tuple[int, Path]] = None


class IndexAccessBusy(RuntimeError):
    """Another process stayed inside the index files for longer than the owner waits."""


def busy(wait_seconds: float) -> IndexAccessBusy:
    """The error for a process that stayed inside the index past ``wait_seconds``. By Lyon."""
    return IndexAccessBusy(_BUSY.format(seconds=round(wait_seconds)))


def enter_or_exit(path: Path) -> None:
    """Probe only the supplied sidecar; never load config to discover its path in the child."""
    if lock_sidecar(path) is None:  # keep the descriptor open until the child exits
        raise SystemExit(BUSY_EXIT_CODE)


@contextmanager
def held_for_call(path: Path) -> Iterator[None]:
    """Give back only the descriptor this standalone check acquired."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = _wait_for(path, WAIT_SECONDS)
    try:
        yield
    finally:
        os.close(fd)


def hold(wait_seconds: float = WAIT_SECONDS, *, path: Optional[Path] = None) -> None:
    """The owner's hold: wait until no probe is inside, then keep the guard until this process dies.

    Idempotent, so the preflight and the opening can both ask. Raises
    IndexAccessBusy when another process stays inside past ``wait_seconds``.
    By Lyon.
    """
    global _held
    if path is None:  # parent-only, when opening the server index
        from .config import config

        path = config.data_dir.resolve() / ACCESS_FILENAME
    if _held is not None:
        if _held[1] != path:
            raise RuntimeError(f"this process holds the index access of {_held[1].parent}, not {path.parent}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _held = (_wait_for(path, wait_seconds), path)


def _wait_for(path: Path, wait_seconds: float) -> int:
    """The guard's descriptor once no other process holds it, polling with a growing pause. By Lyon."""
    deadline = time.monotonic() + wait_seconds
    pause = _FIRST_POLL_SECONDS
    fd = lock_sidecar(path)
    if fd is None:
        print("[ACCESS] Another process is inside the index files; waiting before touching them.", file=sys.stderr)
    while fd is None:
        if time.monotonic() >= deadline:
            raise busy(wait_seconds)
        time.sleep(pause)
        pause = min(2 * pause, _LONGEST_POLL_SECONDS)
        fd = lock_sidecar(path)
    return fd


def is_held() -> bool:
    """Whether this process holds the access guard. By Lyon."""
    return _held is not None


def _release() -> None:
    """Tests only: give up this process's hold the way its death would. By Lyon."""
    global _held
    if _held is not None:
        fd, _held = _held[0], None
        os.close(fd)
