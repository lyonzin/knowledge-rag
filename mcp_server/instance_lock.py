"""Optional single-instance guard for the MCP server process.

Background
----------
MCP stdio servers are 1-process-per-client by protocol design. Multiple
Claude Code windows, Claude Desktop + IDE running simultaneously, or clients
that open extra internal connections during approval/review flows will all
spawn additional `knowledge-rag` processes. Each process holds its own
embedding model, ChromaDB client, BM25 state, and file watcher.

Lazy-loading the embedding model (v3.8.0) reduces idle cost dramatically,
but some users still want a hard cap of one process per data directory.
This module provides that cap as an OPT-IN, never as a default.

Activation
----------
Set the environment variable in your MCP client config:

    KNOWLEDGE_RAG_SINGLE_INSTANCE=1     # also accepts: true, yes, on (case-insensitive)

When unset (default), `single_instance_lock()` is a no-op and the server
behaves exactly as it did before this module existed.

When enabled, the server creates `<data_dir>/knowledge-rag.lock` containing
its PID. A second process starting against the same `data_dir` will detect
the live PID and exit with code 75 (EX_TEMPFAIL). Stale locks (PID gone)
are cleaned up automatically.

Cleanup is wired in three places so the lock does not outlive the process:
1. Normal exit: contextmanager `finally` block removes the lock.
2. SIGINT / SIGTERM: handlers remove the lock and re-raise the default action.
3. Crash / SIGKILL: stale-PID detection on the next startup removes it.

Authors
-------
- Concept and original guard: Sergey Khokhlov (@Hohlas) in PR #31
- Reworked as opt-in + signal handlers + tests: Lyon. (knowledge-rag maintainer)
"""

from __future__ import annotations

import os
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .config import config

LOCK_FILENAME = "knowledge-rag.lock"
ALREADY_RUNNING_EXIT_CODE = 75  # EX_TEMPFAIL from sysexits.h
ENV_VAR = "KNOWLEDGE_RAG_SINGLE_INSTANCE"
_TRUTHY = {"1", "true", "yes", "on"}


class AlreadyRunningError(RuntimeError):
    """Raised when another knowledge-rag server instance already holds the lock."""


def single_instance_enabled() -> bool:
    """Return True if the user opted into the single-instance guard.

    Reads `KNOWLEDGE_RAG_SINGLE_INSTANCE`. Accepts ``1``, ``true``, ``yes``, ``on``
    (case-insensitive, surrounding whitespace ignored). Anything else — including
    unset, empty, ``0``, ``false`` — leaves the guard disabled.
    """
    raw = os.environ.get(ENV_VAR, "").strip().lower()
    return raw in _TRUTHY


def _pid_is_running(pid: int) -> bool:
    """Return True if a process with PID appears to be alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user / has tighter ACLs
        return True
    except OSError:
        return False
    return True


def _read_lock_pid(lock_path: Path) -> Optional[int]:
    try:
        raw = lock_path.read_text(encoding="utf-8").strip().splitlines()[0]
        return int(raw)
    except (IndexError, OSError, ValueError):
        return None


def _lock_path() -> Path:
    return config.data_dir / LOCK_FILENAME


def _remove_if_ours(lock_path: Path) -> None:
    """Remove the lock file ONLY if it still references our PID."""
    if _read_lock_pid(lock_path) == os.getpid():
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # Best-effort; stale-PID check on next startup will recover
            pass


@contextmanager
def single_instance_lock() -> Iterator[Optional[Path]]:
    """Acquire the single-instance lock if opt-in flag is set.

    No-op when ``KNOWLEDGE_RAG_SINGLE_INSTANCE`` is unset / falsy — yields ``None``
    and the caller proceeds normally with no side effects on disk.

    When enabled:
      - Creates ``<data_dir>/knowledge-rag.lock`` containing this process's PID.
      - Raises :class:`AlreadyRunningError` if another live PID already holds it.
      - Recovers stale locks (PID no longer running).
      - Registers SIGINT/SIGTERM handlers that remove the lock and re-raise.
      - Removes the lock on normal exit via ``finally``.
    """
    if not single_instance_enabled():
        yield None
        return

    config.data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path()

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            pid = _read_lock_pid(lock_path)
            if pid is not None and _pid_is_running(pid):
                raise AlreadyRunningError(
                    f"knowledge-rag MCP server is already running (pid {pid}). "
                    f"Refusing to start a second instance because "
                    f"{ENV_VAR} is enabled."
                )
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise AlreadyRunningError(f"Failed to clear stale lock {lock_path}: {exc}") from exc
            continue

        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n")
        break

    # Wire signal handlers so SIGINT/SIGTERM cleanup the lock before exit
    previous_handlers: dict[int, object] = {}

    def _signal_cleanup(signum: int, frame: object) -> None:
        _remove_if_ours(lock_path)
        # Restore original handler and re-raise so default action runs
        prev = previous_handlers.get(signum, signal.SIG_DFL)
        try:
            signal.signal(signum, prev)  # type: ignore[arg-type]
        except (ValueError, OSError):
            pass
        # Re-send the signal to ourselves so the original disposition fires
        os.kill(os.getpid(), signum)

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _signal_cleanup)
        except (ValueError, OSError):
            # signal.signal raises if not on the main thread; tests may hit this
            pass

    try:
        yield lock_path
    finally:
        _remove_if_ours(lock_path)
        for sig, prev in previous_handlers.items():
            try:
                signal.signal(sig, prev)  # type: ignore[arg-type]
            except (ValueError, OSError):
                pass
