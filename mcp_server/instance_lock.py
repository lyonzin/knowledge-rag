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

When enabled, an OS lock on the stable `knowledge-rag.lock.guard` sidecar
serializes acquisition and protects the entire context. The PID file
`knowledge-rag.lock` remains readable for diagnostics and legacy servers.
The sidecar is never unlinked: replacing its inode would split the guard.

Normal exit removes our PID and closes the guarded descriptor. SIGINT and
SIGTERM preserve prior handlers; ignored or handled signals retain ownership
until the context exits. Crash / SIGKILL releases the OS lock automatically;
the next owner recovers the stale PID file while holding the same OS lock.

Authors
-------
- Concept and original guard: Sergey Khokhlov (@Hohlas) in PR #31
- Reworked as opt-in + signal handlers + tests: Lyon. (knowledge-rag maintainer)
"""

from __future__ import annotations

import errno
import os
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Callable, Iterator, Optional

from .config import config

LOCK_FILENAME = "knowledge-rag.lock"
GUARD_FILENAME = f"{LOCK_FILENAME}.guard"
ALREADY_RUNNING_EXIT_CODE = 75  # EX_TEMPFAIL from sysexits.h
ENV_VAR = "KNOWLEDGE_RAG_SINGLE_INSTANCE"
_TRUTHY = {"1", "true", "yes", "on"}
SignalHandler = Optional[Callable[[int, Optional[FrameType]], object] | int]


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
    if os.name == "nt":
        return _windows_pid_is_running(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user / has tighter ACLs
        return True
    except (OSError, OverflowError):
        return False
    return True


def _windows_pid_is_running(pid: int) -> bool:
    """Query a process handle without sending Windows termination signals."""
    import ctypes
    from ctypes import wintypes

    if pid > 0xFFFFFFFF:
        return False  # Win32 PIDs are DWORDs; never truncate an untrusted lock PID.
    # These entry points only exist on Windows; lazy lookup keeps Unix imports
    # and platform-independent handle-probe tests valid without fake stubs.
    kernel = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE access only.
    if not handle:
        # ERROR_INVALID_PARAMETER means PID gone; denied/unknown is inconclusive.
        return bool(getattr(ctypes, "get_last_error")() != 87)
    try:
        # A process object is signaled on exit. WAIT_TIMEOUT or WAIT_FAILED is
        # conservatively live, so uncertainty cannot steal another server's lock.
        return bool(kernel.WaitForSingleObject(handle, 0) != 0)
    finally:
        kernel.CloseHandle(handle)


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


def _already_running(lock_path: Path) -> AlreadyRunningError:
    pid = _read_lock_pid(lock_path)
    owner = f"pid {pid}" if pid is not None else "OS lock held"
    return AlreadyRunningError(
        f"knowledge-rag MCP server is already running ({owner}). "
        f"Refusing to start a second instance because {ENV_VAR} is enabled."
    )


def _acquire_os_lock(fd: int) -> None:
    """Lock the stable sidecar without blocking, including beyond EOF on Windows."""
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


@contextmanager
def _process_guard(lock_path: Path) -> Iterator[None]:
    """Keep one kernel-owned descriptor open, leaving the sidecar inode intact."""
    fd = os.open(lock_path.with_name(GUARD_FILENAME), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            _acquire_os_lock(fd)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise _already_running(lock_path) from exc
            raise
        yield
    finally:
        # Closing releases flock/byte-range locks, even if Python cleanup fails.
        os.close(fd)


def _write_pid(lock_path: Path) -> None:
    """Preserve the PID-file protocol for an already-running legacy server."""
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            pid = _read_lock_pid(lock_path)
            if pid is not None and _pid_is_running(pid):
                raise _already_running(lock_path)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise AlreadyRunningError(f"Failed to clear stale lock {lock_path}: {exc}") from exc
            continue

        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n")
        return


@contextmanager
def _signal_handlers() -> Iterator[None]:
    """Delegate signals without releasing ownership while the caller can continue."""
    previous_handlers: dict[int, SignalHandler] = {}

    def _signal_cleanup(signum: int, frame: Optional[FrameType]) -> None:
        prev = previous_handlers.get(signum, signal.SIG_DFL)
        # os.kill is TerminateProcess on Windows for SIGINT/SIGTERM, bypassing
        # Python cleanup. Delegate Python handlers directly on every platform.
        if callable(prev):
            prev(signum, frame)
        elif prev != signal.SIG_IGN:
            if signum == signal.SIGINT:
                signal.default_int_handler(signum, frame)
            raise SystemExit(128 + signum)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, _signal_cleanup)
            except (ValueError, OSError):
                # Python only permits signal installation on the main thread.
                previous_handlers.pop(sig, None)
        yield
    finally:
        for signum, prev in previous_handlers.items():
            try:
                signal.signal(signum, prev)
            except (ValueError, OSError):
                pass


@contextmanager
def single_instance_lock() -> Iterator[Optional[Path]]:
    """Hold a per-directory OS lock and compatible PID file when opted in.

    Disabled guards yield ``None`` without touching the filesystem. Enabled
    guards reject another owner, including live PID files from older servers.
    The stable sidecar survives exit; its existence never means it is locked.
    """
    if not single_instance_enabled():
        yield None
        return

    config.data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path()
    with _process_guard(lock_path):
        _write_pid(lock_path)
        try:
            with _signal_handlers():
                yield lock_path
        finally:
            _remove_if_ours(lock_path)
