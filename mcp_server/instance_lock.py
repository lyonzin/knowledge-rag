"""One owner per data directory for the MCP server process.

Every MCP stdio client spawns its own ``knowledge-rag`` process. Only one of
them may open ChromaDB: a second writer, and even a reader opening it during
writes, left HNSW segments unloadable. The owner holds a nonblocking OS lock
on the stable ``knowledge-rag.lock.guard`` sidecar until it dies (claimed in
``mcp_server._writer_role``); the others follow. That lock decides: the PID on
line 1 of ``knowledge-rag.lock`` serves diagnostics and older servers, which
lock nothing else, and line 2 (GUARDED_MARK) tells their files from ours.

``KNOWLEDGE_RAG_SINGLE_INSTANCE=1`` (also true/yes/on; HTTP transports set
it) makes a later process exit instead. ``single_instance_lock()`` keeps its
opt-in contract and ``writer_lock()`` scopes ownership for tests and scripts.
The sidecar is never unlinked: a new inode would split the guard. A crash
releases the lock and the next owner replaces the stale PID file.

Authors: Sergey Khokhlov (@Hohlas) wrote the original guard (PR #31); Lyon.
made it opt-in with signal handling and tests, then the single-writer lock.
"""

from __future__ import annotations

import codecs
import os
import signal
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import FrameType
from typing import Callable, Iterator, Optional

from ._sidecar_lock import lock_sidecar as _lock_sidecar
from .config import config

LOCK_FILENAME = "knowledge-rag.lock"
GUARD_FILENAME = f"{LOCK_FILENAME}.guard"
ALREADY_RUNNING_EXIT_CODE = 75  # EX_TEMPFAIL from sysexits.h
ENV_VAR = "KNOWLEDGE_RAG_SINGLE_INSTANCE"
GUARDED_MARK = "guarded"  # line 2 of a PID file written under the OS guard (see _write_pid)
_TRUTHY = {"1", "true", "yes", "on"}
_SHARING_RETRY_DELAYS = (0.01, 0.02, 0.04, 0.08)  # transient Windows PID readers that deny delete sharing
_ERROR_SHARING_VIOLATION = 32
_standalone = threading.local()
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


def _lock_record(lock_path: Path) -> tuple[Optional[int], bool]:
    """Read one snapshot; damaged trailing bytes must not hide an identifiable legacy PID."""
    try:
        raw = lock_path.read_bytes()
    except FileNotFoundError:
        return None, False
    encoding = "utf-16" if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8-sig"
    lines = raw.decode(encoding, errors="replace").splitlines()
    try:
        pid = int(lines[0])
    except (ValueError, IndexError):
        return None, False
    return pid, lines[1:2] == [GUARDED_MARK]


def _read_lock_pid(lock_path: Path) -> Optional[int]:
    """Best-effort PID for diagnostics and cleanup; admission uses _lock_record directly."""
    try:
        return _lock_record(lock_path)[0]
    except OSError:
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
        f"Refusing to start a second instance because {ENV_VAR} is enabled (HTTP transports enable it). "
        "Close the other knowledge-rag sessions on this data directory, stdio clients included, and retry."
    )


def _lock_guard(lock_path: Path) -> int:
    """The writer guard's descriptor, locked without blocking; AlreadyRunningError when busy. By Lyon."""
    fd = _lock_sidecar(lock_path.with_name(GUARD_FILENAME))
    if fd is None:
        raise _already_running(lock_path)
    return fd


def _write_pid(lock_path: Path) -> None:
    """Write our PID file under the OS guard; a live older server's PID file is respected."""
    retries = iter(_SHARING_RETRY_DELAYS)
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            pid, guarded = _lock_record(lock_path)
            if not guarded and pid is not None and _pid_is_running(pid):
                raise _already_running(lock_path)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                if os.name != "nt" or getattr(exc, "winerror", None) != _ERROR_SHARING_VIOLATION:
                    raise
                delay = next(retries, None)
                if delay is None:
                    raise
                time.sleep(delay)  # recheck the record under the same guard before every retry
            continue

        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n{GUARDED_MARK}\n")
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
    with _owned(lock_path.parent.resolve() / lock_path.name) as held_path:
        previous = getattr(_standalone, "owner", None)
        _standalone.owner = (os.getpid(), held_path)
        try:
            yield lock_path
        finally:
            _standalone.owner = previous


@contextmanager
def standalone_guard(lock_path: Path) -> Iterator[None]:
    """Borrow only this thread's opt-in standalone lease, never a running server's claim. By Lyon."""
    lock_path = lock_path.parent.resolve() / lock_path.name
    if getattr(_standalone, "owner", None) == (os.getpid(), lock_path):
        yield
    else:
        with _owned(lock_path):
            yield


@contextmanager
def writer_lock() -> Iterator[bool]:
    """Yield True when this process owns writes to data_dir, False for a read-only follower.

    With the opt-in variable set (HTTP transports set it), a busy lock raises
    AlreadyRunningError as single_instance_lock() does. By Lyon.
    """
    config.data_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        yield _enter_owned(stack, _lock_path())  # outside any except: a body error propagates unchanged


def _enter_owned(stack: ExitStack, lock_path: Path) -> bool:
    """Push ownership onto ``stack``; False when another process owns data_dir. By Lyon."""
    try:
        stack.enter_context(_owned(lock_path))
    except AlreadyRunningError:
        if single_instance_enabled():
            raise
        return False
    return True


@contextmanager
def _owned(lock_path: Path) -> Iterator[Path]:
    """Hold the OS guard and our PID file until the context exits; the sidecar inode stays. By Lyon."""
    fd = _lock_guard(lock_path)
    try:
        _write_pid(lock_path)
        try:
            with _signal_handlers():
                yield lock_path
        finally:
            _remove_if_ours(lock_path)
    finally:
        os.close(fd)  # closing releases flock/byte-range locks, even if Python cleanup fails
