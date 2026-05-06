"""Single-instance guard for the MCP server process."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import config

LOCK_FILENAME = "knowledge-rag.lock"
ALREADY_RUNNING_EXIT_CODE = 75


class AlreadyRunningError(RuntimeError):
    """Raised when another knowledge-rag server instance already holds the lock."""


def _pid_is_running(pid: int) -> bool:
    """Return True if a process with PID appears to be alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_lock_pid(lock_path: Path) -> int | None:
    try:
        raw = lock_path.read_text(encoding="utf-8").strip().splitlines()[0]
        return int(raw)
    except (IndexError, OSError, ValueError):
        return None


def _lock_path() -> Path:
    return config.data_dir / LOCK_FILENAME


@contextmanager
def single_instance_lock() -> Iterator[Path]:
    """Hold an exclusive process-level lock for one MCP server instance."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path()

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            pid = _read_lock_pid(lock_path)
            if pid is not None and _pid_is_running(pid):
                raise AlreadyRunningError(
                    f"knowledge-rag MCP server is already running (pid {pid}). Refusing to start a second instance."
                )

            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(
                    f"[ERROR] Failed to clear stale lock {lock_path}: {exc}",
                    file=sys.stderr,
                )
                raise AlreadyRunningError(f"Failed to clear stale lock {lock_path}: {exc}") from exc
            continue

        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n")
        break

    try:
        yield lock_path
    finally:
        if _read_lock_pid(lock_path) == os.getpid():
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
