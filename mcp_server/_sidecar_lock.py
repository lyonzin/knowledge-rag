"""Nonblocking kernel locks on persistent sidecars. No config or storage access on import."""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path
from typing import Optional

_BUSY_ERRNOS = (errno.EACCES, errno.EAGAIN, errno.EDEADLK)


def lock_sidecar(path: Path) -> Optional[int]:
    """Return a locked descriptor, or None on contention; never unlink the sidecar. By Lyon."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    except PermissionError:
        if sys.platform == "win32":
            raise
        else:
            # Another UID's 0644 sidecar: local flock can lock the same inode read-only.
            fd = os.open(path, os.O_RDONLY)
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # a fresh descriptor starts at byte zero
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in _BUSY_ERRNOS:
            return None
        raise
    return fd
