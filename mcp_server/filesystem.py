"""Windows extended-length paths for I/O, separate from public document identity."""

from __future__ import annotations

import ntpath
import os
import sys
from collections.abc import Iterator
from pathlib import Path

_MAX_PATH = 260
_EXTENDED_PREFIX = "\\\\?\\"
_EXTENDED_UNC_PREFIX = "\\\\?\\UNC\\"


def _extend_windows_path(absolute: str) -> str:
    """Prefix a normalized absolute drive/UNC path without changing its target."""
    normalized = ntpath.normpath(absolute)
    if normalized.startswith(_EXTENDED_PREFIX):
        return normalized
    if normalized.startswith("\\\\"):
        return _EXTENDED_UNC_PREFIX + normalized[2:]
    return _EXTENDED_PREFIX + normalized


def io_path(path: Path, *, extend_directory: bool = False) -> Path:
    """Adapt trusted paths only at I/O boundaries; this is not a containment check.

    Keep short paths and POSIX paths unchanged. Directory traversal opts in
    even for a short root because descendant paths can exceed MAX_PATH.
    Resolve dot components and links before adding the Win32 prefix, which
    otherwise disables their usual interpretation.
    """
    if sys.platform != "win32":
        return path
    absolute = os.path.abspath(path)
    if not extend_directory and len(absolute) < _MAX_PATH:
        return path
    return Path(_extend_windows_path(str(resolve_path(path))))


def _public_windows_path(path: str) -> str:
    """Remove only the I/O namespace prefix from a path returned by os.walk."""
    if path.casefold().startswith(_EXTENDED_UNC_PREFIX.casefold()):
        return "\\\\" + path[len(_EXTENDED_UNC_PREFIX) :]
    if path.startswith(_EXTENDED_PREFIX):
        return path[len(_EXTENDED_PREFIX) :]
    return path


def resolve_path(path: Path) -> Path:
    """Resolve long Windows links without leaking the I/O prefix to callers.

    A normal Win32 resolve can stop at MAX_PATH and leave a deep symlink
    unresolved. Containment must check its real target before any read.
    """
    if sys.platform != "win32":
        return path.resolve()
    absolute = os.path.abspath(path)
    readable = Path(_extend_windows_path(absolute)) if len(absolute) >= _MAX_PATH else path
    resolved = readable.resolve()
    return Path(_public_windows_path(str(resolved)))


def walk_document_paths(directory: Path) -> Iterator[tuple[str, list[str], list[str]]]:
    """Walk long directories while retaining normal paths and in-place pruning."""
    for root, directories, files in os.walk(io_path(directory, extend_directory=True), followlinks=True):
        public_root = _public_windows_path(root) if sys.platform == "win32" else root
        yield public_root, directories, files
