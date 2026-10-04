"""Validated, corpus-relative folder scopes shared by every retrieval backend."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

from .filesystem import io_path

ScopeKey = tuple[tuple[str, ...], tuple[str, ...]]


def _folder(root: Path, value: str) -> Path:
    """Resolve one relative directory, rejecting escapes and Windows aliases."""
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError("Folder filters require non-empty relative paths (at most 4096 characters)")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if PureWindowsPath(value).anchor or path.is_absolute() or ".." in path.parts:
        raise ValueError("Folder filters must be relative to documents_dir and cannot contain '..'")
    if ":" in value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Folder filters cannot contain device names, streams or control characters")
    if os.name == "nt" and any(
        part.endswith((".", " ")) or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part)
        for part in path.parts
    ):
        raise ValueError("Folder filters cannot contain ambiguous Windows path components")
    try:
        resolved = root.joinpath(*path.parts).resolve()
        resolved.relative_to(root)
        if io_path(resolved).exists() and not io_path(resolved).is_dir():
            raise ValueError("Folder filter refers to a file, not a directory")
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("Folder filter must resolve to a directory within documents_dir") from error
    return resolved


def _folders(root: Path, values: Sequence[str] | None) -> tuple[Path, ...]:
    """Bound filter size and canonicalize order/duplicates for cache identity."""
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or len(values) > 32:
        raise ValueError("Folder filters must be a list of at most 32 relative directories")
    return tuple(sorted({_folder(root, value) for value in values}, key=str))


@dataclass(frozen=True)
class FolderScope:
    """Recursive include union minus exclude union; excludes always win."""

    root: Path
    include: tuple[Path, ...]
    exclude: tuple[Path, ...]

    @classmethod
    def create(cls, root: Path, include: Sequence[str] | None, exclude: Sequence[str] | None) -> FolderScope:
        """Validate caller paths even on a query-cache hit."""
        resolved = root.resolve()
        return cls(resolved, _folders(resolved, include), _folders(resolved, exclude))

    @property
    def active(self) -> bool:
        """Whether source selection is required."""
        return bool(self.include or self.exclude)

    @property
    def key(self) -> ScopeKey:
        """Return stable platform-aware relative paths, without private roots."""
        return tuple(os.path.normcase(str(path.relative_to(self.root))) for path in self.include), tuple(
            os.path.normcase(str(path.relative_to(self.root))) for path in self.exclude
        )

    def _contains(self, path: Path) -> bool:
        """Match full path components so ``foo`` cannot match ``foobar``."""
        return (
            path.is_relative_to(self.root)
            and (not self.include or any(path.is_relative_to(folder) for folder in self.include))
            and not any(path.is_relative_to(folder) for folder in self.exclude)
        )

    def document_ids(self, indexed: Mapping[str, Mapping[str, Any]]) -> list[str]:
        """Select stored identities, checking current links before admitting sources."""
        selected = []
        for identity, metadata in list(indexed.items()):
            source = metadata.get("source")
            if not isinstance(source, str) or not source:
                continue
            path = Path(source)
            if not path.is_absolute():
                path = self.root / path
            if not self._contains(path):
                continue
            try:
                if self._contains(path.resolve()):
                    selected.append(identity)
            except (OSError, RuntimeError):
                continue
        return selected
