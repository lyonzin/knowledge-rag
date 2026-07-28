"""
╭─╴ KNOWLEDGE-RAG TEXT PARSER ╶──────────────────────────────────╮
│                                                                │
│   Plain-text parser: default fallback for ``.txt`` files.      │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Date    ·  2026-07-27

Simplest parser in the arsenal. Reads the file with ``errors="ignore"``
so a corrupted byte in the middle of a wall of text does not abort a
batch, and hands the content back untouched — the chunker downstream is
responsible for splitting.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .base import ParserResult
from .registry import register_parser

__all__ = ["TextParser"]


@register_parser
class TextParser:
    """Parse plain-text files."""

    extensions: frozenset[str] = frozenset({".txt"})
    display_name: str = "text"
    optional_deps: tuple[str, ...] = ()

    def can_parse(self, path: Path) -> bool:
        """Return ``True`` for ``.txt`` files.

        Args:
            path: Candidate file path.

        Returns:
            bool: Membership test against :attr:`extensions`.
        """
        return path.suffix.lower() in self.extensions

    def parse(self, path: Path) -> ParserResult:
        """Read a plain-text file and record basic size/line metadata.

        Args:
            path: Existing ``.txt`` file.

        Returns:
            ParserResult: ``(content, metadata)`` with ``type``, ``title``,
            ``file_size``, ``modified`` timestamp, and ``line_count``.
        """
        content = path.read_text(encoding="utf-8", errors="ignore")
        metadata: Dict[str, Any] = {
            "type": "text",
            "title": path.stem,
            "file_size": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            "line_count": content.count("\n") + 1,
        }
        return content, metadata
