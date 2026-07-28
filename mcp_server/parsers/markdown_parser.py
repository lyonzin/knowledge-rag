"""
╭─╴ KNOWLEDGE-RAG MARKDOWN PARSER ╶──────────────────────────────╮
│                                                                │
│   Extracts headers, frontmatter, and body text from            │
│   Markdown source files.                                       │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Date    ·  2026-07-27

The parser exposes header hierarchy so downstream tooling can build a
table of contents, detects YAML frontmatter (and strips it from the
indexed content so tags/metadata do not leak into embeddings), and marks
whether the document contains any fenced code blocks. Chunking is
handled elsewhere by the section-aware chunker in
:mod:`mcp_server.parsers.chunking`.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from .base import ParserResult
from .registry import register_parser

__all__ = ["MarkdownParser"]


@register_parser
class MarkdownParser:
    """Parse ``.md`` files into text plus header metadata."""

    extensions: frozenset[str] = frozenset({".md"})
    display_name: str = "markdown"
    optional_deps: tuple[str, ...] = ()

    def can_parse(self, path: Path) -> bool:
        """Return ``True`` when ``path`` carries a supported suffix.

        Args:
            path: Candidate file path.

        Returns:
            bool: ``True`` for known Markdown suffixes.
        """
        return path.suffix.lower() in self.extensions

    def parse(self, path: Path) -> ParserResult:
        """Extract Markdown content and header metadata.

        Args:
            path: Existing ``.md`` file.

        Returns:
            ParserResult: ``(content, metadata)`` where metadata contains
            ``type``, ``headers`` (list of ``{level, title}``), the
            resolved ``title``, ``has_code_blocks`` flag, ``file_size``,
            ``modified`` timestamp, and — when present — ``has_frontmatter``.
        """
        content = path.read_text(encoding="utf-8", errors="ignore")
        metadata: Dict[str, Any] = {
            "type": "markdown",
            "headers": [],
            "has_code_blocks": "```" in content,
            "file_size": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }

        # Extract headers hierarchy
        header_pattern = r"^(#{1,6})\s+(.+)$"
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            metadata["headers"].append({"level": level, "title": title})

        # Extract title from first H1 or filename
        h1_headers = [h for h in metadata["headers"] if h["level"] == 1]
        if h1_headers:
            metadata["title"] = h1_headers[0]["title"]
        else:
            metadata["title"] = path.stem

        # Extract frontmatter if present (YAML between ---)
        frontmatter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
        if frontmatter_match:
            metadata["has_frontmatter"] = True
            # Remove frontmatter from content for cleaner indexing
            content = content[frontmatter_match.end() :]

        return content, metadata


# Explicit re-export target so ``from .markdown_parser import *`` and the
# ``@register_parser`` side effect stay in sync when tests introspect the
# module.
_ParsedTuple = Tuple[str, Dict[str, Any]]  # noqa: N816 — mirror ParserResult for readers
