"""
╭─╴ KNOWLEDGE-RAG NOTEBOOK PARSER ╶──────────────────────────────╮
│                                                                │
│   Jupyter Notebook (``.ipynb``) reader.                        │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Date    ·  2026-07-27

Design invariant: **only cell sources are indexed**, never outputs.
Cell outputs routinely carry inline base64 images (matplotlib figures,
Pillow renders) that can inflate a single notebook into megabytes of
useless embedding fuel. Skipping outputs also side-steps
security-relevant PII leakage (query results, credentials printed
during debugging).

Code cells are wrapped in triple-backtick fences with a ``python``
language hint so that the downstream Markdown-aware chunker can keep
them intact.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .base import ParserResult
from .registry import register_parser

__all__ = ["IpynbParser"]


@register_parser
class IpynbParser:
    """Parse Jupyter Notebook ``.ipynb`` files."""

    extensions: frozenset[str] = frozenset({".ipynb"})
    display_name: str = "ipynb"
    optional_deps: tuple[str, ...] = ()

    def can_parse(self, path: Path) -> bool:
        """Return ``True`` for ``.ipynb`` files.

        Args:
            path: Candidate file path.

        Returns:
            bool: Membership test against :attr:`extensions`.
        """
        return path.suffix.lower() in self.extensions

    def parse(self, path: Path) -> ParserResult:
        """Extract markdown and code cell sources; drop outputs.

        Malformed notebook JSON is not fatal: the raw text is returned
        with ``is_valid_json=False`` so retrieval still works on partly
        corrupted files. Empty cells (either type) are skipped so they
        do not create zero-content chunks.

        Args:
            path: Existing ``.ipynb`` file.

        Returns:
            ParserResult: ``(content, metadata)`` where content joins
            every non-empty cell source (code cells fenced in
            ```` ```python ```` blocks) with blank lines, and metadata
            carries ``type='jupyter_notebook'``, ``title``,
            ``file_size``, ``modified``, ``is_valid_json``, ``nbformat``,
            ``kernel`` display name, ``cells`` count, and
            ``code_cells``/``markdown_cells`` counts.
        """
        raw = path.read_text(encoding="utf-8", errors="ignore")
        metadata: Dict[str, Any] = {
            "type": "jupyter_notebook",
            "title": path.stem,
            "file_size": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }

        try:
            nb = json.loads(raw)
        except json.JSONDecodeError:
            metadata["is_valid_json"] = False
            return raw, metadata

        metadata["is_valid_json"] = True
        metadata["nbformat"] = nb.get("nbformat", 0)
        kernel = nb.get("metadata", {}).get("kernelspec", {})
        metadata["kernel"] = kernel.get("display_name", kernel.get("name", "unknown"))

        cells = nb.get("cells", [])
        metadata["cells"] = len(cells)
        code_cells = 0
        markdown_cells = 0

        parts = []
        for cell in cells:
            cell_type = cell.get("cell_type", "")
            source = cell.get("source", "")

            if isinstance(source, list):
                source = "".join(source)

            if not source or not source.strip():
                continue

            if cell_type == "markdown":
                parts.append(source)
                markdown_cells += 1
            elif cell_type == "code":
                parts.append(f"```python\n{source}\n```")
                code_cells += 1

        metadata["code_cells"] = code_cells
        metadata["markdown_cells"] = markdown_cells

        content = "\n\n".join(parts)
        return content, metadata
