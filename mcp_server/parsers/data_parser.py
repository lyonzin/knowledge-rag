"""
╭─╴ KNOWLEDGE-RAG DATA PARSER ╶──────────────────────────────────╮
│                                                                │
│   Structured-data parsers: JSON, XML, CSV.                     │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Date    ·  2026-07-27

Grouped in one module because they share a shape (single suffix each,
stdlib-only, no optional deps) and because it keeps the parsers/
directory browseable. Each concrete parser stays a separate class so a
third-party plugin can swap one out (via ``register`` + matching
``display_name``) without disturbing the others.

JSON goes through :func:`json.loads` for validation and structural
metadata — indexed content is the pretty-printed round trip, so the
embedding sees consistent whitespace regardless of how the source
file was formatted. XML metadata surfaces the root element and any
namespace declarations, which is enough for retrieval targeting (e.g.
"where is my SAML metadata?") without paying for a full XML parse. CSV
reads through :mod:`csv` and lays every row out as ``a | b | c`` so
long files still tokenize sensibly.
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .base import ParserResult
from .registry import register_parser

__all__ = ["JsonParser", "XmlParser", "CsvParser"]


@register_parser
class JsonParser:
    """Parse ``.json`` files, pretty-printing on successful load."""

    extensions: frozenset[str] = frozenset({".json"})
    display_name: str = "json"
    optional_deps: tuple[str, ...] = ()

    def can_parse(self, path: Path) -> bool:
        """Return ``True`` for ``.json`` files.

        Args:
            path: Candidate file path.

        Returns:
            bool: Membership test against :attr:`extensions`.
        """
        return path.suffix.lower() in self.extensions

    def parse(self, path: Path) -> ParserResult:
        """Read JSON, capturing structural metadata when valid.

        Invalid JSON is *not* an error — the raw text is indexed as-is
        with ``is_valid_json=False`` so grep-style retrieval still works
        on malformed exports.

        Args:
            path: Existing ``.json`` file.

        Returns:
            ParserResult: ``(content, metadata)`` where content is
            ``json.dumps(data, indent=2)`` on success or the raw text on
            parse failure. Metadata carries ``type``, ``title``,
            ``file_size``, ``modified``, ``is_valid_json``, plus
            ``keys``/``structure`` for objects or ``length``/``structure``
            for arrays.
        """
        raw_content = path.read_text(encoding="utf-8", errors="ignore")
        metadata: Dict[str, Any] = {
            "type": "json",
            "title": path.stem,
            "file_size": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }

        try:
            data = json.loads(raw_content)
            metadata["is_valid_json"] = True

            if isinstance(data, dict):
                metadata["keys"] = list(data.keys())[:20]
                metadata["structure"] = "object"
            elif isinstance(data, list):
                metadata["length"] = len(data)
                metadata["structure"] = "array"

            # Pretty-print for better indexing
            content = json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            metadata["is_valid_json"] = False
            content = raw_content

        return content, metadata


@register_parser
class XmlParser:
    """Parse ``.xml`` files, extracting root element and namespaces."""

    extensions: frozenset[str] = frozenset({".xml"})
    display_name: str = "xml"
    optional_deps: tuple[str, ...] = ()

    def can_parse(self, path: Path) -> bool:
        """Return ``True`` for ``.xml`` files.

        Args:
            path: Candidate file path.

        Returns:
            bool: Membership test against :attr:`extensions`.
        """
        return path.suffix.lower() in self.extensions

    def parse(self, path: Path) -> ParserResult:
        """Extract raw XML plus root-element and namespace metadata.

        A full DOM parse would be more accurate but also 10x slower on
        the multi-MB SAML/SIEM configs users routinely index. The regex
        approach is intentional: cheap, robust to malformed docs, good
        enough for retrieval routing.

        Args:
            path: Existing ``.xml`` file.

        Returns:
            ParserResult: ``(content, metadata)`` where metadata carries
            ``type``, ``title``, ``file_size``, ``modified``,
            ``root_element`` (or ``None``), and ``namespaces`` — a list of
            ``{prefix, uri}`` dicts (``prefix='default'`` for
            ``xmlns=`` declarations).
        """
        content = path.read_text(encoding="utf-8", errors="ignore")
        metadata: Dict[str, Any] = {
            "type": "xml",
            "title": path.stem,
            "file_size": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            "root_element": None,
            "namespaces": [],
        }

        # Extract root element (skip <?xml ...?> declaration and comments)
        for match in re.finditer(r"<(\w[\w\-.:]*)[\s>]", content):
            tag = match.group(1)
            if tag.lower() != "xml":
                metadata["root_element"] = tag
                break

        # Extract namespace declarations
        ns_matches = re.findall(r'xmlns(?::(\w+))?\s*=\s*["\']([^"\']+)["\']', content)
        metadata["namespaces"] = [{"prefix": prefix or "default", "uri": uri} for prefix, uri in ns_matches]

        return content, metadata


@register_parser
class CsvParser:
    """Parse ``.csv`` files as ``|``-joined text tables."""

    extensions: frozenset[str] = frozenset({".csv"})
    display_name: str = "csv"
    optional_deps: tuple[str, ...] = ()

    def can_parse(self, path: Path) -> bool:
        """Return ``True`` for ``.csv`` files.

        Args:
            path: Candidate file path.

        Returns:
            bool: Membership test against :attr:`extensions`.
        """
        return path.suffix.lower() in self.extensions

    def parse(self, path: Path) -> ParserResult:
        """Flatten a CSV into a ``|``-joined text table.

        Args:
            path: Existing ``.csv`` file.

        Returns:
            ParserResult: ``(content, metadata)`` where content is the
            rows joined by ``\\n`` with cells separated by `` | ``, and
            metadata carries ``type``, ``title``, ``file_size``,
            ``modified``, ``rows`` count, and ``columns`` count (from
            the first row).
        """
        raw = path.read_text(encoding="utf-8", errors="ignore")
        metadata: Dict[str, Any] = {
            "type": "csv",
            "title": path.stem,
            "file_size": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }

        parts = []
        reader = csv.reader(io.StringIO(raw))
        rows = list(reader)
        metadata["rows"] = len(rows)
        metadata["columns"] = len(rows[0]) if rows else 0

        for row in rows:
            parts.append(" | ".join(row))

        content = "\n".join(parts)
        return content, metadata
