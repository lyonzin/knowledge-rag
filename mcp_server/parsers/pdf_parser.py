"""
╭─╴ KNOWLEDGE-RAG PDF PARSER ╶───────────────────────────────────╮
│                                                                │
│   PDF text extraction via PyMuPDF (``fitz``).                  │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Date    ·  2026-07-27

PyMuPDF is loaded lazily — the import lands only when a ``.pdf`` shows
up in the corpus. That keeps ``pip install knowledge-rag`` (without the
PDF extra) from tripping on a missing native dep. When the extra is not
installed the parser raises :class:`ImportError` at parse time with the
exact ``pip`` incantation the operator needs.

No markdown conversion here: we emit plain text page-by-page with a
``[Page N]`` prefix so downstream retrieval can cite pages without
trying to reconstruct the original layout.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .base import ParserResult
from .registry import register_parser

__all__ = ["PdfParser"]


try:
    import fitz  # type: ignore[import-not-found]  # PyMuPDF

    _HAS_PYMUPDF = True
except ImportError:  # pragma: no cover - covered indirectly by import guard test
    _HAS_PYMUPDF = False


@register_parser
class PdfParser:
    """Parse ``.pdf`` files using PyMuPDF."""

    extensions: frozenset[str] = frozenset({".pdf"})
    display_name: str = "pdf"
    optional_deps: tuple[str, ...] = ("pymupdf",)

    def can_parse(self, path: Path) -> bool:
        """Return ``True`` for ``.pdf`` files.

        Args:
            path: Candidate file path.

        Returns:
            bool: Membership test against :attr:`extensions`. Note that
            ``can_parse`` returning ``True`` does not guarantee the
            optional dep is installed — :meth:`parse` raises
            :class:`ImportError` in that case, matching the historical
            behaviour of the monolithic parser.
        """
        return path.suffix.lower() in self.extensions

    def parse(self, path: Path) -> ParserResult:
        """Extract text page-by-page from ``path``.

        Also flags whether the source PDF carried an extractable text
        layer at all: a scanned PDF (image-only) yields empty
        ``page.get_text()`` on every page, and the M4.7 confidence
        classifier uses that marker to downgrade the resulting chunks
        to :attr:`~.confidence.SourceConfidence.UNVERIFIED` — a caller
        should not trust text we could not actually extract.

        Args:
            path: Existing ``.pdf`` file.

        Returns:
            ParserResult: ``(content, metadata)`` where content is the
            per-page text joined by blank lines with ``[Page N]``
            markers, and metadata carries ``type``, ``pages`` count,
            ``title`` (from PDF metadata or filename), ``author``,
            ``file_size``, ``modified`` timestamp, and
            ``pdf_has_text_layer`` (``True`` when at least one page
            yielded non-whitespace text).

        Raises:
            ImportError: When PyMuPDF is not installed.
        """
        if not _HAS_PYMUPDF:
            raise ImportError("PyMuPDF (fitz) not installed. Install with: pip install pymupdf")

        metadata: Dict[str, Any] = {
            "type": "pdf",
            "pages": 0,
            "file_size": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            # Pessimistic default: assume no text layer until at least
            # one page proves otherwise. A scanned-only PDF then ends
            # up with ``pdf_has_text_layer=False`` and every chunk gets
            # tagged UNVERIFIED by the confidence classifier without
            # any extra bookkeeping here.
            "pdf_has_text_layer": False,
        }

        text_parts = []

        with fitz.open(path) as doc:
            metadata["pages"] = len(doc)
            metadata["title"] = doc.metadata.get("title", path.stem)
            metadata["author"] = doc.metadata.get("author", "")

            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    metadata["pdf_has_text_layer"] = True
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")

        content = "\n\n".join(text_parts)
        return content, metadata
