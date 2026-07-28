"""
╭─╴ KNOWLEDGE-RAG PARSER BASE ╶──────────────────────────────────╮
│                                                                │
│   Value objects and Parser protocol for the pluggable          │
│   ingestion pipeline.                                          │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Date    ·  2026-07-27

Design
------
``Document`` and ``Chunk`` live here — not in ``ingestion`` — so that
individual parser modules can import them without pulling the whole
ingestion facade (and its heavy optional deps) back into scope. The
public re-export ``from mcp_server.ingestion import Document, Chunk``
still works, keeping the historical import path intact for downstream
consumers.

The ``Parser`` protocol is deliberately narrow: extract raw
``(content, metadata)`` from a file. Every cross-cutting concern
(prompt-injection sanitization, provenance markers, categorization,
keyword extraction, chunking) is the facade's job. Keeping parsers
free of those responsibilities is what makes them safe to plug in from
third-party packages without touching the security invariants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from ..config import config

__all__ = [
    "Chunk",
    "Document",
    "Parser",
    "ParserResult",
]

#: Type alias for the tuple every :class:`Parser` returns: raw text content
#: followed by a mutable metadata dict the facade may enrich further.
ParserResult = Tuple[str, Dict[str, Any]]


@dataclass
class Chunk:
    """A chunk of text carved out of a parent document.

    Attributes:
        content: Chunk text ready to embed.
        index: Zero-based chunk position within the parent document.
        start_char: Inclusive character offset in the original content.
        end_char: Exclusive character offset in the original content.
        metadata: Free-form per-chunk metadata (title, section header,
            provenance flags…).
        embedding: Optional pre-computed dense vector. Populated by chunkers
            that produce embeddings inline — currently only the Late
            Chunking pipeline (Jina 2024, R5.7). When set, the indexing
            layer skips its usual embedding step and stores the vector
            directly. ``None`` on every legacy chunker so existing
            downstream code paths are unaffected.
    """

    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class Document:
    """Fully parsed document ready to be indexed.

    Attributes:
        id: Stable 16-hex-char SHA-256 slice derived from path + mtime + size.
        content: Full extracted text (post-sanitization when external).
        source: Absolute path to the source file on disk.
        format: File suffix, lowercase, including the leading dot.
        category: Corpus category resolved from the file path.
        metadata: Format-specific and pipeline metadata.
        chunks: Chunks ready to embed. Populated by the pipeline, not the
            parser.
        keywords: Extracted technical keywords.
    """

    id: str
    content: str
    source: Path
    format: str
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[Chunk] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    @property
    def filename(self) -> str:
        """Return the source file name (basename)."""
        return self.source.name

    @property
    def relative_path(self) -> str:
        """Return the source path relative to ``config.documents_dir``.

        Falls back to the absolute string when the source lives outside the
        configured corpus (e.g. a fixture in a test's ``tmp_path``).
        """
        try:
            return str(self.source.relative_to(config.documents_dir))
        except ValueError:
            return str(self.source)


@runtime_checkable
class Parser(Protocol):
    """Contract every registered parser fulfils.

    A parser is a stateless extractor: given a path it knows how to handle,
    it returns raw text and format-specific metadata. It must not do any
    of the following — those belong to the facade in :mod:`mcp_server.ingestion`:

    * Prompt-injection neutralization.
    * Provenance marker detection.
    * Chunking.
    * Keyword extraction or category mapping.
    * ID generation.

    Attributes:
        extensions: Lowercase file suffixes (with leading dot) this parser
            claims. Used by :meth:`can_parse` and by the registry to build
            the dispatch view exposed on ``DocumentParser._parsers`` for
            backward compatibility.
        display_name: Human-readable identifier used in log lines and by
            :func:`mcp_server.parsers.registry.unregister`.
        optional_deps: Names of ``pip`` distributions the parser needs at
            runtime. Empty for parsers that rely on the standard library
            only. Purely informational — surfaced by
            ``list_categories`` / diagnostics — never used to short-circuit
            registration.
    """

    extensions: frozenset[str]
    display_name: str
    optional_deps: tuple[str, ...]

    def can_parse(self, path: Path) -> bool:
        """Return ``True`` when the parser should handle ``path``.

        Default matcher is a suffix membership test; parsers with more
        elaborate detection (magic bytes, content sniffing) may override it.

        Args:
            path: Candidate file. May or may not exist yet.

        Returns:
            bool: ``True`` when the suffix belongs to :attr:`extensions`.
        """
        ...

    def parse(self, path: Path) -> ParserResult:
        """Extract ``(content, metadata)`` from ``path``.

        Args:
            path: Existing file within the resolved documents directory.

        Returns:
            ParserResult: ``(content, metadata)`` tuple. ``content`` is a
            plain string; ``metadata`` is a mutable dict the pipeline may
            enrich further.

        Raises:
            ImportError: When an :attr:`optional_deps` package is missing.
            OSError: On unreadable files.
            Exception: Parser-specific failures — the registry catches these
                so one broken parser cannot poison a directory walk.
        """
        ...
