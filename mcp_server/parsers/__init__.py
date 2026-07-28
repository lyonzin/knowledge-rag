"""
╭─╴ KNOWLEDGE-RAG PARSERS ╶──────────────────────────────────────╮
│                                                                │
│   Pluggable document parser subpackage.                        │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Date    ·  2026-07-27

Importing this package registers every built-in parser as a side
effect. Third-party plugins can be discovered on demand via
:func:`mcp_server.parsers.registry.load_third_party`.

The public surface is deliberately narrow:

* :class:`~.base.Parser`, :class:`~.base.Document`, :class:`~.base.Chunk`
  — value objects and the plugin contract.
* :mod:`mcp_server.parsers.registry` — the dispatch table.
* :mod:`mcp_server.parsers.chunking` — shared chunking helpers.

Cross-cutting concerns (prompt-injection sanitization, provenance
markers, category detection, keyword extraction) stay in
:mod:`mcp_server.ingestion`, so a plugin author can drop in a new parser
without having to reproduce any of the security or retrieval invariants.
"""

from __future__ import annotations

# Order matters: default parsers register first so a plugin that
# targets the same suffix wins on tie (see registry.get_parser_for).
# isort: off
from . import markdown_parser  # noqa: F401 — autoregister
from . import text_parser  # noqa: F401 — autoregister
from . import pdf_parser  # noqa: F401 — autoregister
from . import office_parser  # noqa: F401 — autoregister
from . import data_parser  # noqa: F401 — autoregister
from . import notebook_parser  # noqa: F401 — autoregister
# code_parser (tree-sitter AST-boundary chunking), late_chunker (Jina long-context),
# tree_sitter_chunker, and confidence classifier are shipped in a follow-up PR
# (advanced chunking + confidence labels).

# isort: on
from .base import Chunk, Document, Parser, ParserResult
from .registry import (
    build_dispatch_view,
    clear,
    get_parser_for,
    get_parsers,
    load_third_party,
    parse_content,
    register,
    register_parser,
    unregister,
)

__all__ = [
    "Chunk",
    "Document",
    "Parser",
    "ParserResult",
    "build_dispatch_view",
    "clear",
    "get_parser_for",
    "get_parsers",
    "load_third_party",
    "parse_content",
    "register",
    "register_parser",
    "unregister",
]
