"""
╭─╴ KNOWLEDGE-RAG PARSER REGISTRY ╶──────────────────────────────╮
│                                                                │
│   Extensible dispatch table for document parsers, with         │
│   fault-isolated resolution and entry-point discovery.         │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Date    ·  2026-07-27

Design
------
The registry keeps a single ordered list of :class:`~.base.Parser`
instances. Registration is push-based: parser modules import their
``@register_parser`` decorator when :mod:`mcp_server.parsers` is imported
and their instances are appended. Third-party parsers hook the same list
through the ``knowledge_rag.parsers`` entry-point group — see
:func:`load_third_party`.

Resolution is deliberately fault-isolated: :func:`get_parser_for` swallows
exceptions raised by ``can_parse`` and continues with the next candidate,
so a broken third-party parser cannot brick built-in dispatch. The
default parsers' ``can_parse`` is a pure suffix test and never raises;
the guard exists solely for plugins the operator cannot audit.

Ordering matters for overlapping extensions: a plugin that registers
later gets priority over the built-ins for the same suffix. That is the
intended plugin-override mechanism — same reason the built-in
autoloader runs *before* :func:`load_third_party` in
:mod:`mcp_server.parsers.__init__`.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from pathlib import Path
from typing import Callable, List, Optional, Type

from .base import Parser, ParserResult

__all__ = [
    "register",
    "register_parser",
    "unregister",
    "get_parsers",
    "get_parser_for",
    "parse_content",
    "load_third_party",
    "clear",
]

log = logging.getLogger(__name__)

_PARSERS: List[Parser] = []


def register(parser: Parser) -> None:
    """Append ``parser`` to the dispatch list.

    A parser with the same :attr:`~.base.Parser.display_name` as an
    already-registered one is replaced in place — this makes it easy to
    swap the built-in ``markdown`` parser for a customised variant from a
    plugin without also having to call :func:`unregister` first.

    Args:
        parser: Parser instance to register. Must satisfy the
            :class:`~.base.Parser` protocol.
    """
    for i, existing in enumerate(_PARSERS):
        if existing.display_name == parser.display_name:
            _PARSERS[i] = parser
            return
    _PARSERS.append(parser)


def register_parser(cls: Type[Parser]) -> Type[Parser]:
    """Class decorator that instantiates ``cls`` and registers it.

    Every built-in parser module uses this to opt into dispatch at import
    time. The class itself is returned unchanged so the decorator is
    transparent to consumers that import the class directly (typically
    for testing).

    Args:
        cls: Parser class with a no-argument constructor.

    Returns:
        type: ``cls`` unchanged.

    Example:
        >>> from mcp_server.parsers.registry import register_parser
        >>> @register_parser  # doctest: +SKIP
        ... class MyParser: ...
    """
    register(cls())
    return cls


def unregister(display_name: str) -> None:
    """Remove every parser whose :attr:`display_name` matches.

    Used by tests to isolate registry state and by the plugin machinery
    to hot-swap parsers.

    Args:
        display_name: Value of :attr:`~.base.Parser.display_name` to drop.
    """
    global _PARSERS
    _PARSERS = [p for p in _PARSERS if p.display_name != display_name]


def clear() -> None:
    """Drop every registered parser.

    Only intended for isolated test fixtures — production callers must
    re-import :mod:`mcp_server.parsers` (or call
    :func:`_reload_defaults`) after clearing, or resolution will start
    returning ``None`` for everything.
    """
    _PARSERS.clear()


def get_parsers() -> List[Parser]:
    """Return a snapshot copy of the registered parsers, in dispatch order.

    Returns:
        list[Parser]: Fresh list — mutations by the caller do not affect
        the registry.
    """
    return list(_PARSERS)


def get_parser_for(path: Path) -> Optional[Parser]:
    """Resolve which registered parser should handle ``path``.

    Iteration is fault-isolated: a parser whose ``can_parse`` raises is
    logged and skipped, so one broken plugin cannot poison the whole
    dispatch pipeline.

    Args:
        path: File whose format is being resolved. Need not exist yet —
            ``can_parse`` sees only the suffix by default.

    Returns:
        Parser | None: First registered parser that claims ``path``, or
        ``None`` when nothing matches.
    """
    for parser in _PARSERS:
        try:
            if parser.can_parse(path):
                return parser
        except Exception as exc:  # noqa: BLE001 — fault isolation is the point
            log.warning(
                "Parser %s.can_parse raised for %s: %s",
                getattr(parser, "display_name", type(parser).__name__),
                path,
                exc,
            )
            continue
    return None


def parse_content(path: Path) -> Optional[ParserResult]:
    """Dispatch ``path`` to the appropriate parser and return raw output.

    Cross-cutting concerns (sanitization, chunking, keyword extraction)
    are the caller's job — see :meth:`mcp_server.ingestion.DocumentParser.parse_file`
    for the full pipeline.

    Args:
        path: File to parse.

    Returns:
        ParserResult | None: ``(content, metadata)`` on success, ``None``
        when no parser claims the suffix. Failures inside a matched
        parser propagate so the caller can decide whether to skip the
        file or abort a batch.
    """
    parser = get_parser_for(path)
    if parser is None:
        return None
    return parser.parse(path)


def load_third_party() -> None:
    """Discover parsers advertised by other installed packages.

    Plugins ship an entry point under the ``knowledge_rag.parsers`` group
    whose target is any module that self-registers on import (typically
    via :func:`register_parser`). Failure of one plugin never aborts the
    whole scan.

    Called opportunistically — the built-in autoloader does not depend on
    it, and there is no requirement that any plugins be installed.
    """
    try:
        eps = entry_points(group="knowledge_rag.parsers")
    except Exception as exc:  # noqa: BLE001 — very old Python or broken metadata
        log.warning("Third-party parser discovery failed: %s", exc)
        return

    for ep in eps:
        try:
            ep.load()  # module registers via decorator on import
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load parser plugin %s: %s", ep.name, exc)


# ============================================================================
# Backward-compatibility helper for DocumentParser._parsers
# ============================================================================


def build_dispatch_view() -> dict[str, Callable[[Path], ParserResult]]:
    """Return an extension → ``parse`` callable snapshot.

    Preserves the historical shape of ``DocumentParser._parsers`` — a
    dict keyed by file suffix — that tests inspect to assert a format is
    supported. Values are the parser's bound :meth:`~.base.Parser.parse`
    method so calling ``view[".md"](path)`` yields the same
    ``(content, metadata)`` tuple as before.

    Returns:
        dict[str, callable]: Fresh snapshot — mutations do not affect
        the registry. Later parsers win on suffix collisions, matching
        :func:`get_parser_for` resolution order.
    """
    view: dict[str, Callable[[Path], ParserResult]] = {}
    for parser in _PARSERS:
        for ext in parser.extensions:
            view[ext] = parser.parse
    return view
