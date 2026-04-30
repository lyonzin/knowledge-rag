"""Backward-compatible guarded console entry point for knowledge-rag."""

from __future__ import annotations

from .server import main


def guarded_main() -> None:
    """Run the MCP server; server.main performs startup preflight."""
    main()
