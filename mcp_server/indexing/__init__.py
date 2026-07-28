"""
╭─╴ INDEXING PACKAGE ╶───────────────────────────────────────────╮
│                                                                │
│   Index building + filesystem watcher primitives split out of  │
│   server.py in the A2.1 refactor.                              │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``
"""

from .bm25_index import BM25Index
from .watcher import DocumentWatcher

__all__ = ["BM25Index", "DocumentWatcher"]
