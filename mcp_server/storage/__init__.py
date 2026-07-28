"""
╭─╴ STORAGE PACKAGE ╶────────────────────────────────────────────╮
│                                                                │
│   Persistent-store adapters (currently ChromaDB) split out of  │
│   server.py in the A2.1 refactor.                              │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``
"""

from .chroma import ChromaVectorStore, _enable_wal_mode

__all__ = ["ChromaVectorStore", "_enable_wal_mode"]
