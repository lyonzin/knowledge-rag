"""
╭─╴ MCP TOOLS PACKAGE ╶──────────────────────────────────────────╮
│                                                                │
│   ``@mcp.tool()`` handlers + the FastMCP singleton, split out  │
│   of server.py in the A2.1 refactor.                           │
│                                                                │
│   Named ``mcp_tools`` rather than ``mcp`` to avoid shadowing   │
│   the third-party ``mcp.server.fastmcp`` package during        │
│   absolute imports elsewhere in the code base.                 │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``
"""

from .instance import mcp
from .tools import (
    add_document,
    add_from_url,
    evaluate_retrieval,
    get_document,
    get_index_stats,
    get_reindex_status,
    list_categories,
    list_dashboard,
    list_documents,
    reindex_documents,
    remove_document,
    search_global,
    search_knowledge,
    search_similar,
    update_document,
)

__all__ = [
    "add_document",
    "add_from_url",
    "evaluate_retrieval",
    "get_document",
    "get_index_stats",
    "get_reindex_status",
    "list_categories",
    "list_dashboard",
    "list_documents",
    "mcp",
    "reindex_documents",
    "remove_document",
    "search_global",
    "search_knowledge",
    "search_similar",
    "update_document",
]
