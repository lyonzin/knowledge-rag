"""Knowledge RAG MCP Server - Local Retrieval-Augmented Generation System"""

import sys  # noqa: I001

# MCP stdio uses stdout for JSON-RPC. print() on stdout corrupts the stream.
# Save the real stdout for MCP, redirect print() to stderr during init.
# server.py main() restores _original_stdout before mcp.run().
_original_stdout = sys.stdout
sys.stdout = sys.stderr

__version__ = "4.3.1"
__author__ = "Ailton Rocha (Lyon.)"

from .config import Config  # noqa: E402
from .ingestion import Document, DocumentParser  # noqa: E402

__all__ = ["Config", "DocumentParser", "Document"]
