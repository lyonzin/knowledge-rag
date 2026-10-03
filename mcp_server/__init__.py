"""Knowledge RAG MCP Server - Local Retrieval-Augmented Generation System"""

__version__ = "4.9.3"
__author__ = "Ailton Rocha (Lyon.)"

from .config import Config  # noqa: E402
from .ingestion import Document, DocumentParser  # noqa: E402

__all__ = ["Config", "DocumentParser", "Document"]
