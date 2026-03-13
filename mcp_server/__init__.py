"""Knowledge RAG MCP Server - Local Retrieval-Augmented Generation System"""

__version__ = "1.0.1"
__author__ = "Lyon."

from .config import Config
from .ingestion import DocumentParser, Document

__all__ = ["Config", "DocumentParser", "Document"]
