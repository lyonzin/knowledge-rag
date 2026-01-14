"""Knowledge RAG MCP Server - Local Retrieval-Augmented Generation System"""

__version__ = "1.0.0"
__author__ = "Ailton Rocha"

from .config import Config
from .ingestion import DocumentParser, Document

__all__ = ["Config", "DocumentParser", "Document"]
