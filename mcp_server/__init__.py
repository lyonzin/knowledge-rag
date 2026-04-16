"""Knowledge RAG MCP Server - Local Retrieval-Augmented Generation System"""

__version__ = "3.3.2"
__author__ = "Ailton Rocha (Lyon.)"

import sys
import builtins

# MCP protocol uses stdout for communication. 
# Any accidental print() call in libraries will corrupt the protocol stream.
# We redirect all prints to stderr globally.
_orig_print = builtins.print
def _stderr_print(*args, **kwargs):
    kwargs.setdefault('file', sys.stderr)
    _orig_print(*args, **kwargs)
builtins.print = _stderr_print

from .config import Config
from .ingestion import Document, DocumentParser

__all__ = ["Config", "DocumentParser", "Document"]
