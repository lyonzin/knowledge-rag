"""Knowledge RAG MCP Server - Local Retrieval-Augmented Generation System"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "5.0.0"
__author__ = "Ailton Rocha (Lyon.)"

__all__ = ["Config", "DocumentParser", "Document"]

if TYPE_CHECKING:
    from .config import Config
    from .ingestion import Document, DocumentParser

# Re-exports load on first use (PEP 562). Importing .config creates the data and
# documents folders under the base directory, which can be the current folder,
# so `import mcp_server.skills_cli` and other light submodules must not trigger it.
_REEXPORTS = {"Config": ".config", "Document": ".ingestion", "DocumentParser": ".ingestion"}


def __getattr__(name: str) -> Any:
    """Import a re-exported name on first access and keep it as a module attribute."""
    if name not in _REEXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_REEXPORTS[name], __name__), name)
    globals()[name] = value
    return value
