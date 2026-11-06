"""Knowledge RAG MCP Server - Local Retrieval-Augmented Generation System"""

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


def __getattr__(name: str) -> Any:
    """Import a re-exported name on first access and keep it as a module attribute."""
    if name == "Config":
        from .config import Config

        globals()[name] = Config
        return Config
    if name in ("Document", "DocumentParser"):
        from .ingestion import Document, DocumentParser

        globals()["Document"] = Document
        globals()["DocumentParser"] = DocumentParser
        return {"Document": Document, "DocumentParser": DocumentParser}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
