"""
╭─╴ VECTOR STORE PROVIDERS SUBPACKAGE ╶──────────────────────────╮
│                                                                │
│   Opt-in third-party vector store implementations. The bundled │
│   ChromaDB backend lives in ``mcp_server.storage.chroma`` and  │
│   auto-registers via ``mcp_server.providers.__init__``. New    │
│   backends (ColBERT, Weaviate, Qdrant, ...) plug in here and   │
│   register themselves with the ``@register_vector_store``      │
│   decorator so the orchestrator dispatch stays untouched.      │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``

Design note:
    Every module in this subpackage must self-register at import time by
    calling :func:`mcp_server.providers.register_vector_store` (or by
    applying it as a decorator). Heavy imports (torch, model downloads,
    external clients) must stay deferred to first use so that ``import
    mcp_server.providers`` remains cheap for users who never opt in to
    the alternate backend.
"""

from __future__ import annotations

# Explicit imports so registration side-effects fire on subpackage import.
# Bundled providers only — third-party stores rely on ``entry_points`` via
# ``load_third_party`` on the parent registry module.
from . import colbert  # noqa: F401 — imported for registration side-effect

__all__: list[str] = []
