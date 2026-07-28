"""
╭─╴ EMBEDDING PROVIDER SUBPACKAGE ╶──────────────────────────────╮
│                                                                │
│   Opt-in embedding provider modules for Fase 5. Bundled        │
│   ``FastEmbedEmbeddings`` (registered as ``fastembed``) stays  │
│   the default and lives in :mod:`mcp_server.retrieval.         │
│   embeddings`; this subpackage holds the ADDITIONAL opt-in     │
│   backends whose vendor SDKs ship as separate extras.          │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``

Bundled backend registration happens as a side-effect of importing this
subpackage from :mod:`mcp_server.providers`. Each backend module wraps
its ``@register_embedding("<name>")`` call in a module body that costs
nothing until the vendor SDK is actually touched — every SDK import is
deferred to first ``embed_documents`` / ``embed_query`` call. The
contract is stricter than "SDK-optional": it's "SDK-deferred", so a
bare ``pip install knowledge-rag`` NEVER pulls sentence-transformers,
torch, einops, or any other heavy Matryoshka dependency.
"""

from __future__ import annotations

# ── Bundled backend imports (side-effect: registers under name) ─────────
# Each import triggers ``@register_embedding("<name>")`` at module top level.
# Import failure of a backend module is NOT expected — the vendor SDK is
# only imported inside ``_lazy_load()``, so bare import is always safe
# even without the ``[embed-matryoshka]`` / ``[embed-jina]`` extras
# installed. If a backend module somehow becomes uneager (imports its SDK
# at top level), that bug would surface here and the offending backend
# is skipped so siblings still register. Each backend is guarded
# INDIVIDUALLY (not at subpackage granularity) so parallel Fase 5
# rollouts (R5.4 Matryoshka + R5.7 Jina, in progress) can land
# independently without a still-missing sibling module knocking out
# the whole subpackage.
import logging as _logging

_log = _logging.getLogger(__name__)

for _backend in ("matryoshka", "jina_late"):
    try:
        __import__(f"{__name__}.{_backend}")
    except Exception:  # pragma: no cover — defensive; opt-in modules should not raise on import
        _log.debug("Failed to import opt-in embedding backend %r", _backend, exc_info=True)

del _backend, _log, _logging

__all__: list[str] = []
