"""
╭─╴ PROVIDER SUBPACKAGE ╶────────────────────────────────────────╮
│                                                                │
│   Formal Protocol contracts + registry for embedding, vector   │
│   store, and reranker providers. Bundled defaults auto-        │
│   register on import so ``get_embedding("fastembed")``,        │
│   ``get_vector_store("chromadb")`` and                         │
│   ``get_reranker("cross_encoder")`` work out of the box.       │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``

Import shape:

    from mcp_server.providers import (
        EmbeddingProvider, VectorStore, Reranker,     # Protocols
        register_embedding, register_vector_store, register_reranker,
        get_embedding, get_vector_store, get_reranker,
        load_third_party,
    )

Auto-registration is deferred inside :func:`_register_bundled_defaults`
to avoid import-time cycles: the bundled classes live inside
``mcp_server.retrieval`` and ``mcp_server.storage`` and both packages
import lightweight helpers from :mod:`mcp_server.config`, which imports
nothing from ``mcp_server.providers``. Registration therefore triggers
a one-way import chain that terminates cleanly.
"""

from __future__ import annotations

from .base import EmbeddingProvider, Reranker, VectorStore
from .registry import (
    get_embedding,
    get_embedding_class,
    get_reranker,
    get_reranker_class,
    get_vector_store,
    get_vector_store_class,
    is_embedding_registered,
    is_reranker_registered,
    is_vector_store_registered,
    list_embeddings,
    list_rerankers,
    list_vector_stores,
    load_third_party,
    register_embedding,
    register_reranker,
    register_vector_store,
)


def _register_bundled_defaults() -> None:
    """Register the bundled FastEmbed/CrossEncoder/ChromaDB providers.

    Called eagerly at package import time so users with no custom config
    can call ``get_embedding("fastembed")`` (etc.) immediately. Third-party
    plugins are discovered lazily via :func:`load_third_party` — that keeps
    ``importlib.metadata`` cost off the fast import path.

    Opt-in providers (currently only ``"colbert"``) live under the
    ``vector_stores`` subpackage. Importing that subpackage triggers each
    module's ``@register_vector_store`` decorator so the registry surfaces
    the provider name even when its heavy extras (torch, pylate) are not
    installed — the missing-extra failure is deferred to first use with a
    clear pip-install hint.
    """
    from ..retrieval.embeddings import FastEmbedEmbeddings
    from ..retrieval.rerank import CrossEncoderReranker
    from ..storage.chroma import ChromaVectorStore

    register_embedding("fastembed", FastEmbedEmbeddings)
    register_reranker("cross_encoder", CrossEncoderReranker)
    register_vector_store("chromadb", ChromaVectorStore)

    # Trigger the vector_stores subpackage import as the very last step so
    # any registration failure inside an opt-in provider cannot mask the
    # bundled defaults above. Any exception from an opt-in module is
    # swallowed here with a debug log — the missing provider surfaces as
    # ``KeyError`` at ``get_vector_store("colbert")`` call time, matching
    # the behaviour of the third-party entry_points discovery path.
    try:
        from . import vector_stores  # noqa: F401 — imported for registration side-effect
    except Exception:  # pragma: no cover — defensive; opt-in modules should not raise on import
        import logging as _logging

        _logging.getLogger(__name__).debug("Failed to import optional vector_stores subpackage", exc_info=True)

    # Opt-in embedding providers (R5.4: Matryoshka). Same contract as
    # the vector_stores block above — the subpackage import is guarded
    # so a broken opt-in module cannot mask the bundled ``fastembed``
    # registration. Every backend inside this subpackage defers its
    # vendor-SDK import to first ``embed_documents`` / ``embed_query``
    # call, so bare ``pip install knowledge-rag`` never pulls torch,
    # sentence-transformers, or any other heavy dependency here.
    try:
        from . import embeddings  # noqa: F401 — imported for registration side-effect
    except Exception:  # pragma: no cover — defensive; opt-in modules should not raise on import
        import logging as _logging

        _logging.getLogger(__name__).debug("Failed to import optional embeddings subpackage", exc_info=True)


_register_bundled_defaults()


__all__ = [
    "EmbeddingProvider",
    "Reranker",
    "VectorStore",
    "get_embedding",
    "get_embedding_class",
    "get_reranker",
    "get_reranker_class",
    "get_vector_store",
    "get_vector_store_class",
    "is_embedding_registered",
    "is_reranker_registered",
    "is_vector_store_registered",
    "list_embeddings",
    "list_rerankers",
    "list_vector_stores",
    "load_third_party",
    "register_embedding",
    "register_reranker",
    "register_vector_store",
]
