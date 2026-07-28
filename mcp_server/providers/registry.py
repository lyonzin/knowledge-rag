"""
╭─╴ PROVIDER REGISTRY ╶──────────────────────────────────────────╮
│                                                                │
│   Central registry for embedding providers, vector stores and  │
│   rerankers. Bundled defaults auto-register on import of the   │
│   ``mcp_server.providers`` subpackage; third-party plugins     │
│   register via ``importlib.metadata`` entry_points groups.     │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``

Entry-point groups discovered by :func:`load_third_party`:

    ``knowledge_rag.embeddings``      → EmbeddingProvider implementations
    ``knowledge_rag.vector_stores``   → VectorStore implementations
    ``knowledge_rag.rerankers``       → Reranker implementations

A plugin registers by shipping a package with the following
``pyproject.toml`` entry (illustrative — replace names to taste)::

    [project.entry-points."knowledge_rag.embeddings"]
    ollama = "my_plugin:register"

The referenced attribute is loaded at discovery time; the plugin is
expected to call :func:`register_embedding` (or the sibling registrars)
from its module body or from the referenced callable.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any, Callable, Dict, List, Type, TypeVar

from .base import EmbeddingProvider, Reranker, VectorStore

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry state (module-level, single-source)
# ---------------------------------------------------------------------------
_EMBEDDINGS: Dict[str, Type[EmbeddingProvider]] = {}
_VECTOR_STORES: Dict[str, Type[VectorStore]] = {}
_RERANKERS: Dict[str, Type[Reranker]] = {}

# Entry-point groups exposed to third-party plugins
_ENTRY_POINT_GROUPS = (
    "knowledge_rag.embeddings",
    "knowledge_rag.vector_stores",
    "knowledge_rag.rerankers",
)

# One-shot guard so repeated calls to load_third_party() from CLI + server
# don't produce duplicate warnings.
_third_party_loaded = False

_T = TypeVar("_T")


def _make_registrar(
    store: Dict[str, Type[Any]],
    kind: str,
) -> Callable[..., Any]:
    """Build a dual-mode registrar (decorator or direct call).

    Args:
        store: Backing registry dict for this provider kind.
        kind: Human-readable kind label used in log messages.

    Returns:
        A registrar callable accepting either ``(name)`` (decorator form)
        or ``(name, cls)`` (direct-call form).
    """

    def _registrar(name: str, cls: Any = None) -> Any:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{kind} provider name must be a non-empty string")

        def _install(target: Any) -> Any:
            existing = store.get(name)
            if existing is not None and existing is not target:
                _log.debug(
                    "Overriding %s provider %r: %s -> %s",
                    kind,
                    name,
                    existing.__name__,
                    getattr(target, "__name__", repr(target)),
                )
            store[name] = target
            return target

        if cls is None:
            return _install
        return _install(cls)

    return _registrar


# Public registrars — usable as decorators OR direct calls.
register_embedding = _make_registrar(_EMBEDDINGS, "embedding")
register_vector_store = _make_registrar(_VECTOR_STORES, "vector_store")
register_reranker = _make_registrar(_RERANKERS, "reranker")


# ---------------------------------------------------------------------------
# Class accessors (no instantiation)
# ---------------------------------------------------------------------------


def _get_class(store: Dict[str, Type[Any]], kind: str, name: str) -> Type[Any]:
    """Resolve a registered provider class by name.

    Args:
        store: Backing registry for this provider kind.
        kind: Human-readable kind label used in error messages.
        name: Registered provider name.

    Returns:
        The class registered under ``name``.

    Raises:
        KeyError: If ``name`` has no registered implementation.
    """
    if name not in store:
        available = ", ".join(sorted(store)) or "<none>"
        raise KeyError(f"Unknown {kind} provider: {name!r}. Registered: {available}. Install a plugin or check config.")
    return store[name]


def get_embedding_class(name: str) -> Type[EmbeddingProvider]:
    """Return the ``EmbeddingProvider`` class registered under ``name``."""
    return _get_class(_EMBEDDINGS, "embedding", name)


def get_vector_store_class(name: str) -> Type[VectorStore]:
    """Return the ``VectorStore`` class registered under ``name``."""
    return _get_class(_VECTOR_STORES, "vector_store", name)


def get_reranker_class(name: str) -> Type[Reranker]:
    """Return the ``Reranker`` class registered under ``name``."""
    return _get_class(_RERANKERS, "reranker", name)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def get_embedding(name: str, **kwargs: Any) -> EmbeddingProvider:
    """Instantiate the embedding provider registered under ``name``.

    Args:
        name: Registered provider name (e.g. ``"fastembed"``).
        **kwargs: Passed straight through to the provider constructor.

    Returns:
        A fresh provider instance.

    Raises:
        KeyError: If ``name`` has no registered implementation.
    """
    return get_embedding_class(name)(**kwargs)


def get_vector_store(name: str, **kwargs: Any) -> VectorStore:
    """Instantiate the vector store registered under ``name``.

    Args:
        name: Registered provider name (e.g. ``"chromadb"``).
        **kwargs: Passed straight through to the provider constructor.

    Returns:
        A fresh vector store instance.

    Raises:
        KeyError: If ``name`` has no registered implementation.
    """
    return get_vector_store_class(name)(**kwargs)


def get_reranker(name: str, **kwargs: Any) -> Reranker:
    """Instantiate the reranker registered under ``name``.

    Args:
        name: Registered provider name (e.g. ``"cross_encoder"``).
        **kwargs: Passed straight through to the provider constructor.

    Returns:
        A fresh reranker instance.

    Raises:
        KeyError: If ``name`` has no registered implementation.
    """
    return get_reranker_class(name)(**kwargs)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def list_embeddings() -> List[str]:
    """Return the sorted list of registered embedding provider names."""
    return sorted(_EMBEDDINGS)


def list_vector_stores() -> List[str]:
    """Return the sorted list of registered vector store names."""
    return sorted(_VECTOR_STORES)


def list_rerankers() -> List[str]:
    """Return the sorted list of registered reranker names."""
    return sorted(_RERANKERS)


def is_embedding_registered(name: str) -> bool:
    """Return ``True`` if ``name`` is a registered embedding provider."""
    return name in _EMBEDDINGS


def is_vector_store_registered(name: str) -> bool:
    """Return ``True`` if ``name`` is a registered vector store."""
    return name in _VECTOR_STORES


def is_reranker_registered(name: str) -> bool:
    """Return ``True`` if ``name`` is a registered reranker."""
    return name in _RERANKERS


# ---------------------------------------------------------------------------
# Third-party plugin discovery
# ---------------------------------------------------------------------------


def load_third_party(force: bool = False) -> None:
    """Discover and load third-party provider plugins via entry_points.

    Iterates the three entry-point groups (``knowledge_rag.embeddings``,
    ``knowledge_rag.vector_stores``, ``knowledge_rag.rerankers``) and loads
    each referenced attribute. The referenced attribute is expected to
    register providers as a side-effect of import (module-level
    ``@register_*`` decorators) or to be a callable that performs the
    registration when invoked.

    Failures are isolated: one broken plugin logs a warning and the next
    plugin is tried. This mirrors the resilience contract used elsewhere
    in the codebase (e.g. reranker fallback).

    Args:
        force: When ``True`` re-run discovery even if it already ran once.
            The default (once-per-process) avoids duplicate warnings when
            both the CLI and the MCP server initialize the registry.
    """
    global _third_party_loaded
    if _third_party_loaded and not force:
        return

    for group in _ENTRY_POINT_GROUPS:
        try:
            eps = entry_points(group=group)
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning("Failed to enumerate entry_points for %s: %s", group, exc)
            continue

        for ep in eps:
            try:
                loaded = ep.load()
            except Exception as exc:
                _log.warning("Failed to load provider plugin %s (%s): %s", ep.name, group, exc)
                continue
            # If the referenced attribute is a callable and hasn't self-registered,
            # invoke it to give it a second chance to register.
            if callable(loaded) and not isinstance(loaded, type):
                try:
                    loaded()
                except Exception as exc:
                    _log.warning("Provider plugin callable %s (%s) raised: %s", ep.name, group, exc)

    _third_party_loaded = True


def _reset_for_tests() -> None:
    """Clear registry state — used exclusively by unit tests.

    Not part of the public API. Kept module-private via underscore prefix
    and re-exported only through the ``providers`` subpackage tests.
    """
    global _third_party_loaded
    _EMBEDDINGS.clear()
    _VECTOR_STORES.clear()
    _RERANKERS.clear()
    _third_party_loaded = False


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
