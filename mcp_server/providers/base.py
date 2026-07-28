"""
╭─╴ PROVIDER PROTOCOLS ╶─────────────────────────────────────────╮
│                                                                │
│   Formal Protocol contracts for embedding providers, vector    │
│   stores, and rerankers. Third-party plugins conform to these  │
│   Protocols and register themselves via the registry.          │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``

Design note:
    These Protocols are intentionally loose. Bundled defaults
    (``FastEmbedEmbeddings``, ``CrossEncoderReranker`` and
    ``ChromaVectorStore``) predate the Protocol layer — they were
    extracted from the pre-refactor ``server.py`` and their
    method signatures are shaped by ChromaDB's ``embedding_function``
    contract. The Protocols capture the minimum surface that
    third-party providers must expose to be usable from
    ``KnowledgeOrchestrator``; the bundled defaults expose that
    surface plus historical extras (e.g. ``FastEmbedEmbeddings.name()``
    method for ChromaDB compat) and stay backwards-compatible.

    ``@runtime_checkable`` is applied so downstream code can use
    ``isinstance(obj, EmbeddingProvider)`` as a coarse sanity check.
    Structural checks only verify attribute *presence*, not signatures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for dense embedding providers.

    Implementations produce dense vector embeddings for text. Providers may
    load models lazily; construction should not download heavy assets so
    idle server processes stay cheap (matches the pattern used by
    ``FastEmbedEmbeddings`` since v3.8.0).

    Attributes:
        name: Human-readable provider identifier. Bundled providers expose
            this as a method for ChromaDB embedding_function compat; new
            providers may expose it as a plain attribute.
        dimension: Output vector dimensionality. Must be consistent across
            calls — the vector store relies on this for schema validation.
    """

    name: str
    dimension: int

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents into dense vectors.

        Args:
            texts: Documents to embed.

        Returns:
            One vector per input document, each of length ``dimension``.
        """
        ...

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string into a dense vector.

        Args:
            text: Query text to embed.

        Returns:
            Vector of length ``dimension``.
        """
        ...


@runtime_checkable
class TokenAwareEmbeddingProvider(Protocol):
    """Optional extension for embedding providers that expose token-level output.

    A provider implementing this Protocol can be used by the Late Chunking
    pipeline (Jina 2024, R5.7): embed the whole document once, then mean-pool
    token embeddings across each chunk's character span so every chunk vector
    carries global document context.

    Only long-context embedding models are useful here — Jina v3 (8K),
    Alibaba GTE (8K), Cohere embed v3, etc. Short-context models
    (BGE-small 512, MiniLM 512) technically satisfy the Protocol but do
    not benefit from late chunking because the full-document embed call
    would truncate anyway.

    Providers that also implement :class:`EmbeddingProvider` remain fully
    usable in every non-late-chunking code path. The two Protocols are
    additive, not exclusive.

    Attributes:
        name: Human-readable provider identifier.
        dimension: Output vector dimensionality.
        supports_late_chunking: Static ``True`` marker so callers can gate
            the late-chunking code path with a plain attribute check
            (``getattr(provider, "supports_late_chunking", False)``)
            without instantiating an :func:`isinstance` walk.
    """

    name: str
    dimension: int
    supports_late_chunking: bool

    def embed_tokens(self, text: str) -> List[Dict[str, Any]]:
        """Return per-token embeddings with character offsets for ``text``.

        The returned records let the Late Chunking implementation map
        character-based chunk boundaries onto token spans, then mean-pool
        the token vectors within each span.

        Args:
            text: Full document text. May be up to the provider's context
                window (typically 8K tokens for long-context models). Longer
                inputs are truncated per provider policy.

        Returns:
            list[dict]: One dict per token in tokenisation order, each with:

            * ``token_id`` (``int``): Tokeniser vocabulary id, kept purely
              for diagnostics.
            * ``offset_start`` (``int``): Inclusive character offset in
              ``text`` where this token begins. May be ``0`` for special
              tokens (CLS, BOS) — the late chunker filters those via the
              ``offset_end > offset_start`` invariant.
            * ``offset_end`` (``int``): Exclusive character offset where
              this token ends.
            * ``embedding`` (``list[float]``): Token's dense vector of
              length :attr:`dimension`.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage backends.

    Implementations persist ``(id, embedding, metadata, document)`` tuples
    and answer nearest-neighbour queries. The bundled ``ChromaVectorStore``
    wraps ChromaDB's ``PersistentClient`` + ``Collection`` and exposes both
    the Protocol methods and the underlying ChromaDB handles
    (``.client``, ``.collection``) for advanced use inside the orchestrator.

    Attributes:
        name: Human-readable provider identifier (e.g. ``"chromadb"``).
    """

    name: str

    def add(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """Insert or upsert records.

        Args:
            ids: Stable identifiers for each record.
            embeddings: Precomputed embeddings; if ``None`` the store may
                delegate to a provider-side embedding function.
            metadatas: Optional per-record metadata dicts.
            documents: Optional per-record raw text.
        """
        ...

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return the nearest neighbours to each query embedding.

        Args:
            query_embeddings: One or more dense query vectors.
            n_results: Maximum number of hits to return per query.
            where: Optional metadata filter (backend-specific dialect).

        Returns:
            Backend-defined result mapping. Callers should treat this as
            opaque and rely on the bundled orchestrator adapters.
        """
        ...

    def get(
        self,
        ids: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch stored records by id.

        Args:
            ids: Record identifiers to fetch. ``None`` may return all rows
                in backends that support it.
            include: Fields to hydrate (e.g. ``["documents", "metadatas"]``).

        Returns:
            Backend-defined result mapping.
        """
        ...

    def delete(self, ids: List[str]) -> None:
        """Remove records by id.

        Args:
            ids: Record identifiers to delete.
        """
        ...

    def count(self) -> int:
        """Return the total number of stored records."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for cross-encoder rerankers.

    Implementations re-score a candidate set produced by the retrieval
    pipeline. Providers should fail gracefully — the bundled
    ``CrossEncoderReranker`` returns the input ordering when its model
    cannot be loaded so search never breaks because of a reranker outage.

    Attributes:
        name: Human-readable provider identifier (e.g. ``"cross_encoder"``).
    """

    name: str

    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: int,
    ) -> List[Any]:
        """Re-score and return the top-k documents for a query.

        Args:
            query: Original query string.
            documents: Candidate list to rerank. The bundled reranker
                accepts result dicts with a ``"document"`` key; third-party
                implementations may narrow the accepted shape.
            top_k: Maximum number of results to return after reranking.

        Returns:
            The top-k documents, reordered by reranker score. On failure
            the implementation should return ``documents[:top_k]`` rather
            than raising.
        """
        ...


__all__ = [
    "EmbeddingProvider",
    "Reranker",
    "TokenAwareEmbeddingProvider",
    "VectorStore",
]
