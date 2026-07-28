"""
╭─╴ CHROMA STORAGE ADAPTER ╶─────────────────────────────────────╮
│                                                                │
│   ChromaDB SQLite tuning helpers + ``ChromaVectorStore``       │
│   wrapper that satisfies the ``VectorStore`` Protocol while    │
│   exposing the underlying ``chromadb`` client and collection   │
│   for advanced use inside :class:`KnowledgeOrchestrator`.      │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``

Backwards compatibility:
    ``_enable_wal_mode`` is re-exported unchanged and remains callable
    from :mod:`mcp_server.retrieval.orchestrator`. The new
    ``ChromaVectorStore`` wrapper is additive — no existing import
    path changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def _enable_wal_mode(chroma_dir: Path) -> None:
    """Enable WAL journal mode on ChromaDB's SQLite for concurrent reads."""
    import sqlite3

    sqlite_path = chroma_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        return
    try:
        conn = sqlite3.connect(str(sqlite_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.close()
        print("[INFO] ChromaDB SQLite: WAL mode enabled")
    except Exception as e:
        print(f"[WARN] Could not enable WAL mode: {e}")


class ChromaVectorStore:
    """ChromaDB vector-store wrapper satisfying the ``VectorStore`` Protocol.

    Instances hold a ChromaDB ``PersistentClient`` and a single named
    ``Collection``. The orchestrator uses ``.client`` and ``.collection``
    directly for behaviour that is inherently ChromaDB-specific (queries
    that dispatch through the collection's ``embedding_function``, complex
    ``where`` filters, corruption recovery) while third-party callers may
    interact through the Protocol methods.

    Attributes:
        name: Provider identifier — always ``"chromadb"``.
        path: On-disk location of the persistent database.
        collection_name: Name of the collection wrapped by this instance.
        client: Underlying ``chromadb.PersistentClient`` instance.
        collection: Underlying ``chromadb.Collection`` instance.

    Notes:
        Construction is lightweight — ``PersistentClient`` opens the
        SQLite file eagerly (that mirrors the pre-refactor behaviour in
        ``KnowledgeOrchestrator.__init__``). Callers that need lazy
        initialisation should build the wrapper on first use.
    """

    name: str = "chromadb"

    def __init__(
        self,
        path: Path,
        collection_name: str,
        embedding_function: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        enable_wal: bool = False,
    ) -> None:
        """Open the persistent database and materialise the target collection.

        Args:
            path: On-disk location of the persistent database.
            collection_name: Name of the collection to open (created if
                absent).
            embedding_function: Optional ChromaDB embedding function used
                when the caller adds documents without precomputed vectors.
            metadata: Optional metadata dict attached to the collection at
                creation time.
            enable_wal: Enable SQLite WAL mode for concurrent reads. The
                orchestrator opts in for non-stdio transports.
        """
        import chromadb

        self.path: Path = Path(path)
        self.collection_name: str = collection_name
        self.embedding_function = embedding_function
        self._metadata = metadata or {"description": "Knowledge base for RAG"}

        self.client = chromadb.PersistentClient(path=str(self.path))
        if enable_wal:
            _enable_wal_mode(self.path)

        # Build the collection with the supplied embedding_function. Corruption
        # recovery stays in KnowledgeOrchestrator._safe_get_collection because
        # it needs to swap client + collection references atomically on the
        # orchestrator, not just here.
        self.collection = self._create_collection()

    def _create_collection(self):
        """Create or fetch the collection with the wrapper's embedding function."""
        kwargs: Dict[str, Any] = {
            "name": self.collection_name,
            "metadata": self._metadata,
        }
        if self.embedding_function is not None:
            kwargs["embedding_function"] = self.embedding_function
        return self.client.get_or_create_collection(**kwargs)

    # ------------------------------------------------------------------
    # Protocol surface — thin delegates to the underlying collection.
    # ------------------------------------------------------------------

    def add(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """Insert or upsert records via the underlying ChromaDB collection."""
        kwargs: Dict[str, Any] = {"ids": ids}
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        if metadatas is not None:
            kwargs["metadatas"] = metadatas
        if documents is not None:
            kwargs["documents"] = documents
        self.collection.add(**kwargs)

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Nearest-neighbour query against the underlying collection."""
        kwargs: Dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
        }
        if where is not None:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def get(
        self,
        ids: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch stored records by id."""
        kwargs: Dict[str, Any] = {}
        if ids is not None:
            kwargs["ids"] = ids
        if include is not None:
            kwargs["include"] = include
        return self.collection.get(**kwargs)

    def delete(self, ids: List[str]) -> None:
        """Remove records by id."""
        self.collection.delete(ids=ids)

    def count(self) -> int:
        """Return the number of records in the underlying collection."""
        return self.collection.count()


__all__ = ["ChromaVectorStore", "_enable_wal_mode"]
