"""Knowledge RAG MCP Server

MCP Server with semantic search + keyword routing for local document retrieval.
Uses ChromaDB for vector storage and Ollama for embeddings.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# ChromaDB
import chromadb

# Ollama for embeddings
import ollama

# FastMCP
from mcp.server.fastmcp import FastMCP

# Local imports
from .config import config
from .ingestion import DocumentParser, Document, parse_documents


class OllamaEmbeddings:
    """Ollama-based embedding function for ChromaDB (v1.4.0+ compatible)"""

    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or config.ollama_model
        self.base_url = base_url or config.ollama_base_url
        self._client = ollama.Client(host=self.base_url)

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = []
        for text in input:
            response = self._client.embeddings(
                model=self.model,
                prompt=text
            )
            embeddings.append(response["embedding"])
        return embeddings

    def name(self) -> str:
        """Return embedding function name (required by ChromaDB v1.4.0+)"""
        return f"ollama-{self.model}"

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents (alias for __call__)"""
        return self(documents)

    def embed_query(self, input = None, **kwargs) -> List[List[float]]:
        """Embed query text(s) - returns list of embeddings"""
        # Handle both string and list inputs
        if isinstance(input, list):
            texts = input
        elif input is not None:
            texts = [input]
        else:
            texts = [kwargs.get('query', '')]

        embeddings = []
        for text in texts:
            response = self._client.embeddings(model=self.model, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings


class KnowledgeOrchestrator:
    """Main orchestrator for knowledge retrieval with semantic search + keyword routing"""

    def __init__(self):
        self.parser = DocumentParser()
        self.embed_fn = OllamaEmbeddings()

        # Initialize ChromaDB with persistent storage (new API v1.4.0+)
        self.chroma_client = chromadb.PersistentClient(
            path=str(config.chroma_dir)
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=self.embed_fn,
            metadata={"description": "Knowledge base for RAG"}
        )

        # Index metadata cache
        self._metadata_file = config.data_dir / "index_metadata.json"
        self._indexed_docs: Dict[str, Dict] = self._load_metadata()

    # =========================================================================
    # Indexing
    # =========================================================================

    def index_all(self, force: bool = False) -> Dict[str, Any]:
        """Index all documents in the documents directory"""
        stats = {
            "total_files": 0,
            "indexed": 0,
            "skipped": 0,
            "errors": 0,
            "chunks_added": 0,
            "categories": {}
        }

        documents = self.parser.parse_directory()
        stats["total_files"] = len(documents)

        for doc in documents:
            try:
                # Skip if already indexed (unless force)
                if not force and doc.id in self._indexed_docs:
                    stats["skipped"] += 1
                    continue

                # Add chunks to ChromaDB
                self._index_document(doc)

                # Track stats
                stats["indexed"] += 1
                stats["chunks_added"] += len(doc.chunks)
                stats["categories"][doc.category] = stats["categories"].get(doc.category, 0) + 1

                # Update metadata cache
                self._indexed_docs[doc.id] = {
                    "source": str(doc.source),
                    "category": doc.category,
                    "format": doc.format,
                    "chunks": len(doc.chunks),
                    "keywords": doc.keywords,
                    "indexed_at": datetime.now().isoformat()
                }

            except Exception as e:
                stats["errors"] += 1
                print(f"[ERROR] Failed to index {doc.source}: {e}")

        # Persist metadata
        self._save_metadata()

        # ChromaDB PersistentClient auto-persists
        return stats

    def _index_document(self, doc: Document) -> None:
        """Index a single document's chunks into ChromaDB"""
        if not doc.chunks:
            return

        ids = [f"{doc.id}_{chunk.index}" for chunk in doc.chunks]
        documents = [chunk.content for chunk in doc.chunks]
        metadatas = [
            {
                "doc_id": doc.id,
                "source": str(doc.source),
                "filename": doc.filename,
                "category": doc.category,
                "format": doc.format,
                "chunk_index": chunk.index,
                "keywords": ",".join(doc.keywords[:10]),
                **chunk.metadata
            }
            for chunk in doc.chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def reindex_all(self) -> Dict[str, Any]:
        """Force reindex all documents (clears existing index)"""
        # Clear collection
        self.chroma_client.delete_collection(config.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=config.collection_name,
            embedding_function=self.embed_fn,
            metadata={"description": "Knowledge base for RAG"}
        )

        # Clear metadata
        self._indexed_docs = {}

        # Reindex
        return self.index_all(force=True)

    # =========================================================================
    # Search
    # =========================================================================

    def query(
        self,
        query_text: str,
        max_results: int = None,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Main search method with keyword routing + semantic search

        1. Try keyword routing first (deterministic, fast)
        2. Fall back to semantic search
        3. Merge and rank results
        """
        max_results = max_results or config.default_results

        # Step 1: Keyword routing
        routed_category = self._route_by_keywords(query_text)

        # Step 2: Build filter
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}
        elif routed_category:
            where_filter = {"category": routed_category}

        # Step 3: Semantic search
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(max_results, config.max_results),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []

        # Step 4: Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0

                # Normalize score: smaller distance = higher relevance
                # Using 1/(1+distance) to normalize to 0-1 range
                score = 1 / (1 + abs(distance)) if distance else 1.0

                formatted.append({
                    "content": doc,
                    "source": metadata.get("source", ""),
                    "filename": metadata.get("filename", ""),
                    "category": metadata.get("category", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "score": round(score, 4),
                    "distance": round(distance, 2),
                    "keywords": metadata.get("keywords", "").split(","),
                    "routed_by": routed_category if routed_category else "semantic"
                })

        return formatted

    def _route_by_keywords(self, query_text: str) -> Optional[str]:
        """Deterministic keyword routing - returns category if match found"""
        query_lower = query_text.lower()

        for category, keywords in config.keyword_routes.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    return category

        return None

    # =========================================================================
    # Document Retrieval
    # =========================================================================

    def get_document(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Get full document content by filepath"""
        filepath = Path(filepath)

        # Try to parse fresh
        try:
            doc = self.parser.parse_file(filepath)
            if doc:
                return {
                    "content": doc.content,
                    "source": str(doc.source),
                    "filename": doc.filename,
                    "category": doc.category,
                    "format": doc.format,
                    "metadata": doc.metadata,
                    "keywords": doc.keywords,
                    "chunk_count": len(doc.chunks)
                }
        except Exception as e:
            print(f"[ERROR] Failed to read document {filepath}: {e}")

        return None

    def list_categories(self) -> Dict[str, int]:
        """List all categories with document counts"""
        categories = {}
        for doc_info in self._indexed_docs.values():
            cat = doc_info.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        return categories

    def list_documents(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        """List all indexed documents, optionally filtered by category"""
        docs = []
        for doc_id, info in self._indexed_docs.items():
            if category and info.get("category") != category:
                continue
            docs.append({
                "id": doc_id,
                "source": info.get("source", ""),
                "category": info.get("category", ""),
                "format": info.get("format", ""),
                "chunks": info.get("chunks", 0),
                "keywords": info.get("keywords", [])[:5]
            })
        return docs

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_documents": len(self._indexed_docs),
            "total_chunks": self.collection.count(),
            "categories": self.list_categories(),
            "supported_formats": config.supported_formats,
            "embedding_model": config.ollama_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap
        }

    # =========================================================================
    # Metadata persistence
    # =========================================================================

    def _load_metadata(self) -> Dict[str, Dict]:
        """Load index metadata from disk"""
        if self._metadata_file.exists():
            try:
                return json.loads(self._metadata_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_metadata(self) -> None:
        """Save index metadata to disk"""
        self._metadata_file.parent.mkdir(parents=True, exist_ok=True)
        self._metadata_file.write_text(
            json.dumps(self._indexed_docs, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )


# =============================================================================
# MCP Server
# =============================================================================

# Initialize FastMCP server
mcp = FastMCP("knowledge-rag")

# Global orchestrator instance (lazy init)
_orchestrator: Optional[KnowledgeOrchestrator] = None


def get_orchestrator() -> KnowledgeOrchestrator:
    """Get or create the orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = KnowledgeOrchestrator()
    return _orchestrator


@mcp.tool()
def search_knowledge(query: str, max_results: int = 5, category: str = None) -> str:
    """
    Search the knowledge base using semantic search with keyword routing.

    Args:
        query: Search query text
        max_results: Maximum number of results (default: 5, max: 20)
        category: Optional category filter (security, ctf, logscale, development, general)

    Returns:
        JSON string with search results including content, source, and relevance score
    """
    orchestrator = get_orchestrator()
    results = orchestrator.query(query, max_results=max_results, category_filter=category)

    if not results:
        return json.dumps({
            "status": "no_results",
            "query": query,
            "message": "No relevant documents found. Try a different query or check if documents are indexed."
        })

    return json.dumps({
        "status": "success",
        "query": query,
        "result_count": len(results),
        "results": results
    }, indent=2, ensure_ascii=False)


@mcp.tool()
def get_document(filepath: str) -> str:
    """
    Get the full content of a specific document.

    Args:
        filepath: Path to the document file

    Returns:
        JSON string with document content and metadata
    """
    orchestrator = get_orchestrator()
    doc = orchestrator.get_document(filepath)

    if not doc:
        return json.dumps({
            "status": "error",
            "message": f"Document not found or could not be read: {filepath}"
        })

    return json.dumps({
        "status": "success",
        "document": doc
    }, indent=2, ensure_ascii=False)


@mcp.tool()
def reindex_documents(force: bool = False) -> str:
    """
    Index or reindex all documents in the knowledge base.

    Args:
        force: If True, reindex all documents (clears existing index)

    Returns:
        JSON string with indexing statistics
    """
    orchestrator = get_orchestrator()

    if force:
        stats = orchestrator.reindex_all()
    else:
        stats = orchestrator.index_all()

    return json.dumps({
        "status": "success",
        "operation": "reindex" if force else "index",
        "stats": stats
    }, indent=2, ensure_ascii=False)


@mcp.tool()
def list_categories() -> str:
    """
    List all document categories with their document counts.

    Returns:
        JSON string with categories and counts
    """
    orchestrator = get_orchestrator()
    categories = orchestrator.list_categories()

    return json.dumps({
        "status": "success",
        "categories": categories,
        "total_documents": sum(categories.values())
    }, indent=2)


@mcp.tool()
def list_documents(category: str = None) -> str:
    """
    List all indexed documents, optionally filtered by category.

    Args:
        category: Optional category filter

    Returns:
        JSON string with document list
    """
    orchestrator = get_orchestrator()
    docs = orchestrator.list_documents(category=category)

    return json.dumps({
        "status": "success",
        "filter": category or "all",
        "count": len(docs),
        "documents": docs
    }, indent=2, ensure_ascii=False)


@mcp.tool()
def get_index_stats() -> str:
    """
    Get statistics about the knowledge base index.

    Returns:
        JSON string with index statistics
    """
    orchestrator = get_orchestrator()
    stats = orchestrator.get_stats()

    return json.dumps({
        "status": "success",
        "stats": stats
    }, indent=2)


# =============================================================================
# Entry point
# =============================================================================

def main():
    """Run the MCP server"""
    import sys

    # Auto-index on startup if empty
    orchestrator = get_orchestrator()
    if orchestrator.collection.count() == 0:
        print("[INFO] No documents indexed. Running initial indexing...")
        stats = orchestrator.index_all()
        print(f"[INFO] Indexed {stats['indexed']} documents with {stats['chunks_added']} chunks")

    # Run MCP server
    mcp.run()


if __name__ == "__main__":
    main()
