"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                         KNOWLEDGE RAG MCP SERVER                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

MCP Server with semantic search + keyword routing for local document retrieval.
Uses ChromaDB for vector storage and Ollama for embeddings.

Features:
    - Hybrid search (semantic + BM25 keyword) with RRF fusion
    - Incremental indexing (only re-indexes changed files)
    - Query caching with TTL for instant repeat queries
    - Chunk deduplication via content hashing
    - Score normalization (0-1 range)

Autor:   Lyon (Ailton Rocha)
Versão:  1.1.0
Data:    2026-03-03

By Lyon :) Legal Ne?
"""

import json
import hashlib
import asyncio
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ChromaDB
import chromadb

# Ollama for embeddings
import ollama

# BM25 for keyword search (hybrid search)
from rank_bm25 import BM25Okapi

# FastMCP
from mcp.server.fastmcp import FastMCP

# Local imports
from .config import config
from .ingestion import DocumentParser, Document, parse_documents


# =============================================================================
# QUERY CACHE
# =============================================================================

class QueryCache:
    """
    LRU cache with TTL for search queries.

    Avoids redundant searches when the same query is executed multiple times.
    Uses OrderedDict for O(1) LRU eviction.

    Args:
        max_size: Maximum number of cached entries (default: 100)
        ttl_seconds: Time-to-live for cache entries in seconds (default: 300)
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, max_results: int, category: Optional[str], hybrid_alpha: float) -> str:
        """Generate cache key from query parameters"""
        raw = f"{query}|{max_results}|{category}|{hybrid_alpha}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, query: str, max_results: int, category: Optional[str], hybrid_alpha: float) -> Optional[Any]:
        """Get cached result if exists and not expired"""
        key = self._make_key(query, max_results, category, hybrid_alpha)

        if key in self._cache:
            timestamp, result = self._cache[key]

            # Check TTL
            if time.time() - timestamp < self.ttl_seconds:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return result
            else:
                # Expired - remove
                del self._cache[key]

        self._misses += 1
        return None

    def put(self, query: str, max_results: int, category: Optional[str], hybrid_alpha: float, result: Any) -> None:
        """Store result in cache"""
        key = self._make_key(query, max_results, category, hybrid_alpha)

        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (time.time(), result)

    def invalidate(self) -> None:
        """Clear entire cache (call after reindex)"""
        self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics"""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{(self._hits / total * 100):.1f}%" if total > 0 else "0%"
        }


# =============================================================================
# EMBEDDINGS
# =============================================================================

class OllamaEmbeddings:
    """
    Ollama-based embedding function for ChromaDB (v1.4.0+ compatible).

    Uses ThreadPoolExecutor for parallel embedding generation to improve
    indexing performance. Default: 4 parallel workers.
    """

    def __init__(self, model: str = None, base_url: str = None, max_workers: int = 4):
        self.model = model or config.ollama_model
        self.base_url = base_url or config.ollama_base_url
        self.max_workers = max_workers
        self._client = ollama.Client(host=self.base_url)

    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text (internal method for parallel execution)"""
        try:
            response = self._client.embeddings(
                model=self.model,
                prompt=text
            )
            return response["embedding"]
        except Exception as e:
            print(f"[WARN] Embedding failed for text chunk: {e}")
            # Return zero vector on failure (will have low similarity)
            return [0.0] * 768  # nomic-embed-text dimension

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using parallel execution.

        For small batches (< 4 texts): sequential processing
        For larger batches: parallel processing with ThreadPoolExecutor
        """
        if not input:
            return []

        # For small batches, sequential is fine (avoids thread overhead)
        if len(input) < 4:
            return [self._embed_single(text) for text in input]

        # Parallel processing for larger batches
        embeddings = [None] * len(input)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks with their indices
            future_to_idx = {
                executor.submit(self._embed_single, text): idx
                for idx, text in enumerate(input)
            }

            # Collect results maintaining order
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    embeddings[idx] = future.result()
                except Exception as e:
                    print(f"[WARN] Embedding task {idx} failed: {e}")
                    embeddings[idx] = [0.0] * 768

        return embeddings

    def name(self) -> str:
        """Return embedding function name (required by ChromaDB v1.4.0+)"""
        return f"ollama-{self.model}"

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents (alias for __call__)"""
        return self(documents)

    def embed_query(self, input=None, **kwargs) -> List[List[float]]:
        """Embed query text(s) - returns list of embeddings"""
        # Handle both string and list inputs
        if isinstance(input, list):
            texts = input
        elif input is not None:
            texts = [input]
        else:
            texts = [kwargs.get('query', '')]

        # Queries are typically single texts, use sequential
        return [self._embed_single(text) for text in texts]


class BM25Index:
    """
    BM25 keyword index for hybrid search.

    Maintains a BM25 index of all document chunks for fast keyword-based retrieval.
    Used in combination with semantic search for hybrid search.
    """

    def __init__(self):
        self.corpus: List[str] = []  # Original texts
        self.corpus_ids: List[str] = []  # Chunk IDs (doc_id_chunkIndex)
        self.bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric"""
        # Keep technical terms intact (CVE-2021-44228, etc.)
        text_lower = text.lower()
        # Split on whitespace and punctuation, keeping alphanumeric and hyphens
        tokens = re.findall(r'[a-z0-9][-a-z0-9]*[a-z0-9]|[a-z0-9]', text_lower)
        return tokens

    def add_documents(self, chunk_ids: List[str], texts: List[str]) -> None:
        """Add documents to the BM25 index"""
        for chunk_id, text in zip(chunk_ids, texts):
            self.corpus.append(text)
            self.corpus_ids.append(chunk_id)
            self._tokenized_corpus.append(self._tokenize(text))

    def build_index(self) -> None:
        """Build/rebuild the BM25 index from the corpus"""
        if self._tokenized_corpus:
            self.bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search the BM25 index.

        Returns list of (chunk_id, score) tuples sorted by score descending.
        """
        if not self.bm25 or not self.corpus:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k results with their scores
        results = []
        for idx, score in enumerate(scores):
            if score > 0:  # Only include non-zero scores
                results.append((self.corpus_ids[idx], score))

        # Sort by score descending and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        """Clear the index"""
        self.corpus = []
        self.corpus_ids = []
        self._tokenized_corpus = []
        self.bm25 = None

    def __len__(self) -> int:
        return len(self.corpus)


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

        # BM25 index for hybrid search
        self.bm25_index = BM25Index()
        self._bm25_initialized = False

        # Query cache (LRU with TTL)
        self.query_cache = QueryCache(max_size=100, ttl_seconds=300)

        # Index metadata cache
        self._metadata_file = config.data_dir / "index_metadata.json"
        self._indexed_docs: Dict[str, Dict] = self._load_metadata()

        # Chunk dedup tracking (content_hash → chunk_id)
        self._chunk_hashes: Dict[str, str] = self._build_dedup_index()

    def _ensure_bm25_index(self) -> None:
        """Lazy initialization of BM25 index from existing ChromaDB data"""
        if self._bm25_initialized:
            return

        # Load all documents from ChromaDB to build BM25 index
        try:
            count = self.collection.count()
            if count > 0:
                # Get all documents from ChromaDB
                all_data = self.collection.get(
                    include=["documents"],
                    limit=count
                )
                if all_data["ids"] and all_data["documents"]:
                    self.bm25_index.add_documents(
                        all_data["ids"],
                        all_data["documents"]
                    )
                    self.bm25_index.build_index()
                    print(f"[INFO] BM25 index built with {len(self.bm25_index)} documents")
        except Exception as e:
            print(f"[WARN] Failed to build BM25 index: {e}")

        self._bm25_initialized = True

    # =========================================================================
    # Indexing
    # =========================================================================

    def index_all(self, force: bool = False) -> Dict[str, Any]:
        """
        Index documents with incremental change detection.

        Compares file mtime/size against stored metadata to detect changes.
        Only re-indexes files that are new or modified. Cleans up orphaned
        chunks from files that were modified or deleted.

        Args:
            force: If True, skip change detection and index everything

        Returns:
            Dict with indexing statistics
        """
        stats = {
            "total_files": 0,
            "indexed": 0,
            "updated": 0,
            "skipped": 0,
            "deleted": 0,
            "errors": 0,
            "chunks_added": 0,
            "chunks_removed": 0,
            "dedup_skipped": 0,
            "categories": {}
        }

        documents = self.parser.parse_directory()
        stats["total_files"] = len(documents)

        # Build reverse map: filepath → doc_id for detecting modified files
        path_to_docid: Dict[str, str] = {}
        for doc_id, info in self._indexed_docs.items():
            path_to_docid[info.get("source", "")] = doc_id

        # Track which filepaths are still present (for orphan detection)
        current_paths = set()

        for doc in documents:
            current_paths.add(str(doc.source))
            try:
                source_str = str(doc.source)
                existing_doc_id = path_to_docid.get(source_str)

                if not force and existing_doc_id:
                    existing_meta = self._indexed_docs.get(existing_doc_id, {})

                    # Compare mtime and size to detect changes
                    stored_mtime = existing_meta.get("file_mtime", "")
                    stored_size = existing_meta.get("file_size", 0)

                    try:
                        current_stat = doc.source.stat()
                        current_mtime = datetime.fromtimestamp(current_stat.st_mtime).isoformat()
                        current_size = current_stat.st_size
                    except OSError:
                        current_mtime = ""
                        current_size = 0

                    # File unchanged → skip
                    if stored_mtime == current_mtime and stored_size == current_size:
                        stats["skipped"] += 1
                        continue

                    # File changed → delete old chunks first
                    removed = self._remove_document_chunks(existing_doc_id)
                    stats["chunks_removed"] += removed
                    del self._indexed_docs[existing_doc_id]
                    stats["updated"] += 1
                elif not force and doc.id in self._indexed_docs:
                    stats["skipped"] += 1
                    continue

                # Index the document (with dedup)
                chunks_added, dedup_skipped = self._index_document(doc)

                # Track stats
                if not (existing_doc_id and not force):
                    stats["indexed"] += 1
                stats["chunks_added"] += chunks_added
                stats["dedup_skipped"] += dedup_skipped
                stats["categories"][doc.category] = stats["categories"].get(doc.category, 0) + 1

                # Store metadata with file mtime for incremental detection
                try:
                    file_stat = doc.source.stat()
                    file_mtime = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    file_size = file_stat.st_size
                except OSError:
                    file_mtime = datetime.now().isoformat()
                    file_size = 0

                self._indexed_docs[doc.id] = {
                    "source": str(doc.source),
                    "category": doc.category,
                    "format": doc.format,
                    "chunks": chunks_added,
                    "keywords": doc.keywords,
                    "indexed_at": datetime.now().isoformat(),
                    "file_mtime": file_mtime,
                    "file_size": file_size,
                }

            except Exception as e:
                stats["errors"] += 1
                print(f"[ERROR] Failed to index {doc.source}: {e}")

        # Clean up orphaned docs (files that were deleted from disk)
        orphan_ids = []
        for doc_id, info in list(self._indexed_docs.items()):
            if info.get("source", "") not in current_paths:
                removed = self._remove_document_chunks(doc_id)
                stats["chunks_removed"] += removed
                stats["deleted"] += 1
                orphan_ids.append(doc_id)
                print(f"[CLEANUP] Removed orphaned doc: {info.get('source', doc_id)}")

        for doc_id in orphan_ids:
            del self._indexed_docs[doc_id]

        # Persist metadata
        self._save_metadata()

        # Invalidate query cache after indexing changes
        if stats["indexed"] > 0 or stats["updated"] > 0 or stats["deleted"] > 0:
            self.query_cache.invalidate()

        return stats

    def _index_document(self, doc: Document) -> Tuple[int, int]:
        """
        Index a single document's chunks into ChromaDB and BM25.

        Performs content-hash deduplication: chunks with identical content
        are skipped to prevent index bloat.

        Returns:
            Tuple of (chunks_added, chunks_skipped_by_dedup)
        """
        if not doc.chunks:
            return 0, 0

        # Filter out duplicate chunks via content hashing
        unique_ids = []
        unique_docs = []
        unique_metas = []
        dedup_skipped = 0

        for chunk in doc.chunks:
            content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:20]
            chunk_id = f"{doc.id}_{chunk.index}"

            # Skip if identical content already indexed
            if content_hash in self._chunk_hashes:
                dedup_skipped += 1
                continue

            self._chunk_hashes[content_hash] = chunk_id
            unique_ids.append(chunk_id)
            unique_docs.append(chunk.content)
            unique_metas.append({
                "doc_id": doc.id,
                "source": str(doc.source),
                "filename": doc.filename,
                "category": doc.category,
                "format": doc.format,
                "chunk_index": chunk.index,
                "keywords": ",".join(doc.keywords[:10]),
                "content_hash": content_hash,
                **chunk.metadata
            })

        if unique_ids:
            # Add to ChromaDB (semantic search)
            self.collection.add(
                ids=unique_ids,
                documents=unique_docs,
                metadatas=unique_metas
            )

            # Add to BM25 index (keyword search)
            self.bm25_index.add_documents(unique_ids, unique_docs)

        return len(unique_ids), dedup_skipped

    def _remove_document_chunks(self, doc_id: str) -> int:
        """
        Remove all chunks belonging to a document from ChromaDB and BM25.

        Args:
            doc_id: The document ID whose chunks should be removed

        Returns:
            Number of chunks removed
        """
        try:
            # Find all chunks with this doc_id
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"]
            )

            if results["ids"]:
                # Remove content hashes from dedup index
                for meta in results["metadatas"]:
                    content_hash = meta.get("content_hash", "")
                    if content_hash and content_hash in self._chunk_hashes:
                        del self._chunk_hashes[content_hash]

                # Delete from ChromaDB
                self.collection.delete(ids=results["ids"])

                # BM25 needs full rebuild (no delete support)
                self._bm25_initialized = False

                return len(results["ids"])
        except Exception as e:
            print(f"[WARN] Failed to remove chunks for doc {doc_id}: {e}")

        return 0

    def _build_dedup_index(self) -> Dict[str, str]:
        """Build deduplication index from existing ChromaDB data"""
        dedup = {}
        try:
            count = self.collection.count()
            if count > 0:
                all_data = self.collection.get(
                    include=["metadatas"],
                    limit=count
                )
                for chunk_id, meta in zip(all_data["ids"], all_data["metadatas"]):
                    content_hash = meta.get("content_hash", "")
                    if content_hash:
                        dedup[content_hash] = chunk_id
        except Exception as e:
            print(f"[WARN] Failed to build dedup index: {e}")
        return dedup

    def reindex_all(self) -> Dict[str, Any]:
        """
        Smart reindex: incremental detection + BM25 rebuild + orphan cleanup.

        Unlike a nuclear rebuild (which re-embeds ALL chunks via Ollama and takes
        forever), this method:
        1. Detects new/changed/deleted files via mtime+size comparison
        2. Only re-embeds new or changed documents (FAST for small changes)
        3. Removes orphaned chunks from deleted files
        4. Rebuilds BM25 index from existing ChromaDB data (no Ollama needed)
        5. Cleans up orphan UUID folders in ChromaDB storage

        This is what `force=True` calls. For a true nuclear rebuild (corrupted DB),
        use `nuclear_rebuild()` instead.
        """
        import shutil

        print("[REINDEX] Starting smart incremental reindex...")
        start_time = time.time()

        # Step 1: Incremental index (new/changed/deleted detection)
        stats = self.index_all(force=False)

        # Step 2: Rebuild BM25 index from current ChromaDB data
        print("[REINDEX] Rebuilding BM25 index...")
        self.bm25_index.clear()
        self._bm25_initialized = False
        self.bm25_index.build_index()
        self._bm25_initialized = True
        print(f"[REINDEX] BM25 index rebuilt with {len(self.bm25_index)} documents")

        # Step 3: Clean orphan UUID folders (ChromaDB doesn't auto-clean)
        chroma_dir = config.chroma_dir
        orphans_cleaned = 0
        if chroma_dir.exists():
            for item in chroma_dir.iterdir():
                if item.is_dir() and len(item.name) == 36 and '-' in item.name:
                    try:
                        # Check if folder is actually used by ChromaDB
                        if not any(item.iterdir()):
                            shutil.rmtree(item)
                            orphans_cleaned += 1
                            print(f"[CLEANUP] Removed empty orphan folder: {item.name}")
                    except Exception:
                        pass

        # Step 4: Invalidate query cache
        self.query_cache.invalidate()

        elapsed = time.time() - start_time
        stats["orphan_folders_cleaned"] = orphans_cleaned
        stats["elapsed_seconds"] = round(elapsed, 2)
        print(f"[REINDEX] Smart reindex completed in {elapsed:.1f}s "
              f"(indexed: {stats['indexed']}, updated: {stats['updated']}, "
              f"skipped: {stats['skipped']}, deleted: {stats['deleted']})")

        return stats

    def nuclear_rebuild(self) -> Dict[str, Any]:
        """
        Nuclear rebuild: DELETE everything and re-embed ALL documents from scratch.

        WARNING: This is EXTREMELY SLOW (re-embeds all chunks via Ollama).
        Only use when ChromaDB is corrupted or embeddings model changed.

        For normal reindexing, use `reindex_all()` instead (smart incremental).
        """
        import shutil

        print("[NUCLEAR] Starting full rebuild — this will take a long time...")
        start_time = time.time()

        # Step 1: Delete collection from ChromaDB
        try:
            self.chroma_client.delete_collection(config.collection_name)
            print("[NUCLEAR] Deleted ChromaDB collection")
        except Exception:
            pass

        # Step 2: Clean ALL UUID folders
        chroma_dir = config.chroma_dir
        if chroma_dir.exists():
            for item in chroma_dir.iterdir():
                if item.is_dir() and len(item.name) == 36 and '-' in item.name:
                    try:
                        shutil.rmtree(item)
                    except Exception:
                        pass

        # Step 3: Recreate collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=self.embed_fn,
            metadata={"description": "Knowledge base for RAG"}
        )

        # Step 4: Clear ALL caches and metadata
        self._indexed_docs = {}
        self.bm25_index.clear()
        self._bm25_initialized = False
        self._chunk_hashes = {}
        self.query_cache.invalidate()

        # Step 5: Reindex everything (force=True skips mtime check)
        stats = self.index_all(force=True)

        # Step 6: Build BM25 index
        self.bm25_index.build_index()
        self._bm25_initialized = True

        elapsed = time.time() - start_time
        stats["elapsed_seconds"] = round(elapsed, 2)
        print(f"[NUCLEAR] Full rebuild completed in {elapsed:.1f}s "
              f"({stats['indexed']} docs, {stats['chunks_added']} chunks)")

        return stats

    # =========================================================================
    # Search
    # =========================================================================

    def query(
        self,
        query_text: str,
        max_results: int = None,
        category_filter: Optional[str] = None,
        hybrid_alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic search + BM25 keyword search.

        Uses Reciprocal Rank Fusion (RRF) to combine results from both methods.
        Results are cached with TTL for instant repeat queries.
        Scores are normalized to 0-1 range.

        Args:
            query_text: Search query
            max_results: Maximum results to return
            category_filter: Optional category filter
            hybrid_alpha: Weight for semantic vs keyword (0.0 = keyword only, 1.0 = semantic only)

        Returns:
            List of results sorted by combined RRF score (normalized 0-1)
        """
        max_results = max_results or config.default_results

        # Check cache first
        cached = self.query_cache.get(query_text, max_results, category_filter, hybrid_alpha)
        if cached is not None:
            return cached

        # Ensure BM25 index is ready
        self._ensure_bm25_index()

        # Step 1: Keyword routing (for category detection)
        routed_category = self._route_by_keywords(query_text)

        # Step 2: Build filter
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}
        elif routed_category:
            where_filter = {"category": routed_category}

        # Step 3: Semantic search (ChromaDB) - SKIP if hybrid_alpha=0 (keyword only)
        semantic_results = {}
        if hybrid_alpha > 0:
            try:
                n_candidates = min(max_results * 3, config.max_results)
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_candidates,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )

                if results["ids"] and results["ids"][0]:
                    for i, chunk_id in enumerate(results["ids"][0]):
                        semantic_results[chunk_id] = {
                            "rank": i + 1,
                            "distance": results["distances"][0][i] if results["distances"] else 0,
                            "document": results["documents"][0][i] if results["documents"] else "",
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                        }
            except Exception as e:
                print(f"[WARN] Semantic search failed: {e}")

        # Step 4: BM25 keyword search - SKIP if hybrid_alpha=1.0 (semantic only)
        bm25_results = {}
        if hybrid_alpha < 1.0:
            try:
                bm25_hits = self.bm25_index.search(query_text, top_k=max_results * 3)
                for rank, (chunk_id, bm25_score) in enumerate(bm25_hits):
                    bm25_results[chunk_id] = {
                        "rank": rank + 1,
                        "bm25_score": bm25_score
                    }
            except Exception as e:
                print(f"[WARN] BM25 search failed: {e}")

        # Step 5: Reciprocal Rank Fusion (RRF)
        # RRF formula: score = sum(1 / (k + rank)) for each method
        # k is a constant (typically 60) to prevent high scores for top ranks
        RRF_K = 60
        combined_scores: Dict[str, Dict] = {}

        # All unique chunk IDs from both searches
        all_chunk_ids = set(semantic_results.keys()) | set(bm25_results.keys())

        for chunk_id in all_chunk_ids:
            semantic_rank = semantic_results.get(chunk_id, {}).get("rank", 1000)
            bm25_rank = bm25_results.get(chunk_id, {}).get("rank", 1000)

            # RRF scores (weighted by hybrid_alpha)
            semantic_rrf = hybrid_alpha * (1 / (RRF_K + semantic_rank))
            bm25_rrf = (1 - hybrid_alpha) * (1 / (RRF_K + bm25_rank))
            combined_rrf = semantic_rrf + bm25_rrf

            # Get document data (prefer semantic results as they have full metadata)
            if chunk_id in semantic_results:
                data = semantic_results[chunk_id]
            else:
                # Need to fetch from ChromaDB for BM25-only results
                try:
                    fetched = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
                    data = {
                        "document": fetched["documents"][0] if fetched["documents"] else "",
                        "metadata": fetched["metadatas"][0] if fetched["metadatas"] else {},
                        "distance": 0
                    }
                except Exception:
                    continue

            combined_scores[chunk_id] = {
                "rrf_score": combined_rrf,
                "semantic_rank": semantic_rank if chunk_id in semantic_results else None,
                "bm25_rank": bm25_rank if chunk_id in bm25_results else None,
                "document": data.get("document", ""),
                "metadata": data.get("metadata", {}),
                "distance": data.get("distance", 0)
            }

        # Step 6: Sort by RRF score and take top results
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True
        )[:max_results]

        # Step 7: Normalize scores to 0-1 range
        if sorted_results:
            raw_scores = [data["rrf_score"] for _, data in sorted_results]
            max_score = max(raw_scores)
            min_score = min(raw_scores)
            score_range = max_score - min_score

        # Step 8: Format results
        formatted = []
        for chunk_id, data in sorted_results:
            metadata = data["metadata"]

            # Determine search method used
            if data["semantic_rank"] and data["bm25_rank"]:
                search_method = "hybrid"
            elif data["semantic_rank"]:
                search_method = "semantic"
            else:
                search_method = "keyword"

            # Normalize score to 0-1
            if score_range > 0:
                normalized_score = (data["rrf_score"] - min_score) / score_range
            else:
                normalized_score = 1.0 if sorted_results else 0.0

            formatted.append({
                "content": data["document"],
                "source": metadata.get("source", ""),
                "filename": metadata.get("filename", ""),
                "category": metadata.get("category", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "score": round(normalized_score, 4),
                "raw_rrf_score": round(data["rrf_score"], 6),
                "semantic_rank": data["semantic_rank"],
                "bm25_rank": data["bm25_rank"],
                "search_method": search_method,
                "keywords": metadata.get("keywords", "").split(","),
                "routed_by": routed_category if routed_category else "none"
            })

        # Store in cache
        self.query_cache.put(query_text, max_results, category_filter, hybrid_alpha, formatted)

        return formatted

    def _route_by_keywords(self, query_text: str) -> Optional[str]:
        """
        Weighted keyword routing with word boundaries.

        Uses regex word boundaries to avoid false positives (e.g., "RAPID" matching "api").
        Scores each category by number of keyword matches and returns the highest scoring one.
        """
        query_lower = query_text.lower()

        # Score each category by counting keyword matches with word boundaries
        category_scores: Dict[str, Tuple[int, List[str]]] = {}

        for category, keywords in config.keyword_routes.items():
            matches = []
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Use word boundaries for single words, exact match for phrases
                if ' ' in keyword_lower:
                    # Phrase: use exact substring match (phrases are less ambiguous)
                    if keyword_lower in query_lower:
                        matches.append(keyword)
                else:
                    # Single word: use word boundary to avoid false positives
                    # e.g., "api" should not match "RAPID"
                    pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                    if re.search(pattern, query_lower):
                        matches.append(keyword)

            if matches:
                category_scores[category] = (len(matches), matches)

        if not category_scores:
            return None

        # Return category with highest score (most keyword matches)
        best_category = max(category_scores.keys(), key=lambda c: category_scores[c][0])
        return best_category

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
            "unique_content_hashes": len(self._chunk_hashes),
            "categories": self.list_categories(),
            "supported_formats": config.supported_formats,
            "embedding_model": config.ollama_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "query_cache": self.query_cache.stats(),
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
def search_knowledge(
    query: str,
    max_results: int = 5,
    category: str = None,
    hybrid_alpha: float = 0.3
) -> str:
    """
    Hybrid search combining semantic search + BM25 keyword search.

    Args:
        query: Search query text
        max_results: Maximum number of results (default: 5, max: 20)
        category: Optional category filter (security, ctf, logscale, development, general, redteam, blueteam)
        hybrid_alpha: Balance between semantic and keyword search (0.0 = keyword only, 1.0 = semantic only, default: 0.3)

    Returns:
        JSON string with search results including content, source, relevance score, and search method used
    """
    # Input validation
    if not query or not query.strip():
        return json.dumps({
            "status": "error",
            "message": "Query cannot be empty"
        })

    # Clamp max_results to valid range
    max_results = max(1, min(max_results or 5, config.max_results))

    # Clamp hybrid_alpha to valid range
    hybrid_alpha = max(0.0, min(hybrid_alpha if hybrid_alpha is not None else 0.5, 1.0))

    # Validate category if provided
    valid_categories = list(config.keyword_routes.keys()) + ["general"]
    if category and category not in valid_categories:
        return json.dumps({
            "status": "error",
            "message": f"Invalid category '{category}'. Valid categories: {', '.join(valid_categories)}"
        })

    orchestrator = get_orchestrator()
    results = orchestrator.query(
        query.strip(),
        max_results=max_results,
        category_filter=category,
        hybrid_alpha=hybrid_alpha
    )

    if not results:
        return json.dumps({
            "status": "no_results",
            "query": query,
            "message": "No relevant documents found. Try a different query or check if documents are indexed."
        })

    cache_stats = orchestrator.query_cache.stats()
    return json.dumps({
        "status": "success",
        "query": query,
        "hybrid_alpha": hybrid_alpha,
        "result_count": len(results),
        "cache_hit_rate": cache_stats["hit_rate"],
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
def reindex_documents(force: bool = False, full_rebuild: bool = False) -> str:
    """
    Index or reindex all documents in the knowledge base.

    Args:
        force: If True, smart reindex (detects new/changed/deleted files + rebuilds BM25).
               FAST — only re-embeds new or changed documents.
        full_rebuild: If True, nuclear rebuild (deletes everything and re-embeds ALL chunks).
                      EXTREMELY SLOW — only use if ChromaDB is corrupted or embedding model changed.

    Returns:
        JSON string with indexing statistics
    """
    orchestrator = get_orchestrator()

    if full_rebuild:
        stats = orchestrator.nuclear_rebuild()
        operation = "nuclear_rebuild"
    elif force:
        stats = orchestrator.reindex_all()
        operation = "smart_reindex"
    else:
        stats = orchestrator.index_all()
        operation = "incremental_index"

    return json.dumps({
        "status": "success",
        "operation": operation,
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
