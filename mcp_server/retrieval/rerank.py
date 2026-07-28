"""
╭─╴ CROSS-ENCODER RERANKER ╶─────────────────────────────────────╮
│                                                                │
│   Cross-encoder reranker (FastEmbed TextCrossEncoder) applied  │
│   after hybrid RRF fusion. Extracted verbatim from server.py   │
│   in the A2.1 refactor.                                        │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``

Test-patch compatibility:
    ``CrossEncoderReranker._ensure_model`` resolves the underlying
    ``fastembed.rerank.cross_encoder.TextCrossEncoder`` class via
    ``mcp_server.server.TextCrossEncoder`` at call time, not import time.
    ``tests/test_reranker_fallback.py`` patches
    ``mcp_server.server.TextCrossEncoder``; the late-bind lookup keeps
    that patching effective after the module split.
"""

from typing import Any, Dict, List

from ..config import config


def _resolve_text_cross_encoder_class():
    """Late-bind lookup for the ``TextCrossEncoder`` class.

    Tests patch ``mcp_server.server.TextCrossEncoder`` (a re-export). Routing
    the lookup through that module keeps those patches effective after the
    A2.1 module split.
    """
    try:
        from mcp_server import server as _srv

        return _srv.TextCrossEncoder
    except (ImportError, AttributeError):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder


class CrossEncoderReranker:
    """
    Cross-encoder reranker using FastEmbed's TextCrossEncoder.

    Applied after hybrid RRF fusion to re-score the top candidates
    using a cross-encoder model that sees query+document pairs jointly.
    Dramatically improves precision over bi-encoder retrieval alone.

    Model: Xenova/ms-marco-MiniLM-L-6-v2 (ONNX, ~25MB)
    """

    def __init__(self, model: str = None):
        self.model_name = model or config.reranker_model
        self._model = None  # Lazy init
        self._load_failed = False

    def _ensure_model(self) -> bool:
        """Lazy initialization of cross-encoder model"""
        if self._load_failed:
            return False
        if self._model is None:
            print(f"[INFO] Loading reranker model: {self.model_name}...")
            try:
                TextCrossEncoder = _resolve_text_cross_encoder_class()
                self._model = TextCrossEncoder(model_name=self.model_name, cache_dir=str(config.models_cache_dir))
                print("[INFO] Reranker model loaded successfully")
            except Exception as e:
                self._load_failed = True
                print(f"[WARN] Reranker unavailable, using RRF order: {e}")
                return False
        return True

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder scores.

        Args:
            query: Original search query
            documents: List of result dicts (must have 'document' key)
            top_k: Number of top results to return after reranking

        Returns:
            Reranked list of documents, sorted by cross-encoder score (top_k)
        """
        if not documents or not config.reranker_enabled:
            return documents[:top_k]

        if not self._ensure_model():
            return documents[:top_k]

        texts = [doc.get("document", "") for doc in documents]

        try:
            scores = list(self._model.rerank(query, texts))
            for doc, score in zip(documents, scores):
                doc["reranker_score"] = float(score)
            documents.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)
        except Exception as e:
            print(f"[WARN] Reranker failed, using RRF order: {e}")

        return documents[:top_k]
