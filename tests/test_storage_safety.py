"""Index initialization failures must preserve data and remain retryable."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

import mcp_server.server as server


@pytest.mark.parametrize("message", ["database is locked", "permission denied", "embedding function conflict"])
def test_collection_open_failure_preserves_database_and_collections(tmp_path, monkeypatch, message):
    database = tmp_path / "chroma.sqlite3"
    database.write_bytes(b"existing database sentinel")
    segment = tmp_path / "segment"
    segment.mkdir()
    vectors = segment / "vectors.bin"
    vectors.write_bytes(b"existing vectors sentinel")
    monkeypatch.setattr(server.config, "chroma_dir", tmp_path)
    orchestrator = object.__new__(server.KnowledgeOrchestrator)
    orchestrator.embed_fn = Mock()
    orchestrator.chroma_client = Mock()
    failure = RuntimeError(message)
    orchestrator.chroma_client.get_or_create_collection.side_effect = failure

    with pytest.raises(RuntimeError, match="Existing index preserved") as caught:
        orchestrator._safe_get_collection()

    assert caught.value.__cause__ is failure
    assert database.read_bytes() == b"existing database sentinel"
    assert vectors.read_bytes() == b"existing vectors sentinel"
    orchestrator.chroma_client.delete_collection.assert_not_called()
    assert orchestrator.chroma_client.get_or_create_collection.call_count == 1


def test_bm25_retry_does_not_accumulate_duplicate_documents(monkeypatch):
    orchestrator = object.__new__(server.KnowledgeOrchestrator)
    orchestrator.collection = Mock()
    orchestrator.collection.count.return_value = 2
    orchestrator.collection.get.return_value = {"ids": ["a", "b"], "documents": ["alpha", "beta"]}
    orchestrator.bm25_index = server.BM25Index()
    orchestrator._bm25_initialized = False
    build = orchestrator.bm25_index.build_index
    monkeypatch.setattr(orchestrator.bm25_index, "build_index", Mock(side_effect=MemoryError("injected OOM")))
    orchestrator._ensure_bm25_index()
    assert not orchestrator._bm25_initialized
    monkeypatch.setattr(orchestrator.bm25_index, "build_index", build)
    orchestrator._ensure_bm25_index()
    assert orchestrator._bm25_initialized
    assert orchestrator.bm25_index.corpus_ids == ["a", "b"]


@pytest.mark.parametrize("fail", [False, True])
def test_concurrent_reranker_loading_initializes_only_once(monkeypatch, fail):
    start = threading.Barrier(8)
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        time.sleep(0.05)
        if fail:
            raise RuntimeError("offline")
        return object()

    monkeypatch.setattr(server, "TextCrossEncoder", load)
    reranker = server.CrossEncoderReranker()

    def ensure(_):
        start.wait(timeout=3)
        return reranker._ensure_model()

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(ensure, range(8)))
    assert calls and len(calls) == 1
    assert statuses == [not fail] * 8
