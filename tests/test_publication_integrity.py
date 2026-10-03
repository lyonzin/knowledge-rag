"""Real collection regressions for publication, source versions and partial pruning."""

import threading
from concurrent.futures import ThreadPoolExecutor

from mcp_server import server


def seed(orch, name="one.txt", content="original production content"):
    path = server.config.documents_dir / name
    path.write_text(content, encoding="utf-8")
    assert orch.index_all()["errors"] == 0
    return path


def test_swap_waits_for_inflight_query_before_retiring_collection(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    path = seed(orch)
    monkeypatch.setattr(server.config, "reranker_enabled", False)
    entered, release, validated = threading.Event(), threading.Event(), threading.Event()
    old_id = orch.collection.id
    query = type(orch.collection).query
    validate = orch._enforce_staging_validation

    def blocked_query(collection, **kwargs):
        if collection.id == old_id:
            entered.set()
            assert release.wait(timeout=5)
        return query(collection, **kwargs)

    def validation(*args):
        validate(*args)
        validated.set()

    monkeypatch.setattr(type(orch.collection), "query", blocked_query)
    monkeypatch.setattr(orch, "_enforce_staging_validation", validation)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(orch.query, "original", 1, hybrid_alpha=1.0)
        try:
            assert entered.wait(timeout=5)
            path.write_text("new published content", encoding="utf-8")
            writer = pool.submit(orch.nuclear_rebuild)
            assert validated.wait(timeout=5)
            # The writer has finished inference, but this collection is still leased.
            assert orch.chroma_client.get_collection(server.config.collection_name).id == old_id
            assert not writer.done()
        finally:
            release.set()
        result = reader.result(timeout=5)
        assert result and "original production" in result[0]["content"]
        assert writer.result(timeout=5)["indexed"] == 1
    assert orch.collection.id != old_id
    assert "new published" in orch.query("original", 1, hybrid_alpha=1.0)[0]["content"]


def test_parallel_readers_complete_while_staging_inference_is_blocked(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    path = seed(orch)
    path.write_text("replacement production content", encoding="utf-8")
    monkeypatch.setattr(server.config, "reranker_enabled", False)
    staging_entered, release_staging = threading.Event(), threading.Event()
    simultaneous_readers = threading.Barrier(2, timeout=5)
    old_id = orch.collection.id
    upsert, query = type(orch.collection).upsert, type(orch.collection).query

    def blocked_staging(collection, **kwargs):
        if collection.id != old_id:
            staging_entered.set()
            assert release_staging.wait(timeout=10)
        return upsert(collection, **kwargs)

    def parallel_query(collection, **kwargs):
        if collection.id == old_id:
            simultaneous_readers.wait()
        return query(collection, **kwargs)

    monkeypatch.setattr(type(orch.collection), "upsert", blocked_staging)
    monkeypatch.setattr(type(orch.collection), "query", parallel_query)
    with ThreadPoolExecutor(max_workers=3) as pool:
        writer = pool.submit(orch.nuclear_rebuild)
        try:
            assert staging_entered.wait(timeout=5)
            readers = [pool.submit(orch.query, text, 1, hybrid_alpha=0.5) for text in ("original", "production")]
            for reader in readers:
                assert "original production" in reader.result(timeout=5)[0]["content"]
            assert not writer.done()
            assert orch.collection.id == old_id
        finally:
            release_staging.set()
        assert writer.result(timeout=5)["indexed"] == 1


def test_source_edit_during_embedding_is_indexed_on_next_scan(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    path = server.config.documents_dir / "one.txt"
    path.write_text("content captured by the parser", encoding="utf-8")
    upsert = type(orch.collection).upsert
    edited = False

    def edit_after_inference(collection, **kwargs):
        nonlocal edited
        result = upsert(collection, **kwargs)
        if not edited:
            edited = True
            path.write_text("newer source edited while ONNX was busy", encoding="utf-8")
        return result

    monkeypatch.setattr(type(orch.collection), "upsert", edit_after_inference)
    assert orch.index_all()["indexed"] == 1
    assert orch.collection.get(include=["documents"])["documents"] == ["content captured by the parser"]
    stats = orch.index_all()
    assert stats["updated"] == 1 and stats["skipped"] == 0
    assert orch.collection.get(include=["documents"])["documents"] == [path.read_text(encoding="utf-8")]


def test_orphan_failure_preserves_failed_doc_and_persists_successes(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    first = seed(orch)
    second = seed(orch, "two.txt", "second document to retain")
    first_id = orch._source_to_docid[str(first)]
    second_id = orch._source_to_docid[str(second)]
    remove = orch._remove_document_chunks

    def reject_second(doc_id):
        if doc_id == second_id:
            raise RuntimeError("backend temporarily unavailable")
        return remove(doc_id)

    monkeypatch.setattr(orch, "_remove_document_chunks", reject_second)
    first.unlink()
    second.unlink()
    stats = orch.index_all()
    assert stats["deleted"] == 1 and stats["errors"] == 1
    assert first_id not in orch._indexed_docs and second_id in orch._indexed_docs
    assert orch._load_metadata() == orch._indexed_docs
    assert orch.collection.count() == 1


def test_ttl_cleans_retired_backup_when_production_exists(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    now = 2_000_000_000
    monkeypatch.setattr(server.time, "time", lambda: now)
    stale = f"{server.config.collection_name}__old_{now - orch._STAGING_TTL_SECONDS - 1}"
    orch.chroma_client.create_collection(stale)
    stats = orch._cleanup_stale_staging_collections()
    assert stats["removed"] == 1
    assert stale not in {collection.name for collection in orch.chroma_client.list_collections()}


def test_ttl_preserves_backup_when_production_name_is_missing(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    now = 2_000_000_000
    monkeypatch.setattr(server.time, "time", lambda: now)
    backup = f"{server.config.collection_name}__old_{now - orch._STAGING_TTL_SECONDS - 1}"
    orch.collection.modify(name=backup)
    orch._cleanup_stale_staging_collections()
    assert backup in {collection.name for collection in orch.chroma_client.list_collections()}
