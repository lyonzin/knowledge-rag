"""Lifecycle regressions against real Chroma; never download an embedding model."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from mcp_server import server


def seed(orch, content="initial document text"):
    path = server.config.documents_dir / "one.txt"
    path.write_text(content, encoding="utf-8")
    assert orch.index_all()["errors"] == 0
    return path


def test_unchanged_scan_never_parses_content(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    seed(orch)
    parse = Mock(side_effect=AssertionError("unchanged file was parsed"))
    monkeypatch.setattr(orch.parser, "parse_file", parse)
    stats = orch.index_all()
    assert stats["skipped"] == 1
    parse.assert_not_called()


def test_parse_failure_preserves_existing_index(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    path = seed(orch)
    before = orch.collection.get(include=["documents"])
    path.write_text("changed inaccessible document", encoding="utf-8")
    monkeypatch.setattr(orch.parser, "parse_file", Mock(side_effect=PermissionError("denied")))
    stats = orch.index_all()
    assert stats["errors"] == 1 and stats["deleted"] == 0
    assert orch.collection.get(include=["documents"]) == before
    assert len(orch._indexed_docs) == 1


def test_force_replaces_embeddings_for_unchanged_ids(isolated_orchestrator):
    orch = isolated_orchestrator
    seed(orch)
    before_ids = orch.collection.get()["ids"]
    orch.embed_fn.value = 9.0
    stats = orch.index_all(force=True)
    after = orch.collection.get(include=["embeddings"])
    assert stats["updated"] == 1 and stats["errors"] == 0
    assert after["ids"] == before_ids
    assert np.all(after["embeddings"][:, 0] == 9.0)


def test_force_changed_file_removes_old_chunk_ids(isolated_orchestrator):
    orch = isolated_orchestrator
    path = seed(orch)
    old_ids = set(orch.collection.get()["ids"])
    path.write_text("a much longer replacement for the original document", encoding="utf-8")
    assert orch.index_all(force=True)["errors"] == 0
    assert not old_ids.intersection(orch.collection.get()["ids"])
    assert orch.collection.count() == 1
    assert len(orch._indexed_docs) == 1


def test_partial_batch_failure_restores_overwritten_vectors(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    seed(orch, " ".join(f"unique-{i}" for i in range(300)))
    before = orch.collection.get(include=["documents", "embeddings"])
    monkeypatch.setattr(server.config, "batch_size", 1)
    monkeypatch.setattr(server.config, "parallel_workers", 1)
    original_upsert = type(orch.collection).upsert
    calls = 0

    def fail_second(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second batch failed")
        return original_upsert(self, **kwargs)

    monkeypatch.setattr(type(orch.collection), "upsert", fail_second)
    orch.embed_fn.value = 9.0
    assert orch.index_all(force=True)["errors"] == 1
    after = orch.collection.get(include=["documents", "embeddings"])
    assert before["ids"] == after["ids"] and before["documents"] == after["documents"]
    np.testing.assert_array_equal(before["embeddings"], after["embeddings"])


def test_embedding_failure_restores_old_file_vectors_metadata(isolated_orchestrator):
    orch = isolated_orchestrator
    path = seed(orch)
    before = orch.collection.get(include=["documents", "embeddings"])
    persisted = orch._metadata_file.read_bytes()
    orch.embed_fn.fail = True
    with pytest.raises(RuntimeError, match="embedding failure"):
        orch.update_document_content("one.txt", "updated document that fails embedding")
    after = orch.collection.get(include=["documents", "embeddings"])
    assert path.read_text(encoding="utf-8") == "initial document text"
    assert before["ids"] == after["ids"] and before["documents"] == after["documents"]
    np.testing.assert_array_equal(before["embeddings"], after["embeddings"])
    assert orch._metadata_file.read_bytes() == persisted
    assert not list(path.parent.glob(".rag-*"))


def test_metadata_write_failure_rolls_back_crud(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    path = seed(orch)
    before = orch.collection.get(include=["documents"])
    old_metadata = dict(orch._indexed_docs)
    monkeypatch.setattr(orch, "_save_metadata", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        orch.update_document_content("one.txt", "replacement body")
    assert path.read_text(encoding="utf-8") == "initial document text"
    assert orch.collection.get(include=["documents"]) == before
    assert orch._indexed_docs == old_metadata


def test_delete_backend_failure_keeps_metadata(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    path = seed(orch)
    old_metadata = dict(orch._indexed_docs)
    monkeypatch.setattr(type(orch.collection), "delete", Mock(side_effect=RuntimeError("locked")))
    with pytest.raises(RuntimeError, match="locked"):
        orch.remove_document_by_path("one.txt", delete_file=True)
    assert path.exists()
    assert orch._indexed_docs == old_metadata


def test_delete_file_failure_is_reported(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    path = seed(orch)
    unlink = Path.unlink

    def fail_source_only(self, *args, **kwargs):
        if self == path:
            raise PermissionError("read only")
        return unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_source_only)
    result = orch.remove_document_by_path("one.txt", delete_file=True)
    assert result["file_deleted"] is False
    assert "read only" in result["file_delete_error"]
    assert path.exists() and orch.collection.count() == 0


def test_failed_incremental_retains_resumable_checkpoint(isolated_orchestrator):
    orch = isolated_orchestrator
    seed(orch)
    (server.config.documents_dir / "bad.txt").write_text("new document", encoding="utf-8")
    orch.embed_fn.fail = True
    orch._reindex_progress["operation"] = "smart_reindex"
    stats = orch.index_all()
    assert stats["errors"] == 1
    payload = json.loads(orch._checkpoint_file.read_text(encoding="utf-8"))
    assert payload["operation"] == "smart_reindex"


def test_prefix_change_invalidates_resume_signature(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    before = orch._compute_config_signature()
    monkeypatch.setattr(server.config, "passage_prefix", "passage: ")
    assert orch._compute_config_signature() != before


def test_atomic_metadata_failure_preserves_previous_file(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    seed(orch)
    before = orch._metadata_file.read_bytes()
    replace = server.os.replace

    def reject_metadata(source, destination):
        if destination == orch._metadata_file:
            raise PermissionError("cannot replace metadata")
        return replace(source, destination)

    monkeypatch.setattr(server.os, "replace", reject_metadata)
    orch._indexed_docs.clear()
    with pytest.raises(PermissionError):
        orch._save_metadata()
    assert orch._metadata_file.read_bytes() == before
    assert not list(orch._metadata_file.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("indexed_doc_ids", "abc"),
        ("indexed_doc_ids", [1]),
        ("chunks_processed", True),
        ("chunks_processed", -1),
        ("operation", "unknown"),
    ],
)
def test_checkpoint_rejects_wrong_field_types(isolated_orchestrator, field, value):
    orch = isolated_orchestrator
    orch._write_checkpoint("smart_reindex", ["one"], 1)
    data = json.loads(orch._checkpoint_file.read_text(encoding="utf-8"))
    data[field] = value
    orch._checkpoint_file.write_text(json.dumps(data), encoding="utf-8")
    assert orch._load_checkpoint() is None


def test_fts5_follows_incremental_update_delete_and_dedup(isolated_orchestrator, monkeypatch, tmp_path):
    from mcp_server.fts5_index import Fts5LexicalIndex

    orch = isolated_orchestrator
    monkeypatch.setattr(server.config, "fts5_enabled", True)
    orch.fts5_index = Fts5LexicalIndex(tmp_path / "fts5.db", tmp_path / "migration.json")
    path = seed(orch, "x" * 1800)
    assert orch.fts5_index.count() == orch.collection.count()
    path.write_text("new changed text", encoding="utf-8")
    assert orch.index_all()["errors"] == 0
    assert orch.fts5_index.count() == orch.collection.count() == 1
    path.unlink()
    assert orch.index_all()["deleted"] == 1
    assert orch.fts5_index.count() == orch.collection.count() == 0
