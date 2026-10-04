"""Failure journal bounds and containment must hold before any backend write."""

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mcp_server import server


def journal(orch, payload):
    path = orch._metadata_file.with_name("document_repairs.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def payload(sources):
    return {"version": 1, "collection": server.config.collection_name, "sources": sources}


@pytest.mark.parametrize("invalid", ["escape", "bad_id", "many_versions", "many_sources", "root", "wrong_collection"])
def test_invalid_journal_rejected_before_vector_changes(isolated_orchestrator, monkeypatch, invalid):
    orch = isolated_orchestrator
    source = str((server.config.documents_dir / "one.txt").resolve())
    sources = {source: ["a" * 16]}
    if invalid == "escape":
        sources = {str(server.config.documents_dir.parent / "outside.txt"): ["a" * 16]}
    elif invalid == "bad_id":
        sources[source] = ["../../some-other-document"]
    elif invalid == "many_versions":
        sources[source] = [f"{i:016x}" for i in range(17)]
    elif invalid == "many_sources":
        sources = {str(server.config.documents_dir / f"{i}.txt"): ["a" * 16] for i in range(1025)}
    elif invalid == "root":
        sources = {str(server.config.documents_dir.resolve()): ["a" * 16]}
    value = payload(sources)
    if invalid == "wrong_collection":
        value["collection"] = "unrelated-collection"
    journal(orch, value)
    delete = Mock(wraps=orch.collection.delete)
    monkeypatch.setattr(orch.collection, "delete", delete)
    with pytest.raises(ValueError):
        orch.index_all()
    delete.assert_not_called()


def test_oversize_journal_rejected_before_open(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    path = journal(orch, payload({}))
    path.write_bytes(b" " * (orch._DOCUMENT_REPAIR_LIMIT + 1))
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        assert self != path, "oversized journal must not be opened"
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="1 MiB"):
        orch._load_document_repairs()


def test_journal_read_is_bounded_after_growth(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    path = journal(orch, payload({}))
    original_stat, original_open = Path.stat, Path.open

    class GrowingJournal(io.BytesIO):
        def read(self, size=-1):
            assert size == orch._DOCUMENT_REPAIR_LIMIT + 1
            return super().read(size)

    def tiny_stat(self, *args, **kwargs):
        if self == path:
            return SimpleNamespace(st_size=10)
        return original_stat(self, *args, **kwargs)

    def growing_open(self, *args, **kwargs):
        if self == path:
            return GrowingJournal(b" " * (orch._DOCUMENT_REPAIR_LIMIT + 20))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", tiny_stat)
    monkeypatch.setattr(Path, "open", growing_open)
    with pytest.raises(ValueError, match="1 MiB"):
        orch._load_document_repairs()


def test_valid_id_cannot_delete_another_source(isolated_orchestrator):
    orch = isolated_orchestrator
    orch.add_document_from_content("Healthy unrelated synthetic orchard", "healthy.txt", "general")
    doc_id = next(iter(orch._indexed_docs))
    missing = str((server.config.documents_dir / "missing.txt").resolve())
    journal(orch, payload({missing: [doc_id]}))
    assert orch.index_all()["errors"] == 0
    assert orch.collection.count() == 1
    assert doc_id in orch._indexed_docs


def test_healthy_incremental_does_not_enumerate_backend(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    orch.add_document_from_content("Healthy synthetic orchard", "one.txt", "general")
    get = Mock(side_effect=AssertionError("healthy scan must not inspect Chroma rows"))
    parse = Mock(side_effect=AssertionError("unchanged document must not be parsed"))
    monkeypatch.setattr(orch.collection, "get", get)
    monkeypatch.setattr(orch.parser, "parse_file", parse)
    assert orch.index_all()["skipped"] == 1
    get.assert_not_called()
    parse.assert_not_called()


def test_restart_with_pending_repair_does_not_trust_ready_fts(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    orch.add_document_from_content("Original synthetic orchard", "one.txt", "general")
    monkeypatch.setattr(server.config, "fts5_enabled", True)
    orch._initialize_fts5_dispatch()
    orch._fts5_migration_thread.join(5)
    assert orch.fts5_index.is_ready()
    source, doc_id = next(iter(orch._source_to_docid.items()))
    journal(orch, payload({source: [doc_id]}))
    orch.fts5_index.close()
    monkeypatch.setattr(server, "FastEmbedEmbeddings", lambda: orch.embed_fn)
    monkeypatch.setattr(server.KnowledgeOrchestrator, "_init_chroma_client", lambda self: orch.chroma_client)
    restarted = server.KnowledgeOrchestrator()
    try:
        assert not restarted.fts5_index.is_ready()
        assert restarted.index_all()["errors"] == 0
        restarted._fts5_migration_thread.join(5)
        assert restarted.fts5_index.is_ready()
        assert restarted.fts5_index.search("Original", 5)
    finally:
        restarted.fts5_index.close()
