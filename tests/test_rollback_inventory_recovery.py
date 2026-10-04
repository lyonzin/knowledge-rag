"""Double failures must remain actionable across incremental scans and restarts."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from mcp_server import server


@pytest.fixture
def committed_pair(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    monkeypatch.setattr(server.config, "reranker_enabled", False)
    orch.add_document_from_content("Original synthetic orchard", "affected/one.txt", "general")
    orch.add_document_from_content("Unaffected synthetic garden", "healthy/two.txt", "general")
    return orch


def rows(orch):
    result = orch.collection.get(include=["documents", "metadatas", "embeddings"])
    return {
        key: (text, meta, vector.tolist())
        for key, text, meta, vector in zip(
            result["ids"], result["documents"], result["metadatas"], result["embeddings"]
        )
    }


def scoped(orch, folder="."):
    return orch.query(
        "synthetic orchard", max_results=5, hybrid_alpha=1.0, search_method="hybrid", include_folders=[folder]
    )


def inject_double_failure(orch, patcher, failure):
    commit_error, rollback_error = OSError("metadata commit failed"), OSError("rollback failed")
    restore, delete = orch._restore_document_chunks, orch.collection.delete
    save = Mock(side_effect=commit_error)
    patcher.setattr(orch, "_save_metadata", save)
    if failure == "delete_before":

        def failed_delete(*args, **kwargs):
            if save.called and "ids" in kwargs:
                raise rollback_error
            return delete(*args, **kwargs)

        patcher.setattr(orch.collection, "delete", failed_delete)
    else:

        def failed_restore(snapshot):
            if failure == "restore_after":
                restore(snapshot)
            raise rollback_error

        patcher.setattr(orch, "_restore_document_chunks", failed_restore)
    return commit_error, rollback_error


def mutate(orch, operation):
    if operation == "remove":
        return orch.remove_document_by_path("affected/one.txt", delete_file=True)
    if operation == "update":
        return orch.update_document_content("affected/one.txt", "Replacement synthetic orchard")
    return orch.add_document_from_content("Replacement synthetic orchard", "affected/one.txt", "general")


CASES = [
    (operation, failure)
    for operation in ("remove", "update", "add")
    for failure in ("restore_before", "restore_after", "delete_before")
    if operation != "remove" or failure != "delete_before"
]


@pytest.mark.parametrize("operation,failure", CASES)
@pytest.mark.parametrize("restart", [False, True])
def test_failed_rollback_repairs_only_affected_source(committed_pair, monkeypatch, operation, failure, restart):
    orch = committed_pair
    original_rows, original_catalog = rows(orch), deepcopy(orch._indexed_docs)
    original_lookup = dict(orch._source_to_docid)
    original_metadata = orch._metadata_file.read_bytes()
    path = server.config.documents_dir / "affected/one.txt"
    original_bytes = path.read_bytes()
    with monkeypatch.context() as injection:
        commit_error, rollback_error = inject_double_failure(orch, injection, failure)
        with pytest.raises(OSError) as caught:
            mutate(orch, operation)
    assert caught.value is rollback_error
    assert caught.value.__context__ is commit_error
    assert path.read_bytes() == original_bytes
    assert orch._metadata_file.read_bytes() == original_metadata
    assert orch._indexed_docs == original_catalog
    assert orch._source_to_docid == original_lookup
    # Uncommitted replacement IDs are intentionally absent from committed scope.
    assert len(scoped(orch)) == (1 if failure == "delete_before" else orch.collection.count())
    if failure == "restore_after":
        assert rows(orch) == original_rows
        assert "Original" in scoped(orch, "affected")[0]["content"]
    else:
        assert rows(orch) != original_rows
    marker = orch._metadata_file.with_name("document_repairs.json")
    assert marker.is_file()

    if restart:
        monkeypatch.setattr(server, "FastEmbedEmbeddings", lambda: orch.embed_fn)
        monkeypatch.setattr(server.KnowledgeOrchestrator, "_init_chroma_client", lambda self: orch.chroma_client)
        orch = server.KnowledgeOrchestrator()
    parse = Mock(wraps=orch.parser.parse_file)
    monkeypatch.setattr(orch.parser, "parse_file", parse)
    result = orch.index_all(resume_state={"doc_ids": list(original_catalog)})
    assert result["errors"] == 0
    assert result["skipped"] == 1
    assert parse.call_count == 1
    assert rows(orch) == original_rows
    assert not marker.exists()
    assert len(orch._indexed_docs) == 2
    assert len(scoped(orch)) == 2
    assert "Original" in scoped(orch, "affected")[0]["content"]
    parse.reset_mock()
    assert orch.index_all()["skipped"] == 2
    parse.assert_not_called()


@pytest.mark.parametrize("restart", [False, True])
def test_failed_new_add_repairs_orphan_without_source(committed_pair, monkeypatch, restart):
    orch = committed_pair
    original_rows, original_catalog = rows(orch), deepcopy(orch._indexed_docs)
    with monkeypatch.context() as injection:
        inject_double_failure(orch, injection, "delete_before")
        with pytest.raises(OSError, match="rollback failed"):
            orch.add_document_from_content("Uncommitted synthetic orchard", "new.txt", "general")
    assert not (server.config.documents_dir / "new.txt").exists()
    assert orch._indexed_docs == original_catalog
    assert orch.collection.count() == 3
    if restart:
        monkeypatch.setattr(server, "FastEmbedEmbeddings", lambda: orch.embed_fn)
        monkeypatch.setattr(server.KnowledgeOrchestrator, "_init_chroma_client", lambda self: orch.chroma_client)
        orch = server.KnowledgeOrchestrator()
    parse = Mock(wraps=orch.parser.parse_file)
    monkeypatch.setattr(orch.parser, "parse_file", parse)
    result = orch.index_all()
    assert result["errors"] == 0
    assert rows(orch) == original_rows
    parse.assert_not_called()
    assert not orch._metadata_file.with_name("document_repairs.json").exists()


def test_repair_marker_survives_failed_repair_commit(committed_pair, monkeypatch):
    orch = committed_pair
    with monkeypatch.context() as injection:
        inject_double_failure(orch, injection, "restore_before")
        with pytest.raises(OSError):
            mutate(orch, "remove")
    marker = orch._metadata_file.with_name("document_repairs.json")
    with monkeypatch.context() as injection:
        injection.setattr(orch, "_save_metadata", Mock(side_effect=OSError("still unavailable")))
        with pytest.raises(OSError, match="still unavailable"):
            orch.index_all()
    assert marker.exists()
    assert orch.index_all()["errors"] == 0
    assert not marker.exists()


@pytest.mark.parametrize("failure", ["restore_before", "restore_after", "delete_before"])
def test_lexical_indices_rebuild_after_uncertain_rollback(committed_pair, monkeypatch, failure):
    orch = committed_pair
    monkeypatch.setattr(server.config, "fts5_enabled", True)
    orch._initialize_fts5_dispatch()
    orch._fts5_migration_thread.join(5)
    assert orch.fts5_index.is_ready()
    orch._ensure_bm25_index()
    assert orch._bm25_initialized
    with monkeypatch.context() as injection:
        inject_double_failure(orch, injection, failure)
        with pytest.raises(OSError):
            mutate(orch, "update")
    assert not orch.fts5_index.is_ready()
    assert not orch._bm25_initialized
    assert orch.index_all()["errors"] == 0
    orch._fts5_migration_thread.join(5)
    assert orch.fts5_index.is_ready()
    assert orch.fts5_index.count() == orch.collection.count() == 2
    assert orch.fts5_index.search("Original", 5)
    assert not orch.fts5_index.search("Replacement", 5)
    orch._ensure_bm25_index()
    assert "Original synthetic orchard" in orch.bm25_index.corpus
    assert "Replacement synthetic orchard" not in orch.bm25_index.corpus


def test_repair_tracks_versions_not_number_of_chunks(committed_pair, monkeypatch):
    import json

    orch = committed_pair
    orch.parser.chunk_size, orch.parser.chunk_overlap = 60, 0
    old_text = "\n\n".join(f"Original orchard section {i} with synthetic fruit." for i in range(40))
    orch.update_document_content("affected/one.txt", old_text)
    original = rows(orch)
    assert len(original) > 16
    # Old metadata may predate the cached chunk count; recovery does not depend on it.
    for info in orch._indexed_docs.values():
        info.pop("chunks", None)
    orch._save_metadata()
    with monkeypatch.context() as injection:
        inject_double_failure(orch, injection, "delete_before")
        with pytest.raises(OSError):
            orch.update_document_content("affected/one.txt", old_text.replace("Original", "Replacement"))
    marker = orch._metadata_file.with_name("document_repairs.json")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert len(next(iter(payload["sources"].values()))) == 2
    assert orch.index_all()["errors"] == 0
    assert rows(orch) == original


def test_unwritable_marker_preserves_errors_and_in_memory_repair(committed_pair, monkeypatch):
    orch = committed_pair
    original = rows(orch)
    with monkeypatch.context() as injection:
        _, rollback_error = inject_double_failure(orch, injection, "restore_before")
        injection.setattr(orch, "_save_document_repairs", Mock(side_effect=OSError("disk unavailable")))
        with pytest.raises(OSError) as caught:
            mutate(orch, "remove")
    assert caught.value is rollback_error
    assert any("reindex_documents(force=True)" in note for note in caught.value.__notes__)
    assert any("reindex_documents(full_rebuild=True)" in note for note in caught.value.__notes__)
    assert orch._pending_document_repairs
    assert orch.index_all()["errors"] == 0
    assert rows(orch) == original


@pytest.mark.parametrize("fail_publish", [False, True])
def test_full_rebuild_keeps_journal_until_promotion_commit(committed_pair, monkeypatch, fail_publish):
    orch = committed_pair
    original = rows(orch)
    with monkeypatch.context() as injection:
        inject_double_failure(orch, injection, "delete_before")
        with pytest.raises(OSError):
            mutate(orch, "update")
    marker = orch._metadata_file.with_name("document_repairs.json")
    populate = orch._populate_staging

    def check_staging(staging):
        result = populate(staging)
        assert marker.exists(), "staging is not yet a published repair"
        return result

    monkeypatch.setattr(orch, "_populate_staging", check_staging)
    if fail_publish:
        with monkeypatch.context() as injection:
            injection.setattr(orch, "_save_metadata", Mock(side_effect=OSError("publication unavailable")))
            with pytest.raises(OSError, match="publication unavailable"):
                orch.nuclear_rebuild()
        assert marker.exists()
        return
    assert orch.nuclear_rebuild()["errors"] == 0
    assert not marker.exists()
    assert rows(orch) == original


def test_full_rebuild_cleanup_failure_does_not_undo_publication(committed_pair, monkeypatch, capsys):
    orch = committed_pair
    original = rows(orch)
    original_collection_id = orch.collection.id
    with monkeypatch.context() as injection:
        inject_double_failure(orch, injection, "delete_before")
        with pytest.raises(OSError):
            mutate(orch, "update")
    marker = orch._metadata_file.with_name("document_repairs.json")
    monkeypatch.setattr(orch, "_save_document_repairs", Mock(side_effect=OSError("journal cleanup denied")))
    assert orch.nuclear_rebuild()["errors"] == 0
    assert orch.collection.id != original_collection_id
    assert rows(orch) == original
    assert marker.exists()
    assert "Rebuild committed; post-commit repair cleanup failed" in capsys.readouterr().err
