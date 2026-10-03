"""Standalone rebuild must use the requested existing database and stream it."""

from unittest.mock import Mock

import pytest

from mcp_server.config import config
from mcp_server.fts5_index import Fts5LexicalIndex, Fts5MigrationError
from scripts import build_fts5_index as builder


class PagedCollection:
    def __init__(self, count=1201):
        self.total = count
        self.requests = []

    def count(self):
        return self.total

    def get(self, *, include, limit, offset):
        self.requests.append((limit, offset))
        indices = range(offset, min(self.total, offset + limit))
        return {
            "ids": [f"id-{index}" for index in indices],
            "documents": [f"content {index}" for index in indices],
            "metadatas": [{"filename": f"{index}.md", "category": "docs"} for index in indices],
        }


def test_chunk_iteration_is_lazy_bounded_and_aligned():
    collection = PagedCollection()
    chunks = builder._iter_chroma_chunks(collection)
    assert collection.requests == []
    assert next(chunks) == ("id-0", "content 0", "0.md", "docs")
    assert collection.requests == [(500, 0)]
    remainder = list(chunks)
    assert len(remainder) == 1200
    assert remainder[-1] == ("id-1200", "content 1200", "1200.md", "docs")
    assert collection.requests == [(500, 0), (500, 500), (500, 1000)]


def test_cli_data_directory_selects_existing_collection_without_creating_it(tmp_path, monkeypatch):
    selected = tmp_path / "selected"
    database = selected / "chroma_db" / "chroma.sqlite3"
    database.parent.mkdir(parents=True)
    database.touch()
    monkeypatch.setattr(config, "chroma_dir", tmp_path / "unrelated")
    client = Mock()
    persistent_client = Mock(return_value=client)
    monkeypatch.setattr("chromadb.PersistentClient", persistent_client)
    builder._open_collection(selected)
    persistent_client.assert_called_once_with(path=str(selected / "chroma_db"))
    client.get_collection.assert_called_once_with(name=config.collection_name, embedding_function=None)
    client.get_or_create_collection.assert_not_called()


@pytest.mark.parametrize("explicit_override", [False, True])
def test_cli_respects_configured_chroma_unless_data_directory_is_explicit(tmp_path, monkeypatch, explicit_override):
    configured_data = tmp_path / "metadata"
    configured_chroma = tmp_path / "custom-vector-store"
    override = tmp_path / "override"
    selected_data = override if explicit_override else configured_data
    selected_chroma = override / "chroma_db" if explicit_override else configured_chroma
    database = selected_chroma / "chroma.sqlite3"
    database.parent.mkdir(parents=True)
    database.touch()
    monkeypatch.setattr(config, "data_dir", configured_data)
    monkeypatch.setattr(config, "chroma_dir", configured_chroma)
    persistent_client = Mock()
    monkeypatch.setattr("chromadb.PersistentClient", persistent_client)
    index = Mock()
    open_index = Mock(return_value=index)
    monkeypatch.setattr(builder, "_open_index", open_index)
    migrate = Mock()
    monkeypatch.setattr(builder, "_run_migration_sync", migrate)

    argv = ["--data-dir", str(override)] if explicit_override else []
    assert builder.main(argv) == 0

    persistent_client.assert_called_once_with(path=str(selected_chroma))
    persistent_client.return_value.get_collection.assert_called_once_with(
        name=config.collection_name, embedding_function=None
    )
    open_index.assert_called_once_with(selected_data)
    migrate.assert_called_once_with(index, persistent_client.return_value.get_collection.return_value, False)
    index.close.assert_called_once()


def test_missing_source_database_is_not_created(tmp_path, monkeypatch):
    persistent_client = Mock()
    monkeypatch.setattr("chromadb.PersistentClient", persistent_client)
    with pytest.raises(FileNotFoundError, match="No existing ChromaDB"):
        builder._open_collection(tmp_path / "missing")
    persistent_client.assert_not_called()
    assert not (tmp_path / "missing").exists()


def test_sync_rebuild_replays_rows_without_duplicates(tmp_path):
    index = Fts5LexicalIndex(tmp_path / "fts5.db", tmp_path / "state.json")
    try:
        collection = PagedCollection(201)
        builder._run_migration_sync(index, collection, False)
        builder._run_migration_sync(index, collection, False)
        assert index.count() == 201
        assert index.state.read()["docs_indexed"] == 201
    finally:
        index.close()


def test_sync_rebuild_propagates_failure_instead_of_reporting_success(tmp_path):
    index = Fts5LexicalIndex(tmp_path / "fts5.db", tmp_path / "state.json")
    collection = PagedCollection(201)
    collection.get = Mock(side_effect=OSError("source unavailable"))
    try:
        with pytest.raises(Fts5MigrationError, match="source unavailable"):
            builder._run_migration_sync(index, collection, False)
        assert index.state.read()["status"] == "failed"
    finally:
        index.close()
