"""Exercise interrupted migration, legacy schema upgrade and SQL rollback."""

import sqlite3

import pytest

from mcp_server.fts5_index import _FTS5_SCHEMA, Fts5CorruptError, Fts5LexicalIndex


@pytest.fixture
def index(tmp_path):
    lexical = Fts5LexicalIndex(tmp_path / "fts5.db", tmp_path / "state.json")
    try:
        yield lexical
    finally:
        lexical.close()


def rows(count):
    return [(f"chunk_{offset}", f"reference CODE-{offset}", "guide.md", "docs") for offset in range(count)]


def test_failed_migration_retains_committed_progress_and_replay_has_no_duplicates(index):
    corpus = rows(101)

    def interrupted():
        yield from corpus[:100]
        raise OSError("interrupted corpus read")

    index._migration_worker(interrupted, 101, 0, None)
    assert index.state.read()["status"] == "failed"
    assert index.state.read()["docs_indexed"] == 100
    assert index.count() == 100
    # A stale checkpoint can replay already committed rows, including after
    # abrupt process death between the SQLite commit and the JSON checkpoint.
    index._migration_worker(lambda: iter(corpus), 101, 0, None)
    assert index.state.read()["status"] == "complete"
    assert index.count() == 101
    assert len(index.search("CODE-1")) == 1


def test_checkpoint_write_failure_keeps_accurate_committed_count(index, monkeypatch):
    write_state = index._write_state
    fail_once = True

    def write(*args):
        nonlocal fail_once
        if args[0] == "in_progress" and args[2] == 100 and fail_once:
            fail_once = False
            raise OSError("checkpoint disk error")
        return write_state(*args)

    monkeypatch.setattr(index, "_write_state", write)
    index._migration_worker(lambda: iter(rows(101)), 101, 0, None)
    assert index.state.read()["status"] == "failed"
    assert index.state.read()["docs_indexed"] == 100
    index._migration_worker(lambda: iter(rows(101)), 101, 100, None)
    assert index.count() == 101
    assert index.state.is_complete()


def test_reopen_replays_batch_committed_before_checkpoint(tmp_path):
    database, marker = tmp_path / "fts5.db", tmp_path / "state.json"
    initial = Fts5LexicalIndex(database, marker)
    initial._populate_batch(rows(100))
    initial.close()
    restarted = Fts5LexicalIndex(database, marker)
    try:
        restarted._migration_worker(lambda: iter(rows(101)), 101, 0, None)
        assert restarted.count() == 101
        assert restarted.state.is_complete()
    finally:
        restarted.close()


def test_upsert_and_remove_keep_single_searchable_version(index):
    index.add_document("id", "old version", "old.md", "docs")
    index.add_document("id", "new version", "new.md", "docs")
    assert index.count() == 1
    assert not index.search("old")
    assert index.search("new")[0][0] == "id"
    index.remove_document("id")
    assert index.count() == 0
    index.add_document("id", "restored version", "restored.md", "docs")
    assert index.count() == 1
    assert index.search("restored")[0][0] == "id"


def test_batch_failure_rolls_back_id_map_and_search_content(index):
    index.add_document("existing", "original", "guide.md", "docs")
    with pytest.raises(sqlite3.Error):
        index._populate_batch([("existing", "changed", "guide.md", "docs"), ("invalid", object(), "guide.md", "docs")])
    assert index.count() == 1
    assert index.search("original")[0][0] == "existing"
    assert not index.search("changed")
    assert index._conn.execute("SELECT chunk_id FROM fts5_chunk_ids").fetchall() == [("existing",)]


def test_legacy_database_upgrade_preserves_latest_duplicate_and_unique_rows(tmp_path):
    database, marker = tmp_path / "fts5.db", tmp_path / "state.json"
    connection = sqlite3.connect(database)
    try:
        connection.execute(_FTS5_SCHEMA)
        connection.executemany(
            "INSERT INTO fts5_documents (chunk_id, content, filename, category) VALUES (?, ?, ?, ?)",
            [("id", "old", "a.md", "docs"), ("unique", "preserved", "b.md", "docs"), ("id", "new", "a.md", "docs")],
        )
        connection.commit()
    finally:
        connection.close()
    lexical = Fts5LexicalIndex(database, marker)
    try:
        assert lexical.count() == 2
        assert not lexical.search("old")
        assert lexical.search("new")[0][0] == "id"
        assert lexical.search("preserved")[0][0] == "unique"
        lexical.update_document("id", "updated", "a.md", "docs")
        assert lexical.count() == 2
        assert lexical.search("updated")[0][0] == "id"
    finally:
        lexical.close()


def test_closed_index_cannot_become_ready_again_from_complete_marker(index):
    index._migration_worker(lambda: iter(rows(1)), 1, 0, None)
    assert index.is_ready()
    index.close()
    assert not index.is_ready()


def test_failed_schema_upgrade_closes_sqlite_connection(tmp_path, monkeypatch):
    connections = []
    connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    def fail_upgrade(self):
        raise sqlite3.OperationalError("upgrade failed")

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    monkeypatch.setattr(Fts5LexicalIndex, "_ensure_chunk_lookup", fail_upgrade)
    with pytest.raises(Fts5CorruptError, match="upgrade failed"):
        Fts5LexicalIndex(tmp_path / "fts5.db", tmp_path / "state.json")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")
