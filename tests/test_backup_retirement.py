"""Real-storage recovery backups need durable evidence before TTL retirement."""

import time

import pytest

from mcp_server import server


@pytest.fixture
def restart_orchestrator(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    monkeypatch.setattr(server.KnowledgeOrchestrator, "_init_chroma_client", lambda self: orch.chroma_client)
    monkeypatch.setattr(server, "FastEmbedEmbeddings", lambda: orch.embed_fn)
    return server.KnowledgeOrchestrator


@pytest.mark.parametrize("primary_exists", [False, True])
@pytest.mark.parametrize("already_stale", [False, True])
def test_uncommitted_backup_survives_first_and_second_startup(
    isolated_orchestrator, restart_orchestrator, monkeypatch, primary_exists, already_stale
):
    orch = isolated_orchestrator
    now = int(time.time())
    timestamp = now - orch._STAGING_TTL_SECONDS - 1 if already_stale else now
    backup_name = f"{server.config.collection_name}__old_{timestamp}"
    orch.collection.add(ids=["recoverable"], documents=["Only recoverable content"], embeddings=[[1.0, 1.0]])
    orch.collection.modify(name=backup_name)
    if primary_exists:
        orch.chroma_client.create_collection(server.config.collection_name, embedding_function=orch.embed_fn)

    for clock in (now, now + orch._STAGING_TTL_SECONDS + 1):
        monkeypatch.setattr(server.time, "time", lambda: clock)
        restarted = restart_orchestrator()
        assert restarted.collection.count() == 0
        backup = orch.chroma_client.get_collection(backup_name)
        assert backup.get(include=["documents"])["documents"] == ["Only recoverable content"]


def seed_replacement(orch):
    orch.add_document_from_content("Original committed content", "one.txt", "general")
    (server.config.documents_dir / "one.txt").write_text("Replacement committed content", encoding="utf-8")


def test_committed_retirement_survives_restart_and_expires(isolated_orchestrator, restart_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    seed_replacement(orch)
    now = int(time.time())
    monkeypatch.setattr(server.time, "time", lambda: now)
    delete = orch.chroma_client.delete_collection

    def fail_retirement(name):
        if "__old_" in name:
            assert orch._load_metadata() == orch._indexed_docs
            raise OSError("retirement deletion unavailable")
        return delete(name)

    with monkeypatch.context() as temporary:
        temporary.setattr(orch.chroma_client, "delete_collection", fail_retirement)
        assert orch.nuclear_rebuild()["indexed"] == 1
    backup_name = f"{server.config.collection_name}__old_{now}"
    backup = orch.chroma_client.get_collection(backup_name)
    assert backup.metadata[orch._RETIRED_BY_KEY] == str(orch.collection.id)
    monkeypatch.setattr(server.time, "time", lambda: now + orch._STAGING_TTL_SECONDS + 1)
    restarted = restart_orchestrator()
    assert backup_name not in {collection.name for collection in orch.chroma_client.list_collections()}
    assert restarted.collection.get(include=["documents"])["documents"] == ["Replacement committed content"]


def test_failed_retirement_marker_preserves_backup(isolated_orchestrator, restart_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    seed_replacement(orch)
    now = int(time.time())
    monkeypatch.setattr(server.time, "time", lambda: now)
    modify = type(orch.collection).modify

    def fail_marker(collection, **kwargs):
        if orch._RETIRED_BY_KEY in (kwargs.get("metadata") or {}):
            assert orch._load_metadata() == orch._indexed_docs
            raise OSError("retirement marker unavailable")
        return modify(collection, **kwargs)

    monkeypatch.setattr(type(orch.collection), "modify", fail_marker)
    assert orch.nuclear_rebuild()["indexed"] == 1
    backup_name = f"{server.config.collection_name}__old_{now}"
    monkeypatch.setattr(server.time, "time", lambda: now + orch._STAGING_TTL_SECONDS + 1)
    for _ in range(2):
        restart_orchestrator()
        backup = orch.chroma_client.get_collection(backup_name)
        assert backup.count() == 1
        assert orch._RETIRED_BY_KEY not in (backup.metadata or {})


def test_retirement_marker_cannot_authorize_a_new_empty_primary(isolated_orchestrator, restart_orchestrator):
    orch = isolated_orchestrator
    stale = int(time.time()) - orch._STAGING_TTL_SECONDS - 1
    old_primary_id = str(orch.collection.id)
    backup_name = f"{server.config.collection_name}__old_{stale}"
    backup = orch.chroma_client.create_collection(backup_name, metadata={orch._RETIRED_BY_KEY: old_primary_id})
    backup.add(ids=["recoverable"], documents=["Recovery after lost primary"], embeddings=[[1.0, 1.0]])
    orch.chroma_client.delete_collection(server.config.collection_name)

    for _ in range(2):
        restarted = restart_orchestrator()
        assert str(restarted.collection.id) != old_primary_id
        assert orch.chroma_client.get_collection(backup_name).count() == 1
