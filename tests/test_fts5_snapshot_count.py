"""A delayed migration must count the same corpus it actually streams."""

from mcp_server import server


def test_migration_recounts_after_acquiring_mutation_lock(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    orch.add_document_from_content("initial source content", "initial.txt", "general")
    monkeypatch.setattr(server.config, "fts5_enabled", True)
    with orch._index_lock:
        orch._initialize_fts5_dispatch()
        orch.add_document_from_content("second source content", "second.txt", "general")
    orch._fts5_migration_thread.join(5)
    assert not orch._fts5_migration_thread.is_alive()
    state = orch.fts5_index.state.read()
    assert orch.collection.count() == orch.fts5_index.count() == 2
    assert state["status"] == "complete"
    assert state["docs_total"] == state["docs_indexed"] == 2
    assert orch.fts5_index.is_ready()
