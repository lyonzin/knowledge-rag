"""Persisted vectors and completed migrations, not catalog promises, determine startup readiness."""

import asyncio
import json
import subprocess
import sys

import chromadb
import pytest
from support.mcp_stdio import isolated_env

from mcp_server import _writer_role, server
from mcp_server.health import HealthMiddleware


async def _must_not_run(*args):
    pytest.fail("readiness must not reach the downstream app")


def _readyz():
    messages = []

    async def send(message):
        messages.append(message)

    health = HealthMiddleware(_must_not_run)
    asyncio.run(health({"type": "http", "path": "/readyz"}, _must_not_run, send))
    return messages[0]["status"], json.loads(messages[1]["body"])["status"]


@pytest.fixture
def owner(tmp_path, monkeypatch):
    monkeypatch.setattr(server.config, "data_dir", tmp_path / "data")
    monkeypatch.setattr(server.config, "embedding_dim", 3)
    monkeypatch.setattr(_writer_role, "STATE", _writer_role.RoleState())
    monkeypatch.setattr(server, "_startup_thread", None)
    monkeypatch.setattr(server, "_start_watcher", lambda: None)


def _start_and_join():
    server._start_writer_startup(lambda: False)
    server._startup_thread.join(timeout=10)
    assert not server._startup_thread.is_alive()


@pytest.mark.parametrize("failure", [SystemExit(42), BaseException("native startup stopped")])
def test_given_an_abnormal_thread_exit_when_opening_then_readiness_and_tools_report_terminal_error(
    owner, monkeypatch, failure
):
    def broken_open():
        raise failure

    monkeypatch.setattr(server, "_open_orchestrator", broken_open)
    _start_and_join()
    assert _readyz() == (503, "error")
    with pytest.raises(_writer_role.IndexNotReady) as refused:
        _writer_role.raise_unless_index_ready()
    assert refused.value.status == "error"


@pytest.fixture(scope="module")
def disk_client(tmp_path_factory):
    return chromadb.PersistentClient(path=str(tmp_path_factory.mktemp("persisted-chroma")))


@pytest.fixture
def persisted(disk_client, monkeypatch, owner):
    """The real startup checker on a disk collection, with no model or loader. Stored dimension is 2."""
    index = object.__new__(server.KnowledgeOrchestrator)
    index.chroma_client = disk_client
    index.collection = index.chroma_client.create_collection("old-index", embedding_function=None)
    index.collection.add(ids=["chunk"], embeddings=[[1.0, 0.0]])
    index._indexed_docs = {"chunk": {"embedding_dim": 2}}
    monkeypatch.setattr(server, "_open_orchestrator", lambda: index)
    monkeypatch.setattr(index, "_start_bm25_background_build", lambda name: None)
    yield index
    for collection in disk_client.list_collections():
        disk_client.delete_collection(collection.name)


def test_given_a_catalog_with_the_configured_dimension_when_the_vectors_disagree_then_it_needs_migration(persisted):
    persisted._indexed_docs["chunk"]["embedding_dim"] = 3
    assert persisted._check_dimension_mismatch() is True


@pytest.mark.parametrize("fallback", [False, True], ids=["rebuild", "fallback"])
@pytest.mark.parametrize("outcome", ["errors", "skipped_reason", "wrong-dimension", "empty", "compatible"])
def test_given_a_migration_when_it_finishes_then_readyz_requires_success_and_compatible_stored_vectors(
    persisted, monkeypatch, fallback, outcome
):
    def migrate():
        persisted._indexed_docs["chunk"]["embedding_dim"] = 3
        if outcome != "wrong-dimension":
            persisted.collection = persisted.chroma_client.create_collection("new-index", embedding_function=None)
            if outcome != "empty":
                persisted.collection.add(ids=["chunk"], embeddings=[[1.0, 0.0, 0.0]])
        stats = {"indexed": 1, "chunks_added": 1, "errors": int(outcome == "errors")}
        if outcome == "skipped_reason":
            stats["skipped_reason"] = "reindex_already_running"
        return stats

    def rebuild():
        if fallback:
            raise RuntimeError("cannot publish staging collection")
        return migrate()

    def reindex(force):
        assert fallback and force
        return migrate()

    monkeypatch.setattr(persisted, "nuclear_rebuild", rebuild)
    monkeypatch.setattr(persisted, "index_all", reindex)
    _start_and_join()
    assert _readyz() == ((200, "ready") if outcome == "compatible" else (503, "error"))


def test_given_an_isolated_health_middleware_when_imported_and_served_then_no_storage_is_initialized(tmp_path):
    base = tmp_path / "not-a-data-directory"
    base.write_bytes(b"sentinel")
    child = """
import asyncio, sys
from mcp_server.health import HealthMiddleware
async def unused(*args): raise AssertionError('unexpected downstream call')
async def send(event):
    if 'status' in event: assert event['status'] == 200
middleware = HealthMiddleware(unused)
for path in ('/health', '/readyz'):
    asyncio.run(middleware({'type': 'http', 'path': path}, unused, send))
assert 'mcp_server.config' not in sys.modules
"""
    done = subprocess.run(
        [sys.executable, "-c", child], cwd=tmp_path, env=isolated_env(base), capture_output=True, text=True, timeout=20
    )
    assert done.returncode == 0, done.stderr
    assert base.read_bytes() == b"sentinel"
