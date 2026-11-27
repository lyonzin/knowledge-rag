"""Run ``server.main()`` in-process with its startup side effects replaced, then wait for it."""

import threading
import time
from unittest.mock import MagicMock

from mcp_server import _index_access, _writer_role, instance_lock, logging_config, preflight, server

STARTUP_JOIN_SECONDS = 10
TRANSPORT_WAIT_SECONDS = 5


def spawn_daemon_work(mode, resume_state=None):
    """Simulate the real `start_reindex_background` contract.

    v4.9.1 requires the method to spawn a daemon thread and return
    immediately. This side_effect matches that contract by spawning
    a real 2 s thread and returning a status envelope right away. If
    a future refactor makes the wrapper synchronous (drops the thread
    launch), the startup thread would await the 2 s of "indexing work".
    """
    threading.Thread(target=lambda: time.sleep(2.0), daemon=True).start()
    return {"status": "started", "operation": mode}


def mock_orchestrator_empty() -> MagicMock:
    """Orchestrator with collection.count()=0, `start_reindex_background`
    behaves like the real v4.9.1 implementation (spawn daemon, return
    immediately). Kept as a shared helper so every test exercises the
    same realistic contract.
    """
    mock = MagicMock()
    mock._check_dimension_mismatch.return_value = False
    mock.collection.count.return_value = 0
    mock.start_reindex_background.side_effect = spawn_daemon_work
    return mock


def mock_orchestrator_populated() -> MagicMock:
    """Orchestrator with an existing populated collection."""
    mock = MagicMock()
    mock._check_dimension_mismatch.return_value = False
    mock.collection.count.return_value = 42
    return mock


def patch_startup_side_effects(monkeypatch, is_writer=True):
    """Neutralize watcher, transport, logging, cloud check, preflight, the writer claim and index access.

    `main()` imports these locally (`from .X import Y`) at call-time, so
    patches must target the *source* module (`mcp_server.X`), not the
    server namespace — the local import re-reads the module's current
    attribute at execution time. A fresh RoleState keeps one test's role out
    of the next one. ENV_VAR is set, not deleted: deleting an unset variable
    records nothing, so the "1" main() writes for HTTP would leak onwards.
    """
    monkeypatch.setenv("KNOWLEDGE_RAG_WATCHER_DISABLED", "1")
    monkeypatch.setenv(instance_lock.ENV_VAR, "0")
    monkeypatch.setattr(_writer_role, "STATE", _writer_role.RoleState())
    monkeypatch.setattr(server, "_startup_thread", None)
    monkeypatch.setattr(_writer_role, "claim_writes", lambda: is_writer)
    monkeypatch.setattr(preflight, "_check_cloud_sync", lambda: None)
    monkeypatch.setattr(preflight, "_probe_and_repair", lambda timeout_seconds=30: False)
    monkeypatch.setattr(_index_access, "hold", lambda wait_seconds=_index_access.WAIT_SECONDS: None)
    monkeypatch.setattr(logging_config, "setup_logging", lambda fmt="text", level="INFO": None)
    monkeypatch.setattr(server, "_run_transport", lambda transport: None)


def join_startup():
    """Wait for the owner's background startup, so assertions see all it did."""
    if server._startup_thread is not None:
        server._startup_thread.join(timeout=STARTUP_JOIN_SECONDS)
        assert not server._startup_thread.is_alive(), "the writer startup thread did not finish"


def run_main():
    server.main()
    join_startup()


def hold_preflight_until(monkeypatch, transport_started):
    """The preflight returns only once the transport started; ahead of it, it fails."""

    def held_preflight(timeout_seconds=30):
        assert transport_started.wait(TRANSPORT_WAIT_SECONDS), "the startup check ran before the transport started"
        return False

    monkeypatch.setattr(preflight, "_probe_and_repair", held_preflight)
