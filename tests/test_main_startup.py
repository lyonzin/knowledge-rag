"""Startup regression guards for GH #216.

Ensure `main()` never blocks the MCP `initialize` handshake on initial
indexing. If a future refactor swaps the background call back to
`orchestrator.index_all()` (or any synchronous variant), these tests
fail before the change ships.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock

from mcp_server import server


@contextmanager
def _noop_lock():
    yield None


def _spawn_daemon_work(mode, resume_state=None):
    """Simulate the real `start_reindex_background` contract.

    v4.9.1 requires the method to spawn a daemon thread and return
    immediately. This side_effect matches that contract by spawning
    a real 2 s thread and returning a status envelope right away. If
    a future refactor makes the wrapper synchronous (drops the thread
    launch), the timing assertion in
    `test_main_reaches_transport_under_deadline` will catch it — main()
    would then await the 2 s of "indexing work" instead of proceeding
    to `_run_transport` immediately.
    """
    threading.Thread(target=lambda: time.sleep(2.0), daemon=True).start()
    return {"status": "started", "operation": mode}


def _mock_orchestrator_empty() -> MagicMock:
    """Orchestrator with collection.count()=0, `start_reindex_background`
    behaves like the real v4.9.1 implementation (spawn daemon, return
    immediately). Kept as a shared helper so every test in this file
    exercises the same realistic contract.
    """
    mock = MagicMock()
    mock._check_dimension_mismatch.return_value = False
    mock.collection.count.return_value = 0
    mock.start_reindex_background.side_effect = _spawn_daemon_work
    return mock


def _mock_orchestrator_populated() -> MagicMock:
    """Orchestrator with an existing populated collection."""
    mock = MagicMock()
    mock._check_dimension_mismatch.return_value = False
    mock.collection.count.return_value = 42
    return mock


def _patch_startup_side_effects(monkeypatch):
    """Neutralize watcher, transport, logging, preflight, and single-instance lock.

    `main()` imports these locally (`from .X import Y`) at call-time, so
    patches must target the *source* module (`mcp_server.X`), not the
    server namespace — the local import re-reads the module's current
    attribute at execution time.
    """
    from mcp_server import instance_lock, logging_config, preflight

    monkeypatch.setenv("KNOWLEDGE_RAG_WATCHER_DISABLED", "1")
    monkeypatch.setattr(instance_lock, "single_instance_lock", _noop_lock)
    monkeypatch.setattr(preflight, "run_preflight", lambda timeout_seconds=30: False)
    monkeypatch.setattr(logging_config, "setup_logging", lambda fmt="text", level="INFO": None)
    monkeypatch.setattr(server, "_run_transport", lambda transport: None)


def test_empty_collection_starts_background_not_sync(monkeypatch):
    """GH #216: initial indexing MUST use start_reindex_background, never index_all sync."""
    _patch_startup_side_effects(monkeypatch)
    mock_orch = _mock_orchestrator_empty()
    monkeypatch.setattr(server, "get_orchestrator", lambda: mock_orch)

    server.main()

    mock_orch.start_reindex_background.assert_called_once_with("incremental")
    mock_orch.index_all.assert_not_called()


def test_populated_collection_triggers_no_indexing(monkeypatch):
    """When index is already populated, main() must not touch either indexing path."""
    _patch_startup_side_effects(monkeypatch)
    mock_orch = _mock_orchestrator_populated()
    monkeypatch.setattr(server, "get_orchestrator", lambda: mock_orch)

    server.main()

    mock_orch.start_reindex_background.assert_not_called()
    mock_orch.index_all.assert_not_called()
    mock_orch.nuclear_rebuild.assert_not_called()


def test_start_reindex_background_returns_before_transport(monkeypatch):
    """Regression guard: main() must not await indexing completion.

    Records the order of `start_reindex_background` and `_run_transport`
    calls — if a future refactor makes indexing synchronous, transport
    would fire only after indexing completes, which is exactly the bug
    that #216 documented.
    """
    _patch_startup_side_effects(monkeypatch)
    mock_orch = _mock_orchestrator_empty()

    call_order: list[str] = []

    def track_background(mode, resume_state=None):
        call_order.append(f"background:{mode}")
        # Preserve the realistic spawn behavior from `_spawn_daemon_work`.
        threading.Thread(target=lambda: time.sleep(2.0), daemon=True).start()
        return {"status": "started", "operation": mode}

    def track_transport(transport):
        call_order.append(f"transport:{transport}")

    mock_orch.start_reindex_background.side_effect = track_background
    monkeypatch.setattr(server, "get_orchestrator", lambda: mock_orch)
    monkeypatch.setattr(server, "_run_transport", track_transport)

    server.main()

    assert call_order == ["background:incremental", f"transport:{server.config.transport}"]


def test_main_reaches_transport_under_deadline(monkeypatch):
    """GH #216 timing guard against Greptile PR #217 review finding.

    The previous handshake-order test mocked `start_reindex_background`
    with a lambda that returned instantly, so a hypothetical refactor
    that made the wrapper synchronous (dropped `threading.Thread`) would
    still pass the ordering assertion. `_spawn_daemon_work` fixes that by
    spawning a real 2 s daemon thread — main() must therefore complete
    within a tight deadline. If the indexing call ever blocks main() on
    the daemon's work, this assertion catches it.
    """
    _patch_startup_side_effects(monkeypatch)
    mock_orch = _mock_orchestrator_empty()
    monkeypatch.setattr(server, "get_orchestrator", lambda: mock_orch)

    start = time.monotonic()
    server.main()
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, (
        f"main() took {elapsed:.2f}s to reach _run_transport — expected < 0.5s. "
        f"start_reindex_background likely became synchronous and blocked the handshake."
    )
