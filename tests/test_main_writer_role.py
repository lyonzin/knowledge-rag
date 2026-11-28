"""main() and the single writer per data_dir: who checks the index, and when.

The process that owns data_dir answers the handshake first and checks, opens
and prepares the index on a background thread, publishing ready only after
all three, and keeps data_dir until it dies; a follower runs none of it; an HTTP server never follows, and a
cloud-synced data_dir stops both roles before either touches it.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from support.main_startup import (
    STARTUP_JOIN_SECONDS,
    hold_preflight_until,
    mock_orchestrator_populated,
    patch_startup_side_effects,
    run_main,
)

from mcp_server import _writer_role, instance_lock, preflight, server

REAL_CLAIM = _writer_role.claim_writes
REAL_CLOUD_CHECK = preflight._check_cloud_sync
CLOUD_SYNC_EXIT_CODE = 78  # EX_CONFIG, preflight._check_cloud_sync


def test_given_a_slow_preflight_when_main_starts_as_writer_then_the_transport_starts_before_preflight_ends(
    monkeypatch,
):
    patch_startup_side_effects(monkeypatch)
    release, finished = threading.Event(), threading.Event()
    preflight_finished_at_transport = []

    def slow_preflight():
        release.wait(STARTUP_JOIN_SECONDS)
        finished.set()
        return False

    def transport(name):
        preflight_finished_at_transport.append(finished.is_set())
        release.set()

    monkeypatch.setattr(preflight, "_probe_and_repair", slow_preflight)
    monkeypatch.setattr(server, "_open_orchestrator", mock_orchestrator_populated)
    monkeypatch.setattr(server, "_run_transport", transport)

    run_main()

    assert preflight_finished_at_transport == [False]


class _ReadyRecorder(threading.Event):
    """STATE.ready that notes when the startup publishes it."""

    def __init__(self, order):
        super().__init__()
        self.order = order

    def set(self):
        self.order.append("ready")
        super().set()


def test_given_the_owner_startup_when_the_check_passes_then_ready_comes_after_the_open_and_the_preparation(monkeypatch):
    """Tools may use the index only once it is open and prepared, so ready is published after both."""
    patch_startup_side_effects(monkeypatch)
    order = []

    def open_index():
        order.append("open")
        return mock_orchestrator_populated()

    monkeypatch.setattr(_writer_role.STATE, "ready", _ReadyRecorder(order))
    monkeypatch.setattr(server, "_open_orchestrator", open_index)
    monkeypatch.setattr(server, "_prepare_index", lambda orchestrator: order.append("prepare"))

    run_main()

    assert order == ["open", "prepare", "ready"]


def test_given_a_follower_when_main_starts_then_preflight_watcher_migration_and_indexing_never_run(monkeypatch):
    patch_startup_side_effects(monkeypatch, is_writer=False)
    startup_work = MagicMock()
    monkeypatch.setattr(preflight, "_probe_and_repair", startup_work.preflight)
    monkeypatch.setattr(server, "_open_orchestrator", startup_work.open_index)
    monkeypatch.setattr(server, "_prepare_index", startup_work.migrate_or_index)
    monkeypatch.setattr(server, "_start_watcher", startup_work.watch)

    run_main()

    assert (startup_work.mock_calls, _writer_role.STATE.is_follower) == ([], True)


def test_given_http_transport_when_the_writer_lock_is_busy_then_main_exits_75(monkeypatch, tmp_path):
    patch_startup_side_effects(monkeypatch)
    monkeypatch.setattr(_writer_role, "claim_writes", REAL_CLAIM)
    monkeypatch.setattr(server.config, "data_dir", tmp_path)
    monkeypatch.setattr(server.config, "transport", "stdio")
    monkeypatch.setattr(server.sys, "argv", ["knowledge-rag", "--transport", "streamable-http"])

    with instance_lock.writer_lock() as stdio_session_owns_writes:
        assert stdio_session_owns_writes
        with pytest.raises(SystemExit) as exited:
            server.main()

    assert exited.value.code == instance_lock.ALREADY_RUNNING_EXIT_CODE


def test_given_the_owner_when_its_transport_ends_then_the_guard_stays_held_until_the_process_dies(
    monkeypatch, tmp_path
):
    """A daemon thread may still write after main() returns: no one else may take over before the process dies."""
    patch_startup_side_effects(monkeypatch)
    monkeypatch.setattr(_writer_role, "claim_writes", REAL_CLAIM)
    monkeypatch.setattr(server.config, "data_dir", tmp_path)
    monkeypatch.setattr(server, "_open_orchestrator", mock_orchestrator_populated)

    run_main()

    with instance_lock.writer_lock() as another_process_owns:
        pass
    assert another_process_owns is False


def test_given_a_cloud_sync_data_dir_when_main_starts_then_it_exits_78_before_taking_the_lock(monkeypatch, tmp_path):
    patch_startup_side_effects(monkeypatch)
    monkeypatch.setattr(preflight, "_check_cloud_sync", REAL_CLOUD_CHECK)
    monkeypatch.setattr(preflight, "_get", lambda section, key, default: default)  # no allow_cloud_sync opt-in
    monkeypatch.setattr(server.config, "data_dir", tmp_path / "OneDrive" / "data")
    claim = MagicMock()
    monkeypatch.setattr(_writer_role, "claim_writes", claim)

    with pytest.raises(SystemExit) as exited:
        server.main()

    assert (exited.value.code, claim.call_count) == (CLOUD_SYNC_EXIT_CODE, 0)


def test_given_the_writer_starts_when_main_runs_then_the_cloud_check_runs_once(monkeypatch):
    """main() checks before the lock; the owner's background probe must not check (and warn) again."""
    patch_startup_side_effects(monkeypatch)
    checks = []
    monkeypatch.setattr(preflight, "_check_cloud_sync", lambda: checks.append("cloud-sync"))
    monkeypatch.setattr(server, "_open_orchestrator", mock_orchestrator_populated)

    run_main()

    assert checks == ["cloud-sync"]


def test_given_a_dimension_mismatch_when_the_writer_starts_then_the_rebuild_runs_after_the_transport_started(
    monkeypatch,
):
    patch_startup_side_effects(monkeypatch)
    transport_started = threading.Event()
    hold_preflight_until(monkeypatch, transport_started)
    order = []
    mock_orch = mock_orchestrator_populated()
    mock_orch._check_dimension_mismatch.side_effect = [True, False]

    def rebuild():
        order.append("rebuild")
        return {"indexed": 1, "chunks_added": 1, "errors": 0, "elapsed_seconds": 0.0}

    def transport(name):
        order.append("transport")
        transport_started.set()

    mock_orch.nuclear_rebuild.side_effect = rebuild
    monkeypatch.setattr(server, "_open_orchestrator", lambda: mock_orch)
    monkeypatch.setattr(server, "_run_transport", transport)

    run_main()

    assert order == ["transport", "rebuild"]
