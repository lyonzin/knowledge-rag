"""Startup regression guards for GH #216 and GH #321.

Ensure `main()` never blocks the MCP `initialize` handshake on checking,
repairing or indexing the index: the process that owns writes runs that work
on a background thread (tests/test_main_writer_role.py covers the roles). If
a future refactor swaps the background call back to `orchestrator.index_all()`
(or any synchronous variant), these tests fail before the change ships. The
BM25 prewarm must start through the background build, whose lock a
keyword-only search waits on (GH #321).
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest
from support.main_startup import (
    hold_preflight_until,
    join_startup,
    mock_orchestrator_empty,
    mock_orchestrator_populated,
    patch_startup_side_effects,
    run_main,
)

from mcp_server import preflight, server

HANDSHAKE_DEADLINE_SECONDS = 0.5


def test_empty_collection_starts_background_not_sync(monkeypatch):
    """GH #216: initial indexing MUST use start_reindex_background, never index_all sync."""
    patch_startup_side_effects(monkeypatch)
    mock_orch = mock_orchestrator_empty()
    monkeypatch.setattr(server, "_open_orchestrator", lambda: mock_orch)

    run_main()

    mock_orch.start_reindex_background.assert_called_once_with("incremental")
    mock_orch.index_all.assert_not_called()
    mock_orch._start_bm25_background_build.assert_not_called()


def test_populated_collection_triggers_no_indexing(monkeypatch):
    """When index is already populated, main() must not touch either indexing path."""
    patch_startup_side_effects(monkeypatch)
    mock_orch = mock_orchestrator_populated()
    monkeypatch.setattr(server, "_open_orchestrator", lambda: mock_orch)

    run_main()

    mock_orch.start_reindex_background.assert_not_called()
    mock_orch.index_all.assert_not_called()
    mock_orch.nuclear_rebuild.assert_not_called()


def test_given_a_populated_collection_when_main_starts_then_the_bm25_prewarm_goes_through_the_background_build(
    monkeypatch,
):
    patch_startup_side_effects(monkeypatch)
    mock_orch = mock_orchestrator_populated()
    monkeypatch.setattr(server, "_open_orchestrator", lambda: mock_orch)

    run_main()

    mock_orch._start_bm25_background_build.assert_called_once_with("bm25-prewarm")


def test_start_reindex_background_runs_after_the_transport_started(monkeypatch):
    """Regression guard: main() must not await indexing, nor even start it first.

    The transport now starts before the startup check, and the initial
    indexing follows that check on the startup thread. A refactor that made
    the startup synchronous again blocks main() in the held preflight, which
    fails, instead of passing.
    """
    patch_startup_side_effects(monkeypatch)
    transport_started = threading.Event()
    hold_preflight_until(monkeypatch, transport_started)
    mock_orch = mock_orchestrator_empty()

    call_order: list[str] = []

    def track_background(mode, resume_state=None):
        call_order.append(f"background:{mode}")
        # Preserve the realistic spawn behavior from `spawn_daemon_work`.
        threading.Thread(target=lambda: time.sleep(2.0), daemon=True).start()
        return {"status": "started", "operation": mode}

    def track_transport(transport):
        call_order.append(f"transport:{transport}")
        transport_started.set()

    mock_orch.start_reindex_background.side_effect = track_background
    monkeypatch.setattr(server, "_open_orchestrator", lambda: mock_orch)
    monkeypatch.setattr(server, "_run_transport", track_transport)

    run_main()

    assert call_order == [f"transport:{server.config.transport}", "background:incremental"]


def test_main_reaches_transport_under_deadline(monkeypatch):
    """GH #216 timing guard against Greptile PR #217 review finding.

    The previous handshake-order test mocked `start_reindex_background`
    with a lambda that returned instantly, so a hypothetical refactor
    that made the wrapper synchronous (dropped `threading.Thread`) would
    still pass the ordering assertion. `spawn_daemon_work` fixes that by
    spawning a real 2 s daemon thread — main() must therefore complete
    within a tight deadline. If the indexing call ever blocks main() on
    the daemon's work, this assertion catches it.
    """
    patch_startup_side_effects(monkeypatch)
    mock_orch = mock_orchestrator_empty()
    monkeypatch.setattr(server, "_open_orchestrator", lambda: mock_orch)

    start = time.monotonic()
    server.main()
    elapsed = time.monotonic() - start
    join_startup()

    assert elapsed < HANDSHAKE_DEADLINE_SECONDS, (
        f"main() took {elapsed:.2f}s to reach _run_transport — expected < {HANDSHAKE_DEADLINE_SECONDS}s. "
        f"start_reindex_background likely became synchronous and blocked the handshake."
    )


@pytest.mark.parametrize("args", [["--transport", "streamable-http"], ["--transport=sse"]])
def test_cli_transport_reaches_config_before_startup_components(monkeypatch, args):
    from mcp_server import metrics

    patch_startup_side_effects(monkeypatch)
    monkeypatch.setattr(server.sys, "argv", ["knowledge-rag", *args])
    monkeypatch.setattr(server.config, "transport", "stdio")
    monkeypatch.setattr(server.config, "metrics_enabled", True)
    observed = []
    expected = "sse" if "=" in args[0] else "streamable-http"
    monkeypatch.setattr(preflight, "_probe_and_repair", lambda: observed.append(("preflight", server.config.transport)))

    def open_orchestrator():
        observed.append(("orchestrator", server.config.transport))
        return mock_orchestrator_populated()

    monkeypatch.setattr(server, "_open_orchestrator", open_orchestrator)
    monkeypatch.setattr(
        metrics, "start_metrics_server", lambda port: observed.append(("metrics", server.config.transport))
    )
    monkeypatch.setattr(server, "_run_transport", lambda transport: observed.append(("transport", transport)))
    run_main()
    # The startup thread interleaves with the transport, so the order is not compared.
    components = ("metrics", "orchestrator", "preflight", "transport")
    assert sorted(observed) == [(component, expected) for component in components]


def test_invalid_cli_transport_fails_before_opening_database(monkeypatch):
    patch_startup_side_effects(monkeypatch)
    monkeypatch.setattr(server.sys, "argv", ["knowledge-rag", "--transport", "invalid"])
    orchestrator = MagicMock()
    monkeypatch.setattr(server, "_open_orchestrator", orchestrator)
    with pytest.raises(ValueError, match="Unknown transport"):
        server.main()
    orchestrator.assert_not_called()
