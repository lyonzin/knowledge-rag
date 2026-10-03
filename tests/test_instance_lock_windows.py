"""Windows liveness checks must never signal or terminate the inspected PID."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from mcp_server import instance_lock


@pytest.mark.skipif(os.name != "nt", reason="Windows process semantics")
def test_windows_pid_probe_never_uses_os_kill():
    """Regression reproduction with kill intercepted before it can do damage."""
    with patch.object(instance_lock.os, "kill", side_effect=AssertionError("os.kill is not a Windows liveness API")):
        assert instance_lock._pid_is_running(os.getpid()) is True


def test_signal_cleanup_delegates_to_previous_python_handler(tmp_path, monkeypatch):
    """Forward SIGINT as a Python call instead of terminating Windows Python."""
    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)
    with patch.object(instance_lock.os, "kill", side_effect=AssertionError("cleanup must not call os.kill")):
        with instance_lock.single_instance_lock() as lock_path:
            handler = signal.getsignal(signal.SIGINT)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGINT, None)
            # The caller caught the signal and is still inside the guard.
            assert lock_path.exists()
        assert not lock_path.exists()


@pytest.mark.parametrize("wait_result, running", [(0, False), (258, True), (0xFFFFFFFF, True)])
def test_windows_handle_query_closes_handle(wait_result, running):
    """Exited, running and inconclusive wait results preserve handle ownership."""
    kernel = MagicMock()
    kernel.OpenProcess.return_value = 123
    kernel.WaitForSingleObject.return_value = wait_result
    with patch("ctypes.WinDLL", return_value=kernel, create=True):
        assert instance_lock._windows_pid_is_running(456) is running
    kernel.OpenProcess.assert_called_once_with(0x00100000, False, 456)
    kernel.WaitForSingleObject.assert_called_once_with(123, 0)
    kernel.CloseHandle.assert_called_once_with(123)


@pytest.mark.parametrize("error, running", [(87, False), (5, True), (8, True)])
def test_windows_pid_open_failure_is_conservative(error, running):
    """Access denied or an unknown OS failure must not steal a live lock."""
    kernel = MagicMock()
    kernel.OpenProcess.return_value = None
    with (
        patch("ctypes.WinDLL", return_value=kernel, create=True),
        patch("ctypes.get_last_error", return_value=error, create=True),
    ):
        assert instance_lock._windows_pid_is_running(456) is running
    kernel.CloseHandle.assert_not_called()


@pytest.mark.skipif(os.name != "nt", reason="Windows process semantics")
def test_real_windows_child_survives_liveness_queries():
    """Read-only probes leave a child alive until it exits normally through stdin."""
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as child:
        try:
            for _ in range(3):
                assert instance_lock._pid_is_running(child.pid) is True
                assert child.poll() is None
        finally:
            child.communicate(b"x", timeout=10)
        assert child.returncode == 0
        assert instance_lock._pid_is_running(child.pid) is False


def test_default_sigterm_cleanup_exits_without_resending_signal(tmp_path, monkeypatch):
    """Default termination preserves Python cleanup with the conventional code."""
    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        with instance_lock.single_instance_lock() as lock_path:
            with pytest.raises(SystemExit, match="143"):
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            assert lock_path.exists()
        assert not lock_path.exists()
    finally:
        signal.signal(signal.SIGTERM, previous)


@pytest.mark.parametrize("ignored", [False, True])
def test_handled_signal_keeps_process_guard_until_context_exits(tmp_path, monkeypatch, ignored):
    """SIG_IGN and a returning Python handler both leave a running server protected."""
    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)
    previous = signal.getsignal(signal.SIGTERM)
    received = []

    def record_signal(signum, frame):
        received.append(signum)

    handler = signal.SIG_IGN if ignored else record_signal
    signal.signal(signal.SIGTERM, handler)
    try:
        with instance_lock.single_instance_lock() as lock_path:
            for _ in range(2):
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            assert lock_path.exists()
            with pytest.raises(instance_lock.AlreadyRunningError):
                with instance_lock.single_instance_lock():
                    pytest.fail("A continuing process lost its guard")
        assert signal.getsignal(signal.SIGTERM) is handler
        if not ignored:
            assert received == [signal.SIGTERM, signal.SIGTERM]
    finally:
        signal.signal(signal.SIGTERM, previous)
