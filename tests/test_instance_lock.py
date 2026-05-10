"""Tests for the opt-in single-instance guard (v3.8.0+).

Behavior contract:
- Default OFF: contextmanager is a no-op, no lock file created.
- Opt-in via KNOWLEDGE_RAG_SINGLE_INSTANCE=1 (also: true/yes/on, case-insensitive).
- Enabled: a second acquisition with the same live PID raises AlreadyRunningError.
- Stale locks (PID gone) are recovered automatically.
- Cleanup runs on normal exit (finally) and on SIGINT/SIGTERM.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Env-var parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["1", "true", "TRUE", "True", "yes", "YES", "on", "ON", "  1  ", "  true  "],
)
def test_env_var_truthy_enables_guard(monkeypatch, value):
    from mcp_server import instance_lock

    monkeypatch.setenv(instance_lock.ENV_VAR, value)
    assert instance_lock.single_instance_enabled() is True


@pytest.mark.parametrize(
    "value",
    ["", "0", "false", "FALSE", "no", "off", "anything", "  ", "2"],
)
def test_env_var_falsy_keeps_guard_disabled(monkeypatch, value):
    from mcp_server import instance_lock

    monkeypatch.setenv(instance_lock.ENV_VAR, value)
    assert instance_lock.single_instance_enabled() is False


def test_env_var_unset_keeps_guard_disabled(monkeypatch):
    from mcp_server import instance_lock

    monkeypatch.delenv(instance_lock.ENV_VAR, raising=False)
    assert instance_lock.single_instance_enabled() is False


# ---------------------------------------------------------------------------
# No-op default
# ---------------------------------------------------------------------------


def test_disabled_by_default_is_noop(monkeypatch, tmp_path):
    """Without the env var, the contextmanager creates NO lock file."""
    from mcp_server import instance_lock

    monkeypatch.delenv(instance_lock.ENV_VAR, raising=False)
    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)

    with instance_lock.single_instance_lock() as lock:
        assert lock is None
        assert not (tmp_path / instance_lock.LOCK_FILENAME).exists()

    # And nothing left behind after exit either
    assert not (tmp_path / instance_lock.LOCK_FILENAME).exists()


def test_disabled_allows_unlimited_concurrent_acquires(monkeypatch, tmp_path):
    """With guard off, nesting acquires must NOT raise (multi-client friendly)."""
    from mcp_server import instance_lock

    monkeypatch.delenv(instance_lock.ENV_VAR, raising=False)
    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)

    with instance_lock.single_instance_lock():
        with instance_lock.single_instance_lock():
            with instance_lock.single_instance_lock():
                pass  # no AlreadyRunningError, ever


# ---------------------------------------------------------------------------
# Enabled-mode behavior
# ---------------------------------------------------------------------------


def test_enabled_creates_lock_with_pid(monkeypatch, tmp_path):
    from mcp_server import instance_lock

    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)

    lock_path = tmp_path / instance_lock.LOCK_FILENAME
    with instance_lock.single_instance_lock() as held:
        assert held == lock_path
        assert lock_path.exists()
        assert lock_path.read_text(encoding="utf-8").strip() == str(os.getpid())

    # Cleaned up on normal exit
    assert not lock_path.exists()


def test_enabled_rejects_second_instance(monkeypatch, tmp_path):
    from mcp_server import instance_lock

    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)

    with instance_lock.single_instance_lock():
        with pytest.raises(instance_lock.AlreadyRunningError):
            with instance_lock.single_instance_lock():
                pass


def test_enabled_recovers_stale_pid(monkeypatch, tmp_path):
    """A lock file referencing a dead PID must be cleared and re-acquired."""
    from mcp_server import instance_lock

    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)
    monkeypatch.setattr(instance_lock, "_pid_is_running", lambda pid: False)

    lock_path = Path(tmp_path) / instance_lock.LOCK_FILENAME
    lock_path.write_text("999999999\n", encoding="utf-8")

    with instance_lock.single_instance_lock():
        assert lock_path.read_text(encoding="utf-8").strip() == str(os.getpid())

    assert not lock_path.exists()


def test_isolated_data_dirs_are_independent(monkeypatch, tmp_path):
    """Two RAGs pointing at different data_dirs must hold independent locks."""
    from mcp_server import instance_lock

    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    monkeypatch.setattr(instance_lock.config, "data_dir", dir_a)
    with instance_lock.single_instance_lock():
        # Switching the configured data_dir to a sibling RAG must allow a fresh lock
        monkeypatch.setattr(instance_lock.config, "data_dir", dir_b)
        with instance_lock.single_instance_lock():
            assert (dir_a / instance_lock.LOCK_FILENAME).exists()
            assert (dir_b / instance_lock.LOCK_FILENAME).exists()


def test_pid_check_handles_invalid_pid():
    from mcp_server import instance_lock

    assert instance_lock._pid_is_running(0) is False
    assert instance_lock._pid_is_running(-1) is False
