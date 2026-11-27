"""Recovery must not mistake unreadable PID contents, filesystem errors or its own lease for another owner."""

import errno
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mcp_server import _index_access, _writer_role, instance_lock, preflight


@pytest.fixture
def pid_path(tmp_path, monkeypatch):
    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)
    monkeypatch.setattr(instance_lock.config, "chroma_dir", tmp_path / "chroma_db")
    monkeypatch.setenv(instance_lock.ENV_VAR, "0")
    return tmp_path / instance_lock.LOCK_FILENAME


@pytest.mark.parametrize("contents", [b"\xff", b"\xff\xfe3", b"\x80\xff\n", "0\n".encode("utf-16")])
def test_given_malformed_or_stale_pid_contents_when_guard_is_free_then_one_writer_recovers(pid_path, contents):
    pid_path.write_bytes(contents)
    with instance_lock.writer_lock() as first:
        with instance_lock.writer_lock() as second:
            assert (first, second, instance_lock._read_lock_pid(pid_path)) == (True, False, os.getpid())


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-16-be"])
def test_given_identifiable_live_legacy_pid_when_guard_is_free_then_it_is_not_stolen(pid_path, encoding):
    first_line = f"{os.getpid()}\r\n".encode(encoding)
    if encoding == "utf-16-be":
        first_line = b"\xfe\xff" + first_line
    contents = first_line + b"\xff"  # a damaged second line does not hide the first
    pid_path.write_bytes(contents)

    with instance_lock.writer_lock() as owns:
        assert (owns, pid_path.read_bytes(), instance_lock._read_lock_pid(pid_path)) == (False, contents, os.getpid())


def _deny_pid_read(monkeypatch, pid_path):
    original = Path.open

    def unreadable(path, *args, **kwargs):
        if path == pid_path:
            raise PermissionError(errno.EACCES, "cannot read PID")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unreadable)


def test_given_pid_read_io_error_when_holding_guard_then_it_propagates_and_preserves_the_file(pid_path, monkeypatch):
    pid_path.write_bytes(b"legacy\n")
    with monkeypatch.context() as patch:
        _deny_pid_read(patch, pid_path)
        with pytest.raises(PermissionError, match="cannot read PID"):
            _writer_role.claim_writes()
    assert pid_path.read_bytes() == b"legacy\n"


def test_given_an_unreadable_pid_after_losing_the_guard_then_follower_refuses_and_promotes_after_release(
    pid_path, monkeypatch
):
    state = _writer_role.RoleState()
    monkeypatch.setattr(_writer_role, "STATE", state)
    tool = _writer_role.index_tool(lambda: "served")
    with instance_lock.writer_lock() as owner:
        assert owner and not _writer_role.claim_writes()
        _writer_role.become_follower(state.ready.set)
        with monkeypatch.context() as patch:
            _deny_pid_read(patch, pid_path)
            refusal = json.loads(tool())
        assert (refusal["status"], refusal["owner_pid"], refusal["owner_running"]) == ("error", None, False)
        assert state.is_follower and not state.ready.is_set() and pid_path.exists()
    assert json.loads(tool())["status"] == "repairing"
    assert not state.is_follower and tool() == "served"


def test_given_busy_opt_in_or_owner_cleanup_when_pid_is_unreadable_then_neither_claims_or_removes_it(
    pid_path, monkeypatch
):
    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    with monkeypatch.context() as patch:
        with instance_lock.single_instance_lock():
            _deny_pid_read(patch, pid_path)
            with pytest.raises(instance_lock.AlreadyRunningError, match="OS lock held"):
                _writer_role.claim_writes()
    assert pid_path.exists()  # no deletion without identifying our PID
    with instance_lock.single_instance_lock():
        assert instance_lock._read_lock_pid(pid_path) == os.getpid()


@pytest.mark.parametrize("alias", [False, True], ids=["canonical", "dotdot"])
def test_given_a_standalone_owner_when_preflight_is_nested_then_only_its_own_thread_can_borrow(
    pid_path, monkeypatch, alias
):
    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    if alias:
        sub = pid_path.parent / "sub"
        sub.mkdir()
        monkeypatch.setattr(instance_lock.config, "data_dir", sub / ".." / "data")
        pid_path = pid_path.parent / "data" / instance_lock.LOCK_FILENAME
    probes = []
    monkeypatch.setattr(preflight, "_probe_until_conclusive", lambda *args, **kwargs: probes.append("checked"))

    with instance_lock.single_instance_lock():
        assert preflight.run_preflight() is False
        with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(SystemExit) as busy:
            pool.submit(preflight.run_preflight).result(timeout=5)
        with pytest.raises(instance_lock.AlreadyRunningError):
            _writer_role.claim_writes()
        assert (busy.value.code, pid_path.exists(), probes) == (75, True, ["checked"])
    assert not pid_path.exists()


def test_given_a_nested_standalone_owner_with_live_access_when_checked_then_not_even_a_probe_runs(
    pid_path, monkeypatch
):
    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    monkeypatch.setattr(preflight, "_probe_until_conclusive", lambda *args: pytest.fail("access is already held"))
    with instance_lock.single_instance_lock():
        _index_access.hold(wait_seconds=0)
        with pytest.raises(SystemExit) as busy:
            preflight.run_preflight()
        assert (busy.value.code, _index_access.is_held(), pid_path.exists()) == (75, True, True)


@pytest.mark.skipif(sys.platform == "win32", reason="flock on a read-only descriptor is POSIX-specific")
@pytest.mark.parametrize("filename", [instance_lock.GUARD_FILENAME, _index_access.ACCESS_FILENAME])
@pytest.mark.parametrize("writable", [True, False], ids=["owner", "read-only-peer"])
def test_given_an_existing_0644_sidecar_when_reopened_then_the_flock_still_excludes(
    tmp_path, filename, writable, monkeypatch
):
    import fcntl

    path = tmp_path / filename
    path.touch(mode=0o644)
    original = os.open

    def peer_open(filename, flags, *args):
        if not writable and filename == path and flags & os.O_RDWR:
            raise PermissionError(errno.EACCES, "another UID owns this sidecar")
        if filename == path and not writable:
            assert not flags & os.O_CREAT
        return original(filename, flags, *args)

    monkeypatch.setattr(os, "open", peer_open)
    fd = instance_lock._lock_sidecar(path)
    assert fd is not None
    try:
        assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == (os.O_RDWR if writable else os.O_RDONLY)
        assert instance_lock._lock_sidecar(path) is None
    finally:
        os.close(fd)


@pytest.mark.skipif(sys.platform != "win32", reason="requires real Windows no-delete-sharing handles")
def test_given_a_stale_pid_temporarily_open_when_claiming_then_the_guard_winner_retries(pid_path, monkeypatch):
    pid_path.write_bytes(b"0\n")
    original = Path.unlink
    sharing_errors = []

    def unlink_during_read(path, *args, **kwargs):
        if path == pid_path and not sharing_errors:
            with path.open("rb"), pytest.raises(PermissionError) as blocked:
                original(path, *args, **kwargs)
            sharing_errors.append(blocked.value.winerror)
            raise blocked.value
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink_during_read)
    assert (_writer_role.claim_writes(), sharing_errors, instance_lock._read_lock_pid(pid_path)) == (
        True,
        [32],
        os.getpid(),
    )


@pytest.mark.parametrize("windows_code", [5, 32], ids=["access-denied", "persistent-sharing"])
def test_given_stale_pid_cleanup_failure_when_claiming_then_an_io_error_is_not_a_follower(
    pid_path, monkeypatch, windows_code
):
    pid_path.write_bytes(b"0\n")
    attempts = []
    error = PermissionError(errno.EACCES, "cannot remove stale PID")
    error.winerror = windows_code

    def fail_unlink(path, *args, **kwargs):
        attempts.append(path)
        assert len(attempts) < 20, "sharing retry is not bounded"
        raise error

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(PermissionError) as raised:
        _writer_role.claim_writes()
    assert raised.value is error
    assert (len(attempts) > 1) is (sys.platform == "win32" and windows_code == 32)
    assert pid_path.read_bytes() == b"0\n"
