"""The preflight and the index access guard: a probe and the owner never touch the index at once.

Unit tests on the real locks in a temporary data_dir; no Chroma, no model. A
process inside the index is played by a second descriptor on the access
sidecar, which conflicts with this process's own like another process's would.
tests/test_single_writer_processes.py runs the same guard across processes.
"""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import pytest
from support.harmless_probe import busy, healthy, native_crash

from mcp_server import _index_access, _preflight_repair, _writer_role, instance_lock, preflight

LOCKED = OSError(32, "The process cannot access the file because it is being used by another process")


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    """A damaged-looking index in its own data_dir, owned by this process as main() would own it."""
    directory = tmp_path / "data"
    (directory / "chroma_db").mkdir(parents=True)
    (directory / "chroma_db" / "chroma.sqlite3").touch()
    monkeypatch.delenv(instance_lock.ENV_VAR, raising=False)
    monkeypatch.setattr(preflight.config, "data_dir", directory)
    monkeypatch.setattr(preflight.config, "chroma_dir", directory / "chroma_db")
    monkeypatch.setattr(preflight, "PROBE_RETRY_DELAY_SECONDS", 0)
    assert _writer_role.claim_writes()
    return directory


@pytest.fixture
def pauses(monkeypatch):
    """The repair's pauses between move attempts, recorded instead of slept."""
    recorded = []
    monkeypatch.setattr(_preflight_repair, "time", SimpleNamespace(sleep=recorded.append))
    return recorded


def _nobody_else_can_enter(data_dir):
    """True when a probe and a second owner would both be turned away right now."""
    probe = instance_lock._lock_sidecar(data_dir / _index_access.ACCESS_FILENAME)
    if probe is not None:
        os.close(probe)
        return False
    try:
        os.close(instance_lock._lock_guard(data_dir / instance_lock.LOCK_FILENAME))
    except instance_lock.AlreadyRunningError:
        return True
    return False


def test_given_a_probe_when_it_starts_then_it_takes_the_access_guard_before_chroma_and_reads_no_stdin(
    monkeypatch, tmp_path
):
    """Under stdio the child would inherit the client's pipe and could swallow requests meant for the server."""
    monkeypatch.setattr(preflight.config, "data_dir", tmp_path)
    runs = []
    monkeypatch.setattr(preflight.subprocess, "run", lambda command, **options: runs.append((command, options)))

    preflight._probe_chroma(preflight._capture_target())

    [(command, options)] = runs
    code = command[command.index("-c") + 1]
    assert options["stdin"] == subprocess.DEVNULL
    assert code.index("target = _enter(") < code.index("_read_index(target)")


class _TickingClock:
    """preflight's clock: each reading is one second later, and sleeps are only recorded."""

    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def monotonic(self):
        self.now += 1
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)


def test_given_the_index_busy_until_the_limit_when_probing_then_it_is_neither_validated_moved_nor_held(
    monkeypatch, data_dir
):
    """Contention is not damage, and a probe that never got inside is no validation either."""
    probes, clock = [], _TickingClock()
    monkeypatch.setattr(
        preflight, "_probe_chroma", lambda target, timeout_seconds=30: probes.append("busy") or busy(target)
    )
    monkeypatch.setattr(preflight, "time", clock)
    monkeypatch.setattr(_index_access, "WAIT_SECONDS", 3)

    with pytest.raises(_index_access.IndexAccessBusy):
        preflight._probe_and_repair()

    kept = ((data_dir / "chroma_db").exists(), (data_dir / "backups").exists(), _index_access.is_held())
    assert (probes, clock.sleeps, kept) == (
        ["busy"] * 3,
        [preflight.PROBE_RETRY_DELAY_SECONDS] * 2,
        (True, False, False),
    )


def test_given_the_index_busy_then_free_when_probing_then_a_probe_that_got_inside_validates_it_before_the_hold(
    monkeypatch, data_dir
):
    events, results = [], [busy, busy, healthy]

    def probe(target, timeout_seconds=30):
        result = results.pop(0)
        events.append("busy" if result is busy else "validated")
        return result(target)

    real_hold = _index_access.hold
    monkeypatch.setattr(preflight, "_probe_chroma", probe)
    monkeypatch.setattr(_index_access, "hold", lambda *, path: events.append("hold") or real_hold(path=path))

    moved = preflight._probe_and_repair()

    assert (moved, events, (data_dir / "chroma_db").exists()) == (False, ["busy", "busy", "validated", "hold"], True)


def test_given_this_process_holds_the_index_access_when_it_starts_a_probe_then_it_refuses_instead_of_waiting_on_itself(
    monkeypatch, data_dir
):
    _index_access.hold(wait_seconds=0)
    monkeypatch.setattr(preflight.subprocess, "run", lambda *args, **options: pytest.fail("no probe may start"))

    with pytest.raises(RuntimeError, match="holds the index access guard"):
        preflight._probe_chroma(preflight._capture_target())


def test_given_a_file_held_for_one_attempt_when_moving_the_index_aside_then_it_retries_alone_and_succeeds(
    monkeypatch, data_dir, pauses
):
    """Each attempt runs with no other owner possible and no probe able to enter."""
    real_rename, attempts = os.rename, []

    def rename(src, dst):
        attempts.append(_nobody_else_can_enter(data_dir))
        if len(attempts) == 1:
            raise LOCKED
        real_rename(src, dst)

    monkeypatch.setattr(preflight, "_probe_chroma", native_crash)
    monkeypatch.setattr(os, "rename", rename)

    moved = preflight._probe_and_repair()

    assert (moved, attempts, pauses, (data_dir / "chroma_db").exists()) == (True, [True, True], [1], False)


def test_given_a_file_held_through_every_retry_when_moving_the_index_aside_then_it_exits_75_and_keeps_it(
    monkeypatch, data_dir, pauses
):
    def rename(src, dst):
        raise LOCKED

    monkeypatch.setattr(preflight, "_probe_chroma", native_crash)
    monkeypatch.setattr(os, "rename", rename)

    with pytest.raises(SystemExit) as exited:
        preflight._probe_and_repair()

    expected_pauses = list(_preflight_repair.REPAIR_RETRY_DELAYS_SECONDS)
    assert (exited.value.code, pauses, (data_dir / "chroma_db").exists()) == (75, expected_pauses, True)


def test_given_a_process_that_stays_inside_the_index_when_the_owner_waits_then_it_gives_up_with_a_clear_error(
    data_dir,
):
    inside = instance_lock._lock_sidecar(data_dir / _index_access.ACCESS_FILENAME)
    try:
        with pytest.raises(_index_access.IndexAccessBusy) as busy:
            _index_access.hold(wait_seconds=0)
    finally:
        os.close(inside)

    assert (_index_access.is_held(), "restart this client" in str(busy.value)) == (False, True)


def test_given_the_owner_holds_the_index_when_a_late_probe_starts_then_it_exits_busy_without_touching_it(data_dir):
    _index_access.hold(wait_seconds=0)

    with pytest.raises(SystemExit) as exited:
        _index_access.enter_or_exit(data_dir / _index_access.ACCESS_FILENAME)

    assert exited.value.code == _index_access.BUSY_EXIT_CODE
