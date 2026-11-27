"""run_preflight(), the public standalone check: each call owns data_dir and the index access only while it runs.

Real guards and real probe children over a temporary data_dir: each child runs
the real prologue, which takes the index access, then the body of
support/harmless_probe.py, which imports no Chroma and notes in writes.log that
it got inside. Neither _probe_chroma nor a guard is replaced. No model, no sleeps.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from support import harmless_probe
from support.mcp_stdio import ROOT

from mcp_server import _index_access, _preflight_repair, _writer_role, instance_lock, preflight

CHILD_TIMEOUT_SECONDS = 60
BUSY_EXIT_CODE = 75  # EX_TEMPFAIL, as run_preflight already exits when a move is blocked
FREE_CHECK = (
    "import sys\n"
    "from pathlib import Path\n"
    "from mcp_server import _index_access, instance_lock\n"
    "names = (instance_lock.GUARD_FILENAME, _index_access.ACCESS_FILENAME)\n"
    "print(all(instance_lock._lock_sidecar(Path(sys.argv[1]) / name) is not None for name in names))\n"
)
HELD_FILE = OSError(32, "The process cannot access the file because it is being used by another process")


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    """An index in a data_dir that the probe children resolve as well: their base dir is the test's."""
    base = tmp_path / "rag"
    directory = base / "data"
    (directory / "chroma_db").mkdir(parents=True)
    (directory / "chroma_db" / "chroma.sqlite3").touch()
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv(harmless_probe.RELEASE_ENV, raising=False)
    monkeypatch.delenv(instance_lock.ENV_VAR, raising=False)
    monkeypatch.setattr(preflight.config, "data_dir", directory)
    monkeypatch.setattr(preflight.config, "chroma_dir", directory / "chroma_db")
    monkeypatch.setattr(preflight, "_PROBE_BODY", harmless_probe.BODY)
    return directory


def _probes_inside(data_dir):
    """The PID of every probe child that got inside the index, in order."""
    log = data_dir / "writes.log"
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return [pid for pid, entry in (line.split(" ", 1) for line in lines) if entry == harmless_probe.INSIDE]


def _another_process_can_enter(data_dir):
    """Whether a fresh process can take both the writer guard and the index access right now."""
    done = subprocess.run(
        [sys.executable, "-c", FREE_CHECK, str(data_dir)],
        cwd=str(data_dir.parent),
        env={**os.environ, "KNOWLEDGE_RAG_DIR": str(data_dir.parent)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
        check=True,
    )
    return done.stdout.strip() == "True"


def _is_taken(path):
    fd = instance_lock._lock_sidecar(path)
    if fd is None:
        return True
    os.close(fd)
    return False


def test_given_two_standalone_calls_in_one_process_when_run_in_turn_then_each_probes_and_another_process_can_enter_between(
    data_dir,
):
    first = preflight.run_preflight()
    between = _another_process_can_enter(data_dir)
    second = preflight.run_preflight()

    probes = _probes_inside(data_dir)
    assert (first, between, second, len(probes), len(set(probes))) == (False, True, False, 2, 2)


def test_given_another_owner_when_a_standalone_call_runs_then_it_exits_75_and_leaves_its_guards_and_files(data_dir):
    lock_path, access_path = data_dir / instance_lock.LOCK_FILENAME, data_dir / _index_access.ACCESS_FILENAME
    owner_access = instance_lock._lock_sidecar(access_path)
    try:
        with instance_lock.writer_lock() as owns:
            owner_pid_file = lock_path.read_text(encoding="utf-8")
            with pytest.raises(SystemExit) as exited:
                preflight.run_preflight()
            guards = (_is_taken(data_dir / instance_lock.GUARD_FILENAME), _is_taken(access_path))
            pid_file_kept = lock_path.read_text(encoding="utf-8") == owner_pid_file
    finally:
        os.close(owner_access)

    files = ((data_dir / "chroma_db").exists(), (data_dir / "backups").exists(), _probes_inside(data_dir))
    assert (owns, exited.value.code, guards, pid_file_kept, files) == (
        True,
        BUSY_EXIT_CODE,
        (True, True),
        True,
        (True, False, []),
    )


def test_given_this_process_serves_the_index_when_a_standalone_call_runs_then_it_exits_75_without_a_probe(
    monkeypatch, data_dir
):
    """No child on its own guard, and no claim of a validation that did not happen."""
    assert _writer_role.claim_writes()
    _index_access.hold(wait_seconds=0)
    monkeypatch.setattr(preflight.subprocess, "run", lambda *args, **options: pytest.fail("no probe may start"))

    with pytest.raises(SystemExit) as exited:
        preflight.run_preflight()

    kept = (_writer_role._claimed is not None, _index_access.is_held())
    pid = instance_lock._read_lock_pid(data_dir / instance_lock.LOCK_FILENAME)
    assert (exited.value.code, kept, pid) == (BUSY_EXIT_CODE, (True, True), os.getpid())


def test_given_another_process_inside_the_index_when_a_standalone_call_runs_then_it_exits_75_without_quarantine(
    monkeypatch, data_dir
):
    monkeypatch.setattr(_index_access, "WAIT_SECONDS", 0)
    access_path = data_dir / _index_access.ACCESS_FILENAME
    inside = instance_lock._lock_sidecar(access_path)
    try:
        with pytest.raises(SystemExit) as exited:
            preflight.run_preflight()
        still_inside = _is_taken(access_path)
    finally:
        os.close(inside)

    left = ((data_dir / "chroma_db").exists(), (data_dir / "backups").exists())
    owned_after = (data_dir / instance_lock.LOCK_FILENAME).exists()
    assert (exited.value.code, still_inside, left, owned_after) == (BUSY_EXIT_CODE, True, (True, False), False)


def test_given_a_standalone_call_whose_repair_fails_when_it_ends_then_it_holds_no_guard_and_no_pid_file(
    monkeypatch, data_dir
):
    def held_file(src, dst):
        raise HELD_FILE

    monkeypatch.setattr(preflight, "_probe_chroma", harmless_probe.native_crash)
    monkeypatch.setattr(preflight, "PROBE_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(_preflight_repair, "REPAIR_RETRY_DELAYS_SECONDS", ())
    monkeypatch.setattr(os, "rename", held_file)

    with pytest.raises(SystemExit) as exited:
        preflight.run_preflight()

    released = (_another_process_can_enter(data_dir), (data_dir / instance_lock.LOCK_FILENAME).exists())
    assert (exited.value.code, released, (data_dir / "chroma_db").exists()) == (BUSY_EXIT_CODE, (True, False), True)
