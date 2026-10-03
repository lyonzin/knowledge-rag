"""Kernel-backed exclusion and recovery use real processes on every supported OS."""

import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mcp_server import instance_lock


def _contender(data_dir, events, messages, crash=False):
    """Synchronize contenders before acquire, and keep the winning lease alive."""
    os.environ[instance_lock.ENV_VAR] = "1"
    instance_lock.config.data_dir = Path(data_dir)
    start, release = events
    messages.put(("ready", os.getpid()))
    assert start.wait(timeout=15)
    try:
        with instance_lock.single_instance_lock():
            messages.put(("acquired", os.getpid()))
            if crash:
                # Queue uses a feeder thread: publish ownership before abrupt exit.
                messages.close()
                messages.join_thread()
            assert release.wait(timeout=15)
            if crash:
                os._exit(17)
    except instance_lock.AlreadyRunningError:
        messages.put(("rejected", os.getpid()))


def _stop_processes(processes):
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        assert not process.is_alive()


def test_real_processes_serialize_stale_pid_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_RAG_DIR", str(tmp_path))
    (tmp_path / instance_lock.LOCK_FILENAME).write_text("999999999\n", encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    start, release, messages = context.Event(), context.Event(), context.Queue()
    processes = [context.Process(target=_contender, args=(str(tmp_path), (start, release), messages)) for _ in range(2)]
    try:
        for process in processes:
            process.start()
        assert [messages.get(timeout=15)[0] for _ in processes] == ["ready", "ready"]
        start.set()
        outcomes = [messages.get(timeout=15) for _ in processes]
        assert sorted(outcome for outcome, _ in outcomes) == ["acquired", "rejected"]
        winner = next(pid for outcome, pid in outcomes if outcome == "acquired")
        assert instance_lock._read_lock_pid(tmp_path / instance_lock.LOCK_FILENAME) == winner
    finally:
        start.set()
        release.set()
        _stop_processes(processes)
        messages.close()
        messages.join_thread()
    assert all(process.exitcode == 0 for process in processes)
    assert not (tmp_path / instance_lock.LOCK_FILENAME).exists()
    assert (tmp_path / instance_lock.GUARD_FILENAME).exists()


def test_os_releases_guard_after_process_crashes(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_RAG_DIR", str(tmp_path))
    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)
    context = multiprocessing.get_context("spawn")
    start, release, messages = context.Event(), context.Event(), context.Queue()
    process = context.Process(target=_contender, args=(str(tmp_path), (start, release), messages, True))
    process.start()
    try:
        assert messages.get(timeout=15)[0] == "ready"
        start.set()
        assert messages.get(timeout=15) == ("acquired", process.pid)
        with pytest.raises(instance_lock.AlreadyRunningError):
            with instance_lock.single_instance_lock():
                pytest.fail("Live child must own the data directory")
    finally:
        start.set()
        release.set()
        _stop_processes([process])
        messages.close()
        messages.join_thread()
    assert process.exitcode == 17
    lock_path = tmp_path / instance_lock.LOCK_FILENAME
    assert instance_lock._read_lock_pid(lock_path) == process.pid
    with instance_lock.single_instance_lock():
        assert instance_lock._read_lock_pid(lock_path) == os.getpid()
    assert not lock_path.exists()


def test_delayed_stale_pid_probe_cannot_delete_successor_lock(tmp_path, monkeypatch):
    """Pause A after observing stale metadata; B must fail before touching the PID."""
    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)
    lock_path = tmp_path / instance_lock.LOCK_FILENAME
    lock_path.write_text("999999999\n", encoding="utf-8")
    observed, continue_probe, acquired, release = [threading.Event() for _ in range(4)]

    def paused_probe(pid):
        observed.set()
        assert continue_probe.wait(timeout=5)
        return False

    def first_contender():
        with instance_lock.single_instance_lock():
            acquired.set()
            assert release.wait(timeout=5)

    monkeypatch.setattr(instance_lock, "_pid_is_running", paused_probe)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(first_contender)
        try:
            assert observed.wait(timeout=5)
            with pytest.raises(instance_lock.AlreadyRunningError):
                with instance_lock.single_instance_lock():
                    pytest.fail("Second contender stole the pending acquisition")
            continue_probe.set()
            assert acquired.wait(timeout=5)
            assert instance_lock._read_lock_pid(lock_path) == os.getpid()
        finally:
            continue_probe.set()
            release.set()
        first.result(timeout=5)


def test_guard_stays_effective_when_pid_metadata_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)
    with instance_lock.single_instance_lock() as lock_path:
        lock_path.unlink()
        with pytest.raises(instance_lock.AlreadyRunningError, match="OS lock held"):
            with instance_lock.single_instance_lock():
                pytest.fail("OS exclusion must not depend on mutable PID metadata")


def test_live_legacy_pid_rejects_acquisition_without_removing_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)
    lock_path = tmp_path / instance_lock.LOCK_FILENAME
    lock_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    with pytest.raises(instance_lock.AlreadyRunningError):
        with instance_lock.single_instance_lock():
            pytest.fail("A legacy server still owns the directory")
    assert instance_lock._read_lock_pid(lock_path) == os.getpid()
    lock_path.unlink()
    with instance_lock.single_instance_lock():
        assert instance_lock._read_lock_pid(lock_path) == os.getpid()
