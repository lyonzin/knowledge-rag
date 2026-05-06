from pathlib import Path

import pytest


def test_single_instance_lock_rejects_second_process(monkeypatch, tmp_path):
    from mcp_server import instance_lock

    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)

    with instance_lock.single_instance_lock():
        with pytest.raises(instance_lock.AlreadyRunningError):
            with instance_lock.single_instance_lock():
                pass


def test_single_instance_lock_recovers_stale_pid(monkeypatch, tmp_path):
    from mcp_server import instance_lock

    lock_path = Path(tmp_path) / "knowledge-rag.lock"
    lock_path.write_text("999999999\n", encoding="utf-8")

    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)
    monkeypatch.setattr(instance_lock, "_pid_is_running", lambda pid: False)

    with instance_lock.single_instance_lock():
        assert lock_path.read_text(encoding="utf-8").strip() == str(instance_lock.os.getpid())

    assert not lock_path.exists()
