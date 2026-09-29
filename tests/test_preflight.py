"""Tests for startup preflight repair.

GH #216 (v4.9.1): retry + timeout + atomic-rename behavior.
"""

import subprocess

import pytest


def test_probe_failure_backs_up_chroma_and_metadata(tmp_path, monkeypatch):
    """A crashing Chroma probe should move active state aside for rebuild."""
    from mcp_server import preflight

    data_dir = tmp_path / "data"
    chroma_dir = data_dir / "chroma_db"
    metadata_file = data_dir / "index_metadata.json"
    chroma_dir.mkdir(parents=True)
    (chroma_dir / "chroma.sqlite3").write_text("bad", encoding="utf-8")
    metadata_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(preflight.config, "data_dir", data_dir)
    monkeypatch.setattr(preflight.config, "chroma_dir", chroma_dir)
    monkeypatch.setattr(preflight, "PROBE_RETRY_DELAY_SECONDS", 0)

    result = subprocess.CompletedProcess(args=[], returncode=-11, stdout="", stderr="segfault")
    monkeypatch.setattr(preflight, "_probe_chroma", lambda timeout_seconds=30: result)

    assert preflight.run_preflight() is True
    assert not chroma_dir.exists()
    assert not metadata_file.exists()

    backups = list((data_dir / "backups").glob("auto-repair-*"))
    assert len(backups) == 1
    assert (backups[0] / "chroma_db.segfault").exists()
    assert (backups[0] / "index_metadata.segfault.json").exists()


def test_successful_probe_leaves_index_in_place(tmp_path, monkeypatch):
    """A healthy Chroma probe should not move files."""
    from mcp_server import preflight

    data_dir = tmp_path / "data"
    chroma_dir = data_dir / "chroma_db"
    chroma_dir.mkdir(parents=True)
    (chroma_dir / "chroma.sqlite3").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(preflight.config, "data_dir", data_dir)
    monkeypatch.setattr(preflight.config, "chroma_dir", chroma_dir)

    result = subprocess.CompletedProcess(args=[], returncode=0, stdout="1", stderr="")
    monkeypatch.setattr(preflight, "_probe_chroma", lambda timeout_seconds=30: result)

    assert preflight.run_preflight() is False
    assert chroma_dir.exists()


def test_transient_failure_recovers_on_retry(tmp_path, monkeypatch):
    """GH #216: failure followed by success must NOT quarantine.

    Regression guard against the pre-v4.9.1 single-attempt quarantine,
    which turned any transient file-lock/GPU-cold-init flake into a
    full-corpus reindex loop.
    """
    from mcp_server import preflight

    data_dir = tmp_path / "data"
    chroma_dir = data_dir / "chroma_db"
    chroma_dir.mkdir(parents=True)
    (chroma_dir / "chroma.sqlite3").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(preflight.config, "data_dir", data_dir)
    monkeypatch.setattr(preflight.config, "chroma_dir", chroma_dir)
    monkeypatch.setattr(preflight, "PROBE_RETRY_DELAY_SECONDS", 0)

    call_count = {"n": 0}

    def flaky_probe(timeout_seconds=30):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return subprocess.CompletedProcess(args=[], returncode=-11, stdout="", stderr="transient")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="42", stderr="")

    monkeypatch.setattr(preflight, "_probe_chroma", flaky_probe)

    assert preflight.run_preflight() is False
    assert chroma_dir.exists()
    assert not (data_dir / "backups").exists()
    assert call_count["n"] == 2


def test_timeout_is_inconclusive_and_keeps_index(tmp_path, monkeypatch):
    """GH #216: subprocess.TimeoutExpired must NOT quarantine.

    A slow-loading index on a warm machine looks like corruption to the
    old code, which crashed main() with an uncaught TimeoutExpired. The
    lazy path is to treat timeout as inconclusive and let the user
    investigate from preflight.log.
    """
    from mcp_server import preflight

    data_dir = tmp_path / "data"
    chroma_dir = data_dir / "chroma_db"
    chroma_dir.mkdir(parents=True)
    (chroma_dir / "chroma.sqlite3").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(preflight.config, "data_dir", data_dir)
    monkeypatch.setattr(preflight.config, "chroma_dir", chroma_dir)

    def timeout_probe(timeout_seconds=30):
        raise subprocess.TimeoutExpired(cmd=[], timeout=timeout_seconds)

    monkeypatch.setattr(preflight, "_probe_chroma", timeout_probe)

    assert preflight.run_preflight() is False
    assert chroma_dir.exists()
    assert not (data_dir / "backups").exists()


def test_rename_failure_preserves_index_and_exits_75(tmp_path, monkeypatch):
    """GH #216: os.rename failure must SystemExit(75), never crash halfway.

    The pre-v4.9.1 shutil.move fallback did copy-tree + rmtree, leaving
    a partial copy in backups/ AND a partially deleted live chroma_db/
    when another process held a segment file open (WinError 32). The
    fix requires the whole quarantine to be atomic or fail cleanly.
    """
    from mcp_server import preflight

    data_dir = tmp_path / "data"
    chroma_dir = data_dir / "chroma_db"
    chroma_dir.mkdir(parents=True)
    (chroma_dir / "chroma.sqlite3").write_text("locked", encoding="utf-8")

    monkeypatch.setattr(preflight.config, "data_dir", data_dir)
    monkeypatch.setattr(preflight.config, "chroma_dir", chroma_dir)
    monkeypatch.setattr(preflight, "PROBE_RETRY_DELAY_SECONDS", 0)

    result = subprocess.CompletedProcess(args=[], returncode=-11, stdout="", stderr="segfault")
    monkeypatch.setattr(preflight, "_probe_chroma", lambda timeout_seconds=30: result)

    def rename_locked(src, dst):
        raise OSError(32, "The process cannot access the file because it is being used by another process")

    monkeypatch.setattr(preflight.os, "rename", rename_locked)

    with pytest.raises(SystemExit) as exc_info:
        preflight.run_preflight()
    assert exc_info.value.code == 75
    assert chroma_dir.exists()


def test_all_attempts_fail_calls_probe_three_times(tmp_path, monkeypatch):
    """GH #216: persistent failure path must run exactly PROBE_ATTEMPTS probes.

    Guards against a future refactor collapsing the retry loop back to
    a single attempt.
    """
    from mcp_server import preflight

    data_dir = tmp_path / "data"
    chroma_dir = data_dir / "chroma_db"
    chroma_dir.mkdir(parents=True)
    (chroma_dir / "chroma.sqlite3").write_text("bad", encoding="utf-8")

    monkeypatch.setattr(preflight.config, "data_dir", data_dir)
    monkeypatch.setattr(preflight.config, "chroma_dir", chroma_dir)
    monkeypatch.setattr(preflight, "PROBE_RETRY_DELAY_SECONDS", 0)

    call_count = {"n": 0}

    def always_fail(timeout_seconds=30):
        call_count["n"] += 1
        return subprocess.CompletedProcess(args=[], returncode=-11, stdout="", stderr="segfault")

    monkeypatch.setattr(preflight, "_probe_chroma", always_fail)

    assert preflight.run_preflight() is True
    assert call_count["n"] == preflight.PROBE_ATTEMPTS
    assert not chroma_dir.exists()
    assert (data_dir / "preflight.log").exists()


def test_chroma_rename_failure_rolls_back_metadata(tmp_path, monkeypatch):
    """GH #216 (PR #217 CodeRabbit + Greptile review): two-step quarantine
    must be all-or-nothing.

    Before this fix `_backup_active_index` moved `chroma_db/` first and
    `index_metadata.json` second; if the second failed with OSError, the
    process exited (75) with chroma_db gone and metadata still live. On
    the next startup the probe reported "missing" (returncode 0), main()
    saw an empty collection, incremental reindex loaded the orphan
    metadata and marked every doc as already indexed — silently
    producing zero results.

    The fix moves metadata first (small, low lock probability), then
    chroma_db inside a try/except that rolls the metadata rename back
    on failure. This test wires os.rename so metadata (call 1) succeeds
    and chroma_db (call 2) fails with WinError 32, then verifies both
    files are back at their original paths after SystemExit(75).
    """
    from mcp_server import preflight

    data_dir = tmp_path / "data"
    chroma_dir = data_dir / "chroma_db"
    metadata_file = data_dir / "index_metadata.json"
    chroma_dir.mkdir(parents=True)
    (chroma_dir / "chroma.sqlite3").write_text("bad", encoding="utf-8")
    metadata_file.write_text('{"docs": []}', encoding="utf-8")

    monkeypatch.setattr(preflight.config, "data_dir", data_dir)
    monkeypatch.setattr(preflight.config, "chroma_dir", chroma_dir)
    monkeypatch.setattr(preflight, "PROBE_RETRY_DELAY_SECONDS", 0)

    result = subprocess.CompletedProcess(args=[], returncode=-11, stdout="", stderr="segfault")
    monkeypatch.setattr(preflight, "_probe_chroma", lambda timeout_seconds=30: result)

    # os.rename call order: 1) metadata → backup (OK), 2) chroma_db → backup
    # (FAIL, simulated WinError 32), 3) metadata rollback (OK).
    real_rename = preflight.os.rename
    call_count = {"n": 0}

    def selective_rename(src, dst):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError(32, "chroma_db still locked by previous PID")
        return real_rename(src, dst)

    monkeypatch.setattr(preflight.os, "rename", selective_rename)

    with pytest.raises(SystemExit) as exc_info:
        preflight.run_preflight()

    assert exc_info.value.code == 75
    # Rollback contract: both files at their pre-preflight paths.
    assert metadata_file.exists(), "metadata rollback failed — orphan state on restart"
    assert metadata_file.read_text(encoding="utf-8") == '{"docs": []}', "metadata content changed"
    assert chroma_dir.exists(), "chroma_db moved despite the OSError on its rename"
    # Rename call sequence: metadata-forward, chroma_db-forward (raises), metadata-rollback.
    assert call_count["n"] == 3
