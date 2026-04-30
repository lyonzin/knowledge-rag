"""Tests for startup preflight repair."""

import subprocess


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
