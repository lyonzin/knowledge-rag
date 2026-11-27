"""An index alias cannot transfer a guard or a repair decision to someone else's storage."""

import os
import subprocess

import pytest
from support.harmless_probe import native_crash

from mcp_server import _index_access, preflight

DB = "chroma.sqlite3"
CATALOG = "index_metadata.json"


def _seed(directory, contents):
    (directory / "index").mkdir(parents=True)
    (directory / "index" / DB).write_bytes(contents)
    (directory / CATALOG).write_bytes(contents)


def _link(path, dest, kind):
    if kind == "junction":
        if os.name != "nt":
            pytest.skip("Windows junction")
        # Only fixed, suite-generated names enter this native cmd built-in.
        command = ["cmd.exe", "/d", "/c", "mklink", "/J", path.name, str(dest)]
        done = subprocess.run(command, cwd=path.parent, capture_output=True, check=False)
        assert done.returncode == 0, done.stderr
        return
    if kind == "hardlink":
        path.hardlink_to(dest)
        return
    try:
        path.symlink_to(dest, target_is_directory=dest.is_dir())
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows did not grant symlink creation")
        raise


@pytest.mark.parametrize(
    "edge, kind",
    [
        ("directory", "symlink"),
        ("directory", "junction"),
        ("ancestor", "junction"),
        ("sqlite", "symlink"),
        ("sqlite", "hardlink"),
    ],
)
def test_index_alias_cannot_probe_or_repair_a_second_store(tmp_path, monkeypatch, edge, kind):
    owner, other = tmp_path / "A", tmp_path / "B"
    _seed(owner, b"A")
    _seed(other, b"B")
    active = owner / "index"
    if edge == "ancestor":
        _link(owner / "alias", other, kind)
        active = owner / "alias" / "index"
    elif edge == "directory":
        active = owner / "alias"
        _link(active, other / "index", kind)
    else:
        active = owner / "alias"
        active.mkdir()
        _link(active / DB, other / "index" / DB, kind)
    monkeypatch.setattr(preflight.config, "data_dir", owner)
    monkeypatch.setattr(preflight.config, "chroma_dir", active)
    monkeypatch.setattr(preflight, "PROBE_RETRY_DELAY_SECONDS", 0)
    calls = []

    def crash(target, timeout_seconds):
        calls.append(target.chroma_dir)
        return native_crash(target)

    monkeypatch.setattr(preflight, "_probe_chroma", crash)
    with pytest.raises(RuntimeError, match="alias"):
        preflight.run_preflight()
    assert not calls
    for store, contents in ((owner, b"A"), (other, b"B")):
        assert (store / "index" / DB).read_bytes() == (store / CATALOG).read_bytes() == contents
        assert not (store / "backups").exists()


@pytest.mark.parametrize("kind", ["symlink", "junction"])
def test_alias_of_entire_data_dir_keeps_the_physical_guard_with_its_index(tmp_path, monkeypatch, kind):
    owner, alias = tmp_path / "A", tmp_path / "alias"
    _seed(owner, b"A")
    _link(alias, owner, kind)
    monkeypatch.setattr(preflight.config, "data_dir", alias)
    monkeypatch.setattr(preflight.config, "chroma_dir", alias / "index")
    monkeypatch.setattr(preflight, "_PROBE_BODY", 'print("counting\\n1", flush=True)\n')
    assert preflight.run_preflight() is False
    assert (owner / _index_access.ACCESS_FILENAME).samefile(alias / _index_access.ACCESS_FILENAME)
    assert (owner / "index" / DB).read_bytes() == b"A"


@pytest.mark.parametrize(
    "edge, kind", [("directory", "junction" if os.name == "nt" else "symlink"), ("sqlite", "symlink")]
)
def test_alias_inserted_between_read_and_move_is_rejected_even_if_inode_matches(tmp_path, monkeypatch, edge, kind):
    owner, other = tmp_path / "A", tmp_path / "B"
    _seed(owner, b"A")
    _seed(other, b"B")
    active = owner / "index"
    path = active if edge == "directory" else active / DB
    # A different spelling for the same inode defeats a post-read comparison based on stat alone.
    moved = path.with_name("original")
    monkeypatch.setattr(preflight.config, "data_dir", owner)
    monkeypatch.setattr(preflight.config, "chroma_dir", active)
    monkeypatch.setattr(preflight, "PROBE_RETRY_DELAY_SECONDS", 0)
    calls = []

    def retarget(target, timeout_seconds):
        calls.append(target)
        if len(calls) == preflight.PROBE_ATTEMPTS:
            path.rename(moved)
            _link(path, moved, kind)
        return native_crash(target)

    monkeypatch.setattr(preflight, "_probe_chroma", retarget)
    with pytest.raises(RuntimeError, match="alias"):
        preflight.run_preflight()
    for store, contents in ((owner, b"A"), (other, b"B")):
        assert (store / "index" / DB).read_bytes() == (store / CATALOG).read_bytes() == contents
        assert not (store / "backups").exists()
