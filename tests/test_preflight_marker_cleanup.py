"""A busy marker may delay cleanup, never change which catalog and index are moved."""

import errno
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from support.harmless_probe import native_crash

from mcp_server import _preflight_repair as repair
from mcp_server import preflight

MARKER = ".auto-repair-pending"
CATALOG = "index_metadata.json"
SQLITE = "chroma.sqlite3"
OLD = "auto-repair-20000101-000000"


@pytest.fixture
def store(tmp_path, monkeypatch):
    chroma = tmp_path / "chroma_db"
    chroma.mkdir()
    (chroma / SQLITE).write_bytes(b"index bytes")
    (tmp_path / CATALOG).write_bytes(b"catalog bytes")
    (tmp_path / "backups" / OLD).mkdir(parents=True)
    monkeypatch.setattr(preflight.config, "data_dir", tmp_path)
    monkeypatch.setattr(preflight.config, "chroma_dir", chroma)
    monkeypatch.setattr(preflight, "_probe_chroma", native_crash)
    monkeypatch.setattr(preflight, "PROBE_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(repair, "AUTO_REPAIR_BACKUPS_KEPT", 1)  # any pruning would remove OLD
    return tmp_path


def _unlink_while_held(path, unlink):
    if os.name != "nt":  # POSIX can unlink an open file; inject the equivalent busy filesystem error.
        raise OSError(errno.EBUSY, "marker busy")
    with path.open("rb"), pytest.raises(PermissionError) as held:
        unlink(path)  # real Windows handle without FILE_SHARE_DELETE
    assert held.value.winerror == 32
    raise held.value


def _record_moves(monkeypatch, rollback):
    moves, rename = [], os.rename

    def record(src, dst):
        moves.append(Path(src).name)
        if rollback and len(moves) == 2:
            raise PermissionError("index temporarily held")
        rename(src, dst)

    monkeypatch.setattr(os, "rename", record)
    return moves


def _hold_marker(monkeypatch, moves, persistent):
    attempts, unlink = [], Path.unlink

    def remove(path, *args, **kwargs):
        if path.name == MARKER:
            attempts.append(len(moves))
            if persistent or len(attempts) == 1:
                _unlink_while_held(path, unlink)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", remove)
    return attempts


@pytest.mark.parametrize("rollback", [False, True], ids=["after-move", "after-rollback"])
@pytest.mark.parametrize("persistent", [False, True], ids=["transient", "persistent"])
def test_marker_cleanup_retries_unlink_not_the_pair(store, monkeypatch, rollback, persistent):
    moves = _record_moves(monkeypatch, rollback)
    attempts = _hold_marker(monkeypatch, moves, persistent)
    pauses = []
    monkeypatch.setattr(repair, "time", SimpleNamespace(sleep=pauses.append))
    first_cleanup = 3 if rollback else 2  # both forward renames; optionally the catalog rollback as well
    if persistent:
        with pytest.raises(SystemExit) as stopped:
            preflight.run_preflight()
        assert stopped.value.code == 65
        assert attempts == [first_cleanup] * (1 + len(repair.REPAIR_RETRY_DELAYS_SECONDS))
        assert len(moves) == first_cleanup and pauses == list(repair.REPAIR_RETRY_DELAYS_SECONDS)
    else:
        assert preflight.run_preflight() is True
        # On rollback, the original index-move error permits one subsequent *ordinary* repair attempt.
        assert attempts == [first_cleanup, first_cleanup] + ([5] if rollback else [])
        assert pauses == [repair.REPAIR_RETRY_DELAYS_SECONDS[0]] * (2 if rollback else 1)
    assert (store / MARKER).exists() is persistent
    assert (store / "backups" / OLD).exists() is persistent
    _assert_pair(store, in_backup=not (persistent and rollback))


def _assert_pair(store, in_backup):
    live_index, live_catalog = store / "chroma_db" / SQLITE, store / CATALOG
    if in_backup:
        assert not live_index.exists() and not live_catalog.exists()
        [catalog] = (store / "backups").glob("*/index_metadata.segfault.json")
        index = catalog.parent / "chroma_db.segfault" / SQLITE
    else:
        assert not list((store / "backups").glob("*/index_metadata.*.json"))
        index, catalog = live_index, live_catalog
    assert (index.read_bytes(), catalog.read_bytes()) == (b"index bytes", b"catalog bytes")


def test_failed_rollback_stays_pending_and_logs_with_closed_stderr(store, monkeypatch):
    rename = os.rename
    closed = io.StringIO()
    closed.close()

    def fail_moves(src, dst):
        if Path(src).name == CATALOG:
            return rename(src, dst)
        monkeypatch.setattr(repair, "sys", SimpleNamespace(stderr=closed))
        raise PermissionError("test: index move and catalog rollback blocked")

    monkeypatch.setattr(os, "rename", fail_moves)
    with pytest.raises(SystemExit) as stopped:
        preflight.run_preflight()
    assert stopped.value.code == 65 and (store / MARKER).is_file()
    assert "Catalog rollback failed" in (store / "preflight.log").read_text(encoding="utf-8")
    [stranded] = (store / "backups").glob("*/index_metadata.segfault.json")
    assert stranded.read_bytes() == b"catalog bytes" and (store / "chroma_db" / SQLITE).read_bytes() == b"index bytes"


def test_nontransient_marker_io_failure_stops_without_sleep(store, monkeypatch):
    marker = store / MARKER
    marker.touch()

    def fail_unlink(path):
        raise OSError(errno.EIO, "disk error")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    monkeypatch.setattr(repair, "time", SimpleNamespace(sleep=lambda _: pytest.fail("not contention")))
    with pytest.raises(SystemExit) as stopped:
        repair._clear_pending_repair(marker)
    assert (stopped.value.code, marker.is_file()) == (65, True)
