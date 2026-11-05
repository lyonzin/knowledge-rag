"""Auto-repair backup retention in startup preflight (GH #267).

Every repair moved the whole index into a new ``data/backups/auto-repair-*``
folder and none was ever deleted: one install piled up 197 backups (76 GB)
next to a 0.6 GB index.
"""

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

NEW_BACKUP = "auto-repair-20261105-120000"
OLD_BACKUPS = [
    "auto-repair-20000101-000000",
    "auto-repair-20000102-000000",
    "auto-repair-20000103-000000",
    "auto-repair-20000104-000000",
]


class _FrozenClock(datetime):
    """preflight's clock, stopped at ``moment``: a repair creates ``backups/NEW_BACKUP`` unless a test moves it."""

    moment = datetime(2026, 11, 5, 12, 0, 0)

    @classmethod
    def now(cls, tz=None):
        return cls.moment


def _crash_probe_with_backups(tmp_path, monkeypatch, *names):
    """Make every probe crash on a fresh data_dir whose ``backups/`` already holds ``names``."""
    from mcp_server import preflight

    data_dir = tmp_path / "data"
    chroma_dir = data_dir / "chroma_db"
    chroma_dir.mkdir(parents=True)
    (chroma_dir / "chroma.sqlite3").write_text("bad", encoding="utf-8")
    backups = data_dir / "backups"
    for name in names:
        (backups / name).mkdir(parents=True)
        (backups / name / "chroma.sqlite3").write_text("old", encoding="utf-8")

    monkeypatch.setattr(preflight, "datetime", _FrozenClock)
    monkeypatch.setattr(preflight.config, "data_dir", data_dir)
    monkeypatch.setattr(preflight.config, "chroma_dir", chroma_dir)
    monkeypatch.setattr(preflight, "PROBE_RETRY_DELAY_SECONDS", 0)
    result = subprocess.CompletedProcess(args=[], returncode=-11, stdout="", stderr="segfault")
    monkeypatch.setattr(preflight, "_probe_chroma", lambda timeout_seconds=30: result)
    return backups


def _fail_renames(monkeypatch, *failing_calls):
    """Make preflight's ``os.rename`` raise WinError 32 on the given 1-based call numbers."""
    from mcp_server import preflight

    real_rename = preflight.os.rename
    calls = []

    def rename(src, dst):
        calls.append(src)
        if len(calls) in failing_calls:
            raise OSError(32, "The process cannot access the file because it is being used by another process")
        real_rename(src, dst)

    monkeypatch.setattr(preflight.os, "rename", rename)


@pytest.mark.parametrize("existing", [2, 3, 4])
def test_repair_keeps_only_the_three_newest_auto_repair_backups(tmp_path, monkeypatch, existing):
    """GH #267: every repair added a full index copy and none was ever deleted.

    One install piled up 197 backups (76 GB) next to a 0.6 GB index. With two
    old backups nothing goes; past that, only the two newest stay with the new one.
    """
    from mcp_server import preflight

    old = OLD_BACKUPS[:existing]
    backups = _crash_probe_with_backups(tmp_path, monkeypatch, *old)

    assert preflight.run_preflight() is True
    assert (backups / NEW_BACKUP / "chroma_db.segfault").exists()
    assert sorted(path.name for path in backups.iterdir()) == [*old[-2:], NEW_BACKUP]


def test_repair_never_prunes_what_auto_repair_did_not_create(tmp_path, monkeypatch):
    """GH #267: only dirs named exactly ``auto-repair-<8 digits>-<6 digits>`` count.

    Near-miss names, unrelated folders and plain files (one named like a backup)
    are never counted, deleted or even attempted.
    """
    from mcp_server import preflight

    foreign_dirs = ["auto-repair-19990101-000000-old", "auto-repair-manual", "notes", "old-auto-repair-19990101-000000"]
    foreign_files = ["auto-repair-19990102-000000", "README.txt"]
    backups = _crash_probe_with_backups(tmp_path, monkeypatch, *OLD_BACKUPS[:3], *foreign_dirs)
    for name in foreign_files:
        (backups / name).write_text("not a backup", encoding="utf-8")

    assert preflight.run_preflight() is True
    expected = sorted([*OLD_BACKUPS[1:3], *foreign_dirs, *foreign_files, NEW_BACKUP])
    assert sorted(path.name for path in backups.iterdir()) == expected
    assert "could not delete" not in (tmp_path / "data" / "preflight.log").read_text(encoding="utf-8")


def test_backup_that_cannot_be_deleted_does_not_break_the_repair(tmp_path, monkeypatch):
    """GH #267: a file still held open (WinError 32) must not fail the repair.

    The locked folder is logged in one line and kept; the other old backups still go.
    """
    from mcp_server import preflight

    backups = _crash_probe_with_backups(tmp_path, monkeypatch, *OLD_BACKUPS)
    locked = backups / OLD_BACKUPS[0]
    real_rmtree = preflight.shutil.rmtree

    def rmtree_unless_locked(path, *args, **kwargs):
        if Path(path) == locked:
            raise OSError(32, "The process cannot access the file because it is being used by another process")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(preflight.shutil, "rmtree", rmtree_unless_locked)

    assert preflight.run_preflight() is True
    assert not (tmp_path / "data" / "chroma_db").exists()
    assert sorted(path.name for path in backups.iterdir()) == [OLD_BACKUPS[0], *OLD_BACKUPS[2:], NEW_BACKUP]
    log_lines = (tmp_path / "data" / "preflight.log").read_text(encoding="utf-8").splitlines()
    failures = [line for line in log_lines if "could not delete" in line]
    assert len(failures) == 1
    assert str(locked) in failures[0] and "being used by another process" in failures[0]
    assert any("removed 1 old auto-repair backup" in line for line in log_lines)


def test_new_backup_is_kept_even_when_its_stamp_sorts_first(tmp_path, monkeypatch):
    """GH #267: the backup just made is never pruned, even after the clock went back."""
    from mcp_server import preflight

    ahead = ["auto-repair-20261106-000000", "auto-repair-20261107-000000", "auto-repair-20261108-000000"]
    backups = _crash_probe_with_backups(tmp_path, monkeypatch, *ahead)

    assert preflight.run_preflight() is True
    assert sorted(path.name for path in backups.iterdir()) == [NEW_BACKUP, *ahead[1:]]


def test_pruning_never_follows_or_deletes_links(tmp_path, monkeypatch):
    """GH #267: a link named like an old backup is left alone, and so is its target."""
    from mcp_server import preflight

    backups = _crash_probe_with_backups(tmp_path, monkeypatch, *OLD_BACKUPS[:3])
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("not a backup", encoding="utf-8")
    link = backups / "auto-repair-19990101-000000"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # Windows without Developer Mode or admin rights
        pytest.skip(f"cannot create a symlink here: {exc}")

    assert preflight.run_preflight() is True
    assert link.is_symlink() and (outside / "keep.txt").exists()
    assert sorted(path.name for path in backups.iterdir()) == [link.name, *OLD_BACKUPS[1:3], NEW_BACKUP]
    assert "could not delete" not in (tmp_path / "data" / "preflight.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("failing_call", [1, 2], ids=["metadata-rename", "chroma-rename"])
def test_failed_move_prunes_nothing_and_leaves_no_folder(tmp_path, monkeypatch, failing_call):
    """GH #267: exit 75 prunes nothing and leaves no empty folder to take a kept slot."""
    from mcp_server import preflight

    backups = _crash_probe_with_backups(tmp_path, monkeypatch, *OLD_BACKUPS)
    metadata_file = tmp_path / "data" / "index_metadata.json"
    metadata_file.write_text("{}", encoding="utf-8")
    _fail_renames(monkeypatch, failing_call)

    with pytest.raises(SystemExit) as exc_info:
        preflight.run_preflight()

    assert exc_info.value.code == 75
    assert metadata_file.exists() and (tmp_path / "data" / "chroma_db").exists()
    assert sorted(path.name for path in backups.iterdir()) == OLD_BACKUPS


def test_failed_rollback_keeps_the_folder_holding_the_metadata(tmp_path, monkeypatch):
    """GH #267: only an empty folder is removed; one holding a stuck metadata file stays as evidence."""
    from mcp_server import preflight

    backups = _crash_probe_with_backups(tmp_path, monkeypatch)
    (tmp_path / "data" / "index_metadata.json").write_text("{}", encoding="utf-8")
    _fail_renames(monkeypatch, 2, 3)  # the chroma_db move and the metadata rollback both fail

    with pytest.raises(SystemExit):
        preflight.run_preflight()

    assert (backups / NEW_BACKUP / "index_metadata.segfault.json").exists()


def test_failed_attempts_do_not_push_real_backups_out(tmp_path, monkeypatch):
    """GH #267: empty folders left by exit-75 attempts outranked, and so deleted, real backups."""
    from mcp_server import preflight

    backups = _crash_probe_with_backups(tmp_path, monkeypatch, *OLD_BACKUPS[2:])
    for minute in (57, 58, 59):
        with monkeypatch.context() as failing, pytest.raises(SystemExit):
            failing.setattr(_FrozenClock, "moment", datetime(2026, 11, 5, 11, minute))
            _fail_renames(failing, 1)
            preflight.run_preflight()

    assert preflight.run_preflight() is True
    assert sorted(path.name for path in backups.iterdir()) == [*OLD_BACKUPS[2:], NEW_BACKUP]


def test_repairs_with_nothing_to_move_never_push_out_a_real_backup(tmp_path, monkeypatch):
    """GH #267 review: a broken environment fails the probe even when no index is left to move.

    Each such repair left an empty folder that took a kept slot, so the third one deleted
    the only real backup, possibly the healthy index that the first repair had moved aside.
    """
    from mcp_server import preflight

    backups = _crash_probe_with_backups(tmp_path, monkeypatch, OLD_BACKUPS[0])
    shutil.rmtree(tmp_path / "data" / "chroma_db")  # already moved aside by an earlier repair
    broken = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="No module named 'chromadb'")
    monkeypatch.setattr(preflight, "_probe_chroma", lambda timeout_seconds=30: broken)
    for minute in (1, 2, 3):
        monkeypatch.setattr(_FrozenClock, "moment", datetime(2026, 11, 5, 12, minute))
        preflight.run_preflight()

    assert sorted(path.name for path in backups.iterdir()) == [OLD_BACKUPS[0]]
    assert "moved active index" not in (tmp_path / "data" / "preflight.log").read_text(encoding="utf-8")


def test_unlistable_backups_folder_does_not_break_the_repair(tmp_path, monkeypatch):
    """GH #267 review: pruning is best effort, so a failing scandir after the move must not crash startup."""
    from mcp_server import preflight

    backups = _crash_probe_with_backups(tmp_path, monkeypatch, *OLD_BACKUPS)

    def scandir_denied(path):
        raise PermissionError(13, "Access is denied", str(path))

    with monkeypatch.context() as denied:
        denied.setattr(preflight.os, "scandir", scandir_denied)
        assert preflight.run_preflight() is True

    assert sorted(path.name for path in backups.iterdir()) == [*OLD_BACKUPS, NEW_BACKUP]
    assert "could not prune old backups" in (tmp_path / "data" / "preflight.log").read_text(encoding="utf-8")


def test_backup_folder_stat_failure_after_the_move_does_not_break_the_repair(tmp_path, monkeypatch):
    """GH #267 review: Python 3.11's ``Path.exists`` re-raises PermissionError; after the move it is only logged."""
    from mcp_server import preflight

    backups = _crash_probe_with_backups(tmp_path, monkeypatch)
    real_exists = Path.exists

    def exists_denied_for_new_backup(path, *args, **kwargs):
        if path == backups / NEW_BACKUP:
            raise PermissionError(13, "Access is denied", str(path))
        return real_exists(path, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", exists_denied_for_new_backup)

    assert preflight.run_preflight() is True
    assert (backups / NEW_BACKUP / "chroma_db.segfault").exists()
    assert "could not prune old backups" in (tmp_path / "data" / "preflight.log").read_text(encoding="utf-8")
