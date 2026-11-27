"""Fail closed on misleading diagnostics, independent transient failures or a replaced index."""

import os
from dataclasses import replace
from pathlib import Path

import pytest
from support import harmless_probe

from mcp_server import _preflight_repair, preflight


@pytest.fixture
def target(tmp_path, monkeypatch):
    chroma = tmp_path / "index"
    chroma.mkdir()
    (chroma / "chroma.sqlite3").write_bytes(b"old database")
    (tmp_path / "index_metadata.json").write_bytes(b"old catalog")
    monkeypatch.setattr(preflight.config, "data_dir", tmp_path)
    monkeypatch.setattr(preflight.config, "chroma_dir", chroma)
    monkeypatch.setattr(preflight, "PROBE_RETRY_DELAY_SECONDS", 0)
    return preflight._capture_target()


@pytest.mark.parametrize(
    "code, events, foreign",
    [
        (0, (), False),
        (0, ("1",), False),
        (1, (), False),
        (139, ("counting",), False),
        (-11, (), False),
        (0xC0000005, (), False),
        (75, (), False),
        (-11, ("counting", "1"), False),
        (-11, ("counting", "read-error"), False),
        (-11, ("counting",), True),
        (65, ("hnsw-load-failed",), False),
        (65, ("counting", "hnsw-load-failed"), True),
        (1, ("counting",), False),
    ],
)
def test_stderr_exit_or_wrong_target_cannot_supply_missing_read_evidence(target, monkeypatch, code, events, foreign):
    reported = replace(target, collection_name="wrong_collection") if foreign else target
    monkeypatch.setattr(
        preflight,
        "_probe_chroma",
        lambda target, timeout_seconds: harmless_probe.reply(
            reported, code, *events, stderr="chromadb.errors.InternalError: Error loading hnsw index"
        ),
    )
    with pytest.raises(RuntimeError, match="Could not verify"):
        preflight.run_preflight()
    assert (target.chroma_dir / "chroma.sqlite3").read_bytes() == b"old database"
    assert not (target.data_dir / "backups").exists()


@pytest.mark.parametrize(
    "exception",
    [
        "PermissionError('Error loading hnsw index')",
        "RuntimeError('Error loading hnsw index')",
        "InternalError('Permission denied: Error loading hnsw index')",
        "InternalError('Error loading hnsw index') from PermissionError('denied')",
    ],
)
def test_unexpected_exception_even_in_count_is_not_hnsw_evidence(target, monkeypatch, exception):
    # Run the production exception boundary and real access guard in a fresh child, without Rust/model startup.
    fake = f"""
import types
chroma = types.ModuleType('chromadb')
errors = types.ModuleType('chromadb.errors')
class NotFoundError(Exception): pass
class InternalError(Exception): pass
class Store:
    def __init__(self, path): pass
    def get_collection(self, **kwargs): return self
    def count(self): raise {exception}
errors.NotFoundError, errors.InternalError = NotFoundError, InternalError
chroma.PersistentClient = Store
sys.modules.update({{'chromadb': chroma, 'chromadb.errors': errors}})
"""
    monkeypatch.setattr(preflight, "_PROBE_BODY", fake + preflight._PROBE_BODY)
    with pytest.raises(RuntimeError, match="Could not verify"):
        preflight.run_preflight()
    assert (target.chroma_dir / "chroma.sqlite3").read_bytes() == b"old database"
    assert not (target.data_dir / "backups").exists()


@pytest.mark.parametrize("code", [-11, 0xC0000005])
def test_repeated_native_crash_only_during_count_of_this_index_can_move(target, monkeypatch, code):
    monkeypatch.setattr(
        preflight, "_probe_chroma", lambda target, timeout_seconds: harmless_probe.reply(target, code, "counting")
    )
    assert preflight.run_preflight()
    [backup] = (target.data_dir / "backups").glob("auto-repair-*/chroma_db.segfault/chroma.sqlite3")
    assert backup.read_bytes() == b"old database"


def test_mixed_unverified_and_hnsw_without_intermediate_success_is_not_confirmation(target, monkeypatch):
    results = iter(
        [
            harmless_probe.reply(target, 1, stderr="database is locked"),
            harmless_probe.reply(target, 65, "counting", "hnsw-load-failed"),
            harmless_probe.reply(target, -11, "counting"),
        ]
    )
    monkeypatch.setattr(preflight, "_probe_chroma", lambda target, timeout_seconds: next(results))
    with pytest.raises(RuntimeError, match="Could not verify"):
        preflight.run_preflight()
    assert not (target.data_dir / "backups").exists()


def test_changed_parent_config_between_probe_and_decision_does_not_redirect_lock_move_or_logs(target, monkeypatch):
    other = target.data_dir / "B"
    other.mkdir()
    (other / "untouched").touch()

    def change_config(snapshot, timeout_seconds):
        monkeypatch.setattr(preflight.config, "data_dir", other)
        monkeypatch.setattr(preflight.config, "chroma_dir", other / "other-index")
        monkeypatch.setattr(preflight.config, "collection_name", "other_collection")
        return harmless_probe.native_crash(snapshot)

    monkeypatch.setattr(preflight, "_probe_chroma", change_config)
    assert preflight.run_preflight()
    assert [p.name for p in other.iterdir()] == ["untouched"]
    assert target.access_path.exists()
    assert not target.chroma_dir.exists()


@pytest.mark.parametrize(
    "swapped, stage",
    [
        ("directory", "between-probes"),
        ("sqlite", "before-decision"),
        ("sqlite", "during-move"),
        ("directory", "move-retry"),
    ],
)
def test_replacement_at_same_path_invalidates_repair_evidence(target, monkeypatch, swapped, stage):
    directory, catalog = target.chroma_dir, target.data_dir / "index_metadata.json"
    db = directory / "chroma.sqlite3"

    def swap():
        old = directory if swapped == "directory" else db
        old.rename(old.with_name("previous"))
        directory.mkdir(exist_ok=True)
        db.write_bytes(b"replacement database")

    real_probe = preflight._probe_chroma
    attempts = []

    def probe(snapshot, timeout_seconds):
        if stage == "between-probes" and attempts:
            # A real child must reject the replacement *before* it can enter the probe body.
            return real_probe(snapshot, timeout_seconds)
        attempts.append(snapshot)
        if (stage == "between-probes" and len(attempts) == 1) or (stage == "before-decision" and len(attempts) == 3):
            swap()
        return harmless_probe.native_crash(snapshot)

    monkeypatch.setattr(preflight, "_probe_chroma", probe)
    monkeypatch.setattr(preflight, "_PROBE_BODY", "(target.data_dir / 'unexpected-read').touch()\n")
    if stage in ("during-move", "move-retry"):
        _swap_during_rename(monkeypatch, catalog, swap, stage)
    with pytest.raises(RuntimeError, match="Could not verify|Index changed"):
        preflight.run_preflight()
    assert db.read_bytes() == b"replacement database"
    assert catalog.read_bytes() == b"old catalog"
    assert not (target.data_dir / "unexpected-read").exists()
    assert not list((target.data_dir / "backups").glob("auto-repair-*"))


def _swap_during_rename(monkeypatch, catalog, swap, stage):
    real_rename = os.rename

    def rename(src, dst):
        if Path(src) == catalog and stage == "move-retry":
            swap()
            raise OSError("test: retry move")
        real_rename(src, dst)
        if Path(src) == catalog:
            swap()

    monkeypatch.setattr(os, "rename", rename)
    monkeypatch.setattr(_preflight_repair, "REPAIR_RETRY_DELAYS_SECONDS", (0,))
