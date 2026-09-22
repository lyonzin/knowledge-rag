"""Regression tests for GH #214 — watcher loop on non-content FS events.

Watchdog's ``on_modified`` fires for every metadata event (chmod, xattr,
``utime`` that keeps mtime, cloud-sync client chatter, backup agent scans,
antivirus, OS file indexers). Before the fix each such event enqueued a
full incremental reindex that scanned every indexed file only to skip
them all — the ``0 new, N skipped`` loop reported in #214.

These tests exercise ``DocumentWatcher._is_real_change`` and
``on_modified`` directly, without spinning up a server, so they are
deterministic and finish in milliseconds.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcp_server.server import DocumentWatcher


def _make_orch(known_files: dict[str, tuple[str, int]]) -> MagicMock:
    """Fake orchestrator exposing the two attributes the watcher inspects.

    ``known_files`` maps resolved-path str -> (stored_mtime_iso, stored_size).
    """
    orch = MagicMock()
    orch._source_to_docid = {path: f"docid-{i}" for i, path in enumerate(known_files)}
    orch._indexed_docs = {
        f"docid-{i}": {"file_mtime": mtime, "file_size": size}
        for i, (mtime, size) in enumerate(known_files.values())
    }
    return orch


def _stored_stat(path: Path) -> tuple[str, int]:
    st = path.stat()
    return datetime.fromtimestamp(st.st_mtime).isoformat(), st.st_size


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    p = tmp_path / "note.md"
    p.write_text("# hello\n")
    return p


def test_is_real_change_returns_false_for_unchanged_file(sample_file: Path) -> None:
    """mtime + size match stored → treat watchdog event as ATTRIB-only, drop."""
    orch = _make_orch({str(sample_file.resolve()): _stored_stat(sample_file)})
    watcher = DocumentWatcher(orchestrator_getter=lambda: orch)
    assert watcher._is_real_change(str(sample_file)) is False


def test_is_real_change_returns_true_when_mtime_drifts(sample_file: Path) -> None:
    """mtime moved forward → real edit → enqueue."""
    stored_mtime, size = _stored_stat(sample_file)
    orch = _make_orch({str(sample_file.resolve()): ("1999-01-01T00:00:00", size)})
    watcher = DocumentWatcher(orchestrator_getter=lambda: orch)
    assert watcher._is_real_change(str(sample_file)) is True
    # Guard against accidental symmetry: stored value used above must differ from current
    assert stored_mtime != "1999-01-01T00:00:00"


def test_is_real_change_returns_true_when_size_drifts(sample_file: Path) -> None:
    """Same mtime but different size → real content change → enqueue."""
    mtime, _ = _stored_stat(sample_file)
    orch = _make_orch({str(sample_file.resolve()): (mtime, 999_999)})
    watcher = DocumentWatcher(orchestrator_getter=lambda: orch)
    assert watcher._is_real_change(str(sample_file)) is True


def test_is_real_change_returns_true_for_unknown_file(tmp_path: Path) -> None:
    """New file not yet indexed → real create → enqueue."""
    p = tmp_path / "brand_new.md"
    p.write_text("fresh\n")
    orch = _make_orch({})  # empty index
    watcher = DocumentWatcher(orchestrator_getter=lambda: orch)
    assert watcher._is_real_change(str(p)) is True


def test_is_real_change_returns_true_when_stat_fails(tmp_path: Path) -> None:
    """Path can't be stat'd → treat as change so index_all's delete-detect runs."""
    missing = tmp_path / "does_not_exist.md"
    orch = _make_orch({str(missing.resolve()): ("2020-01-01T00:00:00", 10)})
    watcher = DocumentWatcher(orchestrator_getter=lambda: orch)
    assert watcher._is_real_change(str(missing)) is True


def test_is_real_change_returns_true_when_orchestrator_not_ready(sample_file: Path) -> None:
    """Orchestrator getter raises during early startup → enqueue conservatively."""

    def boom() -> None:
        raise RuntimeError("orchestrator not initialized yet")

    watcher = DocumentWatcher(orchestrator_getter=boom)
    assert watcher._is_real_change(str(sample_file)) is True


def test_on_modified_drops_attrib_only_event_for_indexed_file(sample_file: Path) -> None:
    """End-to-end: ATTRIB-only event on an unchanged indexed file must NOT enqueue."""
    orch = _make_orch({str(sample_file.resolve()): _stored_stat(sample_file)})
    watcher = DocumentWatcher(orchestrator_getter=lambda: orch)
    watcher._schedule_reindex = MagicMock()

    event = MagicMock()
    event.is_directory = False
    event.src_path = str(sample_file)

    watcher.on_modified(event)

    watcher._schedule_reindex.assert_not_called()


def test_on_modified_enqueues_when_content_actually_changed(sample_file: Path) -> None:
    """Real content edit → mtime + size drift → enqueue as before."""
    orch = _make_orch({str(sample_file.resolve()): ("1999-01-01T00:00:00", 1)})
    watcher = DocumentWatcher(orchestrator_getter=lambda: orch)
    watcher._schedule_reindex = MagicMock()

    event = MagicMock()
    event.is_directory = False
    event.src_path = str(sample_file)

    watcher.on_modified(event)

    watcher._schedule_reindex.assert_called_once_with(str(sample_file))


def test_on_modified_ignores_directory_events(tmp_path: Path) -> None:
    """Directory events short-circuit before touching the orch — no matter the state."""
    orch = MagicMock()
    watcher = DocumentWatcher(orchestrator_getter=lambda: orch)
    watcher._schedule_reindex = MagicMock()

    event = MagicMock()
    event.is_directory = True
    event.src_path = str(tmp_path)

    watcher.on_modified(event)

    watcher._schedule_reindex.assert_not_called()
    orch._source_to_docid.get.assert_not_called()


def test_on_modified_ignores_unsupported_suffix(tmp_path: Path) -> None:
    """Non-indexable extension (.exe) is dropped before the change check."""
    p = tmp_path / "installer.exe"
    p.write_bytes(b"MZ\x90\x00")
    orch = MagicMock()
    watcher = DocumentWatcher(orchestrator_getter=lambda: orch)
    watcher._schedule_reindex = MagicMock()

    event = MagicMock()
    event.is_directory = False
    event.src_path = str(p)

    watcher.on_modified(event)

    watcher._schedule_reindex.assert_not_called()


def test_on_created_still_enqueues_unconditionally(sample_file: Path) -> None:
    """on_created must not run the real-change check — a create IS the change."""
    orch = _make_orch({str(sample_file.resolve()): _stored_stat(sample_file)})
    watcher = DocumentWatcher(orchestrator_getter=lambda: orch)
    watcher._schedule_reindex = MagicMock()

    event = MagicMock()
    event.is_directory = False
    event.src_path = str(sample_file)

    watcher.on_created(event)

    watcher._schedule_reindex.assert_called_once_with(str(sample_file))


def test_on_deleted_still_enqueues_unconditionally(sample_file: Path) -> None:
    """on_deleted must not run the real-change check — the file is gone."""
    orch = _make_orch({str(sample_file.resolve()): _stored_stat(sample_file)})
    watcher = DocumentWatcher(orchestrator_getter=lambda: orch)
    watcher._schedule_reindex = MagicMock()

    event = MagicMock()
    event.is_directory = False
    event.src_path = str(sample_file)

    watcher.on_deleted(event)

    watcher._schedule_reindex.assert_called_once_with(str(sample_file))
