"""The preflight's repair half: move a damaged index aside, keep a few backups, log to preflight.log.

Moved out of ``preflight``; ``move_index_aside`` adds retries for a move that
another program blocks. The owner runs it while it holds
both guards: no other owner exists, no probe is inside the index and none can
enter, and followers never open the index, so only an outside program (an
antivirus scan, a backup or sync client, a knowledge-rag older than this lock)
can hold one of its files, and a retry is safe.
"""

from __future__ import annotations

import contextlib
import errno
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NoReturn

from ._preflight_probe import _Target

AUTO_REPAIR_BACKUPS_KEPT = 3  # GH #267: unbounded auto-repair backups once filled a disk (197 copies, 76 GB)
REPAIR_RETRY_DELAYS_SECONDS = (1, 2, 4)  # a scan or sync client usually lets a file go within seconds
REPAIR_BLOCKED_EXIT_CODE = 75  # EX_TEMPFAIL: the index is left as it was
_INCOMPLETE_EXIT_CODE = 65  # EX_DATAERR: restarting cannot finish an interrupted repair
_PENDING_REPAIR = ".auto-repair-pending"
_AUTO_REPAIR_DIR_NAME = re.compile(r"auto-repair-[0-9]{8}-[0-9]{6}")  # what _backup_active_index creates


def _log(msg: str, data_dir: Path) -> None:
    """Emit preflight message to stderr and append to data/preflight.log.

    Persisting to a file matters because stdio MCP clients hide the server's
    stderr in their own logs — without the file, users could only ever see
    the truncated exit code, not the actual Chroma stack trace.
    """
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    with contextlib.suppress(OSError, ValueError):  # a closed pipe must not skip the on-disk diagnostic
        print(line, file=sys.stderr)
    try:
        log_file = data_dir / "preflight.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _stop_incomplete(message: str, data_dir: Path) -> NoReturn:
    # A closed stderr must not turn a failed rollback into a retryable OSError.
    with contextlib.suppress(OSError, ValueError):
        _log(f"[RECOVERY] {message}", data_dir)
    raise SystemExit(_INCOMPLETE_EXIT_CODE)


def _check_pending_repair(data_dir: Path) -> None:
    """A previous process may have stopped between the two renames. Never replay marker contents."""
    try:
        (data_dir / _PENDING_REPAIR).lstat()  # even an unreadable or dangling marker is a stop signal
    except FileNotFoundError:
        return
    _stop_incomplete(
        f"{_PENDING_REPAIR} present; automatic repair and retention stopped. "
        "Check preflight.log and the interrupted-repair procedure before restarting.",
        data_dir,
    )


def move_index_aside(target: _Target, reason: str) -> None:
    """Move the damaged index aside and prune old backups, retrying while a file is held.

    Retry only after a complete rollback. An incomplete transaction stops here
    and also blocks startup and retention after a process restart.
    """
    for delay in (*REPAIR_RETRY_DELAYS_SECONDS, None):
        target.check_identity()  # a previous failed move may have outlived the files that failed the read
        try:
            backup_dir = _backup_active_index(target, reason)
        except OSError as exc:
            _check_pending_repair(target.data_dir)  # e.g. the marker was created but writing its hint failed
            if delay is None:
                _log(
                    f"[PREFLIGHT] could not move index aside ({exc}); leaving it untouched "
                    f"to avoid partial corruption. Retry after freeing file handles.",
                    target.data_dir,
                )
                raise SystemExit(REPAIR_BLOCKED_EXIT_CODE) from exc
            _log(
                f"[PREFLIGHT] could not move index aside ({exc}); a file is held, retrying in {delay}s", target.data_dir
            )
            time.sleep(delay)
            continue
        _prune_after(backup_dir, target.data_dir)
        return


def _prune_after(backup_dir: Path, data_dir: Path) -> None:
    """Log the new backup and drop the oldest; pruning never fails the repair. By Lyon."""
    try:
        if backup_dir.exists():  # GH #267: the empty folder is dropped when there was nothing to move
            _log(f"[RECOVERY] moved active index to {backup_dir}", data_dir)
            _prune_auto_repair_backups(backup_dir, data_dir)
    except OSError as exc:  # best effort: old backups never fail the repair
        _log(f"[RECOVERY] could not prune old backups in {backup_dir.parent} ({exc})", data_dir)


def _clear_pending_repair(marker: Path) -> None:
    """Retry the unlink alone; a held marker is not a reason to move the data again."""
    for delay in (*REPAIR_RETRY_DELAYS_SECONDS, None):
        try:
            marker.unlink()
            return
        except OSError as exc:
            if delay is None or exc.errno not in (errno.EACCES, errno.EPERM, errno.EBUSY):
                _stop_incomplete(
                    f"Could not clear {marker} ({exc}); stopped without pruning. Verify the active or backed-up pair "
                    "before archiving the marker (docs/TROUBLESHOOTING.md, Interrupted index repair).",
                    marker.parent,
                )
            time.sleep(delay)


def _backup_active_index(target: _Target, reason: str) -> Path:
    """Move the catalog first, so an old catalog can never mask an empty index (#216)."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = target.data_dir / "backups" / f"auto-repair-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    marker = target.data_dir / _PENDING_REPAIR
    try:
        # Publish before either rename. Its presence alone blocks restart, even if the process dies mid-move.
        with marker.open("x", encoding="ascii") as intent:
            intent.write(f"{backup_dir.name}/index_metadata.{reason}.json\n")
            intent.flush()
            os.fsync(intent.fileno())
        try:
            _move_pair(target, reason, backup_dir)
        except (OSError, RuntimeError):  # no move or complete rollback; a failed rollback exits instead
            _clear_pending_repair(marker)
            raise
        _clear_pending_repair(marker)
        return backup_dir
    finally:
        with contextlib.suppress(OSError):
            backup_dir.rmdir()  # only an empty attempt; never the folder containing the stranded catalog


def _move_pair(target: _Target, reason: str, backup_dir: Path) -> None:
    metadata_file = target.data_dir / "index_metadata.json"
    metadata_backup = backup_dir / f"index_metadata.{reason}.json"
    try:
        metadata_file.lstat()
    except FileNotFoundError:
        metadata_moved = False  # An index can legitimately have no catalog.
    else:
        os.rename(str(metadata_file), str(metadata_backup))
        metadata_moved = True
    try:
        target.check_identity()
        os.rename(str(target.chroma_dir), str(backup_dir / f"chroma_db.{reason}"))
    except (OSError, RuntimeError):
        if metadata_moved:
            try:
                os.rename(str(metadata_backup), str(metadata_file))
            except OSError as exc:
                _stop_incomplete(
                    f"Catalog rollback failed ({exc}). Keep {metadata_backup}; "
                    "the index remains in place. See the interrupted-repair procedure.",
                    target.data_dir,
                )
        raise


def _prune_auto_repair_backups(new_backup: Path, data_dir: Path) -> None:
    """Keep only the newest AUTO_REPAIR_BACKUPS_KEPT auto-repair backups, ``new_backup`` included."""
    with os.scandir(new_backup.parent) as entries:
        # Symlinks are skipped here; rmtree refuses Windows junctions, so no link is ever followed.
        other_dirs = {entry.name for entry in entries if entry.is_dir(follow_symlinks=False)} - {new_backup.name}
    older = sorted((name for name in other_dirs if _AUTO_REPAIR_DIR_NAME.fullmatch(name)), reverse=True)
    removed = 0
    # new_backup takes one kept slot even when a clock change made its stamp sort below the others.
    for name in older[AUTO_REPAIR_BACKUPS_KEPT - 1 :]:
        try:
            shutil.rmtree(new_backup.parent / name)
            removed += 1
        except OSError as exc:  # e.g. a file still held open on Windows
            _log(
                f"[RECOVERY] could not delete old backup {new_backup.parent / name} ({exc}); will retry next repair",
                data_dir,
            )
    _log(f"[RECOVERY] removed {removed} old auto-repair backup(s)", data_dir)
