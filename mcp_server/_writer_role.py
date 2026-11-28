"""Which process owns data_dir, and what tools answer meanwhile.

Only the owner opens ChromaDB: a second writer left HNSW segments unloadable,
and so did readers that opened it while it was written. The first server
process claims data_dir for the rest of its life (``claim_writes``); later ones
follow: every tool answers a refusal that names the owner, and the first call
after the owner's process is gone takes over. An owner answers the MCP
handshake at once and checks, opens and prepares the index on a background
thread: until then tools answer ``repairing``, and a failure turns every
answer into ``error`` instead of an exit.

Code reads ``STATE`` through this module, so tests can replace the whole object.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Callable, Iterator, Optional, ParamSpec

P = ParamSpec("P")

RETRY_AFTER_SECONDS = 5
READY = "ready"
REPAIRING = "repairing"
ERROR = "error"
# No other knowledge-rag session can hold the index files at that point (see
# _preflight_repair), so the message names the programs that can.
_REPAIR_BLOCKED = (
    "The index is damaged and could not be moved aside for the repair: another program kept one of its "
    "files open through every retry (an antivirus scan, a backup or sync client, or a knowledge-rag older "
    "than this version). Close it, then restart this client (details in preflight.log)."
)
_OWNER_CHECKING = (
    "The index is being checked, opened and prepared after startup (a damaged index is repaired first). "
    f"Retry in {RETRY_AFTER_SECONDS} seconds."
)
_TOOK_OVER = (
    "The knowledge-rag process that owned this data directory exited, so this session took over and is "
    f"checking and opening the index. Retry in {RETRY_AFTER_SECONDS} seconds."
)
_FOLLOWING = (
    "Another knowledge-rag process (pid {pid}) owns this data directory, so this session does not open the "
    "index and serves no tool. It takes over on the first call after that process exits."
)
_OWNER_LEAVING = (
    "The knowledge-rag process that owns this data directory is exiting; this session takes over on a call "
    "after it is gone."
)
_NOT_OPENED_HERE = "This knowledge-rag session follows another process and never opens the index."
HTTP_HINT = (
    "To use one index from several clients at once, run one shared server "
    "(knowledge-rag --transport streamable-http) and connect every client to it; see docs/single-instance.md."
)

# This process's claim (guard descriptor, PID file): only the process's death releases it.
_claimed: Optional[tuple[int, Path]] = None


def _set_event() -> threading.Event:
    """An event that starts set, so code that never runs main() is never gated. By Lyon."""
    event = threading.Event()
    event.set()
    return event


@dataclass
class RoleState:
    """This process's role on data_dir: main() sets it, every tool call reads it.

    ``ready`` starts set, so the library and tests that never run main() see
    the index as always ready. ``start_owner`` starts the owner's startup if
    this follower takes over; ``promotion`` lets one call at a time try. By Lyon.
    """

    ready: threading.Event = field(default_factory=_set_event)
    failure: Optional[str] = None
    is_follower: bool = False
    start_owner: Optional[Callable[[], None]] = None
    promotion: threading.Lock = field(default_factory=threading.Lock)
    repair: threading.Lock = field(default_factory=threading.Lock)
    transport_closed: bool = False


STATE = RoleState()


@contextmanager
def startup_repair_guard() -> Iterator[None]:
    """EOF either precedes the move or waits for its completion, including rollback. By Lyon."""
    state = STATE
    with state.repair:
        if state.transport_closed:
            raise RuntimeError("Transport closed before index repair; the index was not moved.")
        yield


def close_startup_repair() -> None:
    """Close admission to repair without joining probes, index opening, or follower threads. By Lyon."""
    state = STATE
    with state.repair:
        state.transport_closed = True


class IndexNotReady(RuntimeError):
    """The index cannot serve this call; ``status`` is "repairing" (retry later) or "error"."""

    def __init__(self, status: str, message: str) -> None:
        """Keep the answer status next to the message the tool returns. By Lyon."""
        super().__init__(message)
        self.status = status


def claim_writes() -> bool:
    """Own data_dir until this process dies; False when another process owns it.

    The guard's descriptor is never closed: only the process's death releases
    it, so a daemon thread still writing while the interpreter shuts down never
    meets a new owner's preflight probe, itself a reader. The PID file goes at
    exit. With the opt-in variable set, a busy lock raises AlreadyRunningError.
    By Lyon.
    """
    from . import instance_lock

    global _claimed
    lock_path = instance_lock._lock_path()
    if _claimed is not None:
        if _claimed[1] != lock_path:
            raise RuntimeError(f"this process already owns {_claimed[1].parent}, not {lock_path.parent}")
        return True
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = instance_lock._lock_guard(lock_path)
        try:
            instance_lock._write_pid(lock_path)
        except BaseException:
            os.close(fd)
            raise
    except instance_lock.AlreadyRunningError:
        if instance_lock.single_instance_enabled():
            raise
        return False
    _claimed = (fd, lock_path)
    atexit.register(instance_lock._remove_if_ours, lock_path)
    return True


def _release_claim() -> None:
    """Tests only: give up this process's claim and index access the way its death would. By Lyon."""
    from . import _index_access, instance_lock

    global _claimed
    _index_access._release()
    if _claimed is None:
        return
    fd, lock_path = _claimed
    _claimed = None
    atexit.unregister(instance_lock._remove_if_ours)
    instance_lock._remove_if_ours(lock_path)
    os.close(fd)


def record_failure(reason: str, exc: Optional[BaseException] = None) -> bool:
    """Make every tool answer ``error`` with ``reason`` until restart; always False.

    A background startup has no caller to raise to and no exit code to set,
    so the reason goes where users look: the tool answers and stderr. By Lyon.
    """
    STATE.failure = reason
    print(f"[STARTUP] {reason}", file=sys.stderr)
    if exc is not None:
        traceback.print_exception(exc, file=sys.stderr)
    return False


def run_startup_check(run_preflight: Callable[[], object]) -> bool:
    """Run the preflight on the startup thread; record a failure instead of exiting.

    An uncaught SystemExit ends only the thread, silently: every tool would
    then answer "repairing" forever. By Lyon.
    """
    from . import _index_access
    from ._preflight_repair import REPAIR_BLOCKED_EXIT_CODE

    try:
        run_preflight()
    except SystemExit as exc:
        if exc.code == REPAIR_BLOCKED_EXIT_CODE:
            return record_failure(_REPAIR_BLOCKED)
        return record_failure(f"The startup check of the index stopped with exit code {exc.code} (see preflight.log).")
    except _index_access.IndexAccessBusy as exc:
        return record_failure(str(exc))
    except Exception as exc:
        return record_failure(f"The startup check of the index failed: {exc!r}", exc)
    return True


def become_follower(start_owner: Callable[[], None]) -> None:
    """Follow the owner of data_dir; ``start_owner`` runs if this process takes over. By Lyon."""
    from .config import config

    STATE.is_follower = True
    STATE.start_owner = start_owner
    STATE.ready.clear()
    print(
        f"[WRITER] Process {_owner_pid()} owns {config.data_dir}: this session does not open the index and "
        "answers every tool with a refusal until that process exits, then takes over.",
        file=sys.stderr,
    )


def raise_unless_index_ready() -> None:
    """Raise IndexNotReady while this process must not open or use the index. By Lyon."""
    state = STATE
    if state.is_follower:
        raise IndexNotReady(ERROR, _NOT_OPENED_HERE)
    if state.failure is not None:
        raise IndexNotReady(ERROR, state.failure)
    if not state.ready.is_set():
        raise IndexNotReady(REPAIRING, _OWNER_CHECKING)


def index_status() -> str:
    """READY, REPAIRING or ERROR for health probes: the state without its reason, which can hold paths. By Lyon."""
    state = STATE
    if state.is_follower or state.failure is not None:
        return ERROR
    return READY if state.ready.is_set() else REPAIRING


def _start_owner(state: RoleState) -> None:
    """Run the owner's startup after a take-over; a failure becomes every tool's error. By Lyon."""
    from .config import config

    print(f"[WRITER] The owner of {config.data_dir} exited; pid {os.getpid()} took over.", file=sys.stderr)
    try:
        if state.start_owner is None:
            raise RuntimeError("become_follower() registered no owner startup")
        state.start_owner()
    except Exception as exc:
        record_failure(f"Taking over the data directory failed: {exc!r}", exc)


def _take_over_or_refuse(state: RoleState) -> Optional[str]:
    """A follower's answer: the refusal, or "repairing" when this call took over; None once it owns.

    The promotion lock lets one call at a time try, so two calls never start
    two owner startups. By Lyon.
    """
    with state.promotion:
        if not state.is_follower:
            return None
        if not claim_writes():
            return follower_refusal()
        state.is_follower = False
        state.ready.clear()
        _start_owner(state)
        failure = state.failure
    return _not_ready_response(IndexNotReady(ERROR, failure) if failure else IndexNotReady(REPAIRING, _TOOK_OVER))


def _owner_pid() -> Optional[int]:
    """PID on line 1 of the lock file: the process that owns data_dir, if any. By Lyon."""
    from . import instance_lock

    return instance_lock._read_lock_pid(instance_lock._lock_path())


def _not_ready_response(exc: IndexNotReady) -> str:
    """The JSON a tool returns in place of its result while the index cannot serve. By Lyon."""
    payload: dict[str, object] = {"status": exc.status, "message": str(exc)}
    if exc.status == REPAIRING:
        payload["retry_after_seconds"] = RETRY_AFTER_SECONDS
    return json.dumps(payload)


def follower_refusal() -> str:
    """Why this session serves nothing, which process owns the index, and how to share one. By Lyon."""
    from . import instance_lock

    pid = _owner_pid()
    running = pid is not None and instance_lock._pid_is_running(pid)
    message = _FOLLOWING.format(pid=pid) if running else _OWNER_LEAVING
    return json.dumps(
        {"status": ERROR, "message": message, "owner_pid": pid, "owner_running": running, "hint": HTTP_HINT}
    )


def index_tool(tool: Callable[P, str]) -> Callable[P, str]:
    """Answer a refusal, "repairing" or "error" instead of running while this process cannot serve. By Lyon."""

    @wraps(tool)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> str:
        state = STATE
        if state.is_follower:
            answer = _take_over_or_refuse(state)
            if answer is not None:
                return answer
        try:
            return tool(*args, **kwargs)
        except IndexNotReady as exc:
            return _not_ready_response(exc)

    return guarded
