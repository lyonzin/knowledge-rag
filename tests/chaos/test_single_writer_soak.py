"""Soak: only the owner of a data_dir opens Chroma, through its writes and a change of owner.

Readers that open Chroma while another process writes to it crash (0xC0000005),
fail with "Error loading hnsw index" and can leave the index unloadable for good.
So only the process that owns data_dir opens it, and only inside its index
access guard; followers answer every tool with a refusal. Each round, in a
fresh data_dir: owner 1 seeds 1,500 vectors (past the default HNSW sync
threshold) and churns while two long-lived followers call tools without pause
and short sessions come and go; owner 1 exits, one follower takes over (the
real preflight probe, open, a served search) and churns on while sessions go
on; then two fresh processes claim data_dir and its access guard, which must
both be free, and reopen the index. No follower or session may open Chroma:
the workers log every opening and refuse any by a process that does not own
data_dir and hold its access guard. The processes run
tests/support/single_writer_soak_worker.py.

    KNOWLEDGE_RAG_SOAK_ROUNDS=60 python -m pytest tests/chaos/test_single_writer_soak.py -m chaos -s
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, contextmanager

import pytest
from support.mcp_stdio import ROOT, isolated_env
from support.single_writer_soak_worker import EVENT_PREFIX, GO, OPENINGS_LOG, OWNER_OPENING, QUERY_RESULTS, SAMPLE_IDS

pytestmark = pytest.mark.chaos

WORKER = ROOT / "tests" / "support" / "single_writer_soak_worker.py"
ROUNDS_ENV = "KNOWLEDGE_RAG_SOAK_ROUNDS"
CHURN_ENV = "KNOWLEDGE_RAG_SOAK_CHURN_SECONDS"
DEFAULT_ROUNDS = 3  # fits the nightly chaos job; the decision run uses 60
DEFAULT_CHURN_SECONDS = 20.0  # per round, split between the two owners
OWNER_CHURN_SECONDS = float(os.environ.get(CHURN_ENV, DEFAULT_CHURN_SECONDS)) / 2
FOLLOWERS = 2
FRESH_PROCESSES = 2
STARTUP_TIMEOUT_SECONDS = 120
EXIT_TIMEOUT_SECONDS = 60
STDERR_TAIL_LINES = 40
EXIT_CODE_MASK = 0xFFFFFFFF  # shows a Windows crash as its NTSTATUS, e.g. 0xc0000005 (access violation)
HNSW_LOAD_FAILURE = "Error loading hnsw index"


class _RoundProblem(AssertionError):
    """What went wrong in a round: the process, the failure and its stderr tail."""


class _Worker:
    """One soak process; its events arrive on stdout as JSON lines after EVENT_PREFIX.

    ``inbox``, when given, also receives (worker, event) so a round can wait on
    several workers at once.
    """

    def __init__(self, label, process, inbox=None):
        self.label, self.process, self.inbox, self.seen, self.last = label, process, inbox, {}, "none"
        self._events, self._stderr = queue.Queue(), []
        self._readers = [
            threading.Thread(target=self._read_events, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for reader in self._readers:
            reader.start()

    def _read_events(self):
        for line in self.process.stdout:
            if line.startswith(EVENT_PREFIX):
                event = json.loads(line[len(EVENT_PREFIX) :])
                self._events.put(event)
                if self.inbox is not None:
                    self.inbox.put((self, event))
        self._events.put(None)

    def _read_stderr(self):
        self._stderr.extend(self.process.stderr)

    def problem(self, message):
        tail = "".join(self._stderr[-STDERR_TAIL_LINES:])
        return _RoundProblem(f"{self.label} {message} (last event: {self.last})\n--- stderr tail ---\n{tail}")

    def _take(self, timeout, waiting_for):
        try:
            event = self._events.get(timeout=max(timeout, 0))
        except queue.Empty:
            raise self.problem(f"sent no {waiting_for!r} event in time") from None
        if event is None:
            raise self.problem(f"exited with code {self._exit_code()} before its {waiting_for!r} event")
        if event["event"] == "error":
            raise self.problem(f"failed:\n{event['error']}")
        self.seen[event["event"]], self.last = event, event["event"]

    def expect(self, name, timeout=STARTUP_TIMEOUT_SECONDS):
        deadline = time.monotonic() + timeout
        while name not in self.seen:
            self._take(deadline - time.monotonic(), name)
        return self.seen[name]

    def has(self, name):
        """Without blocking: whether ``name`` arrived. An error event raises."""
        while name not in self.seen and not self._events.empty():
            self._take(0, name)
        return name in self.seen

    def send(self, line):
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def close_stdin(self):
        with contextlib.suppress(OSError):  # a child that already died leaves a broken pipe
            self.process.stdin.close()

    def finish(self):
        """Wait for a clean exit and the last of its output; anything else is a problem."""
        self.close_stdin()
        try:
            code = self.process.wait(timeout=EXIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            raise self.problem(f"did not exit within {EXIT_TIMEOUT_SECONDS}s") from None
        self._join_readers()
        if code != 0:
            raise self.problem(f"exited with code {code}")
        if any(HNSW_LOAD_FAILURE in line for line in self._stderr):
            raise self.problem(f"logged {HNSW_LOAD_FAILURE!r}")

    def stop(self):
        """Cleanup path: kill the process if it still runs, then release its pipes."""
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=EXIT_TIMEOUT_SECONDS)
        self.close_stdin()
        self._join_readers()
        self.process.stdout.close()
        self.process.stderr.close()

    def _join_readers(self):
        for reader in self._readers:
            reader.join(timeout=EXIT_TIMEOUT_SECONDS)

    def _exit_code(self):
        """stdout closed: wait for the exit and the rest of stderr (a native crash leaves only these)."""
        try:
            code = self.process.wait(timeout=EXIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return "unknown (still running)"
        self._join_readers()
        return f"{code} ({code & EXIT_CODE_MASK:#x})"


@contextmanager
def _spawn(label, round_dir, role, *arguments, inbox=None):
    """Start one worker on the round's data_dir; kill it if the round ends early."""
    paths = os.pathsep.join([str(ROOT), str(ROOT / "tests")])  # mcp_server, and support for the worker
    env = isolated_env(round_dir, KNOWLEDGE_RAG_SINGLE_INSTANCE="0", ANONYMIZED_TELEMETRY="False", PYTHONPATH=paths)
    command = [sys.executable, str(WORKER), role, *map(str, arguments)]
    pipes = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    process = subprocess.Popen(command, cwd=ROOT, env=env, text=True, encoding="utf-8", **pipes)
    worker = _Worker(label, process, inbox)
    try:
        yield worker
    finally:
        worker.stop()


def _expect_lock(worker, owns_writes):
    lock = worker.expect("lock")
    if lock["owns_writes"] is not owns_writes:
        raise worker.problem(f"got owns_writes={lock['owns_writes']}, expected {owns_writes}")
    return lock["pid"]


def _start_followers(running, round_dir, inbox):
    followers = []
    for number in range(1, FOLLOWERS + 1):
        label = f"follower {number}"
        follower = running.enter_context(_spawn(label, round_dir, "follower", OWNER_CHURN_SECONDS, inbox=inbox))
        _expect_lock(follower, owns_writes=False)
        follower.expect("following")
        followers.append(follower)
    return followers


def _sessions_until_done(owner, owner_pid, round_dir):
    """Short follower sessions start and end one after another until the owner's churn ends."""
    sessions, deadline = 0, time.monotonic() + OWNER_CHURN_SECONDS + STARTUP_TIMEOUT_SECONDS
    while not owner.has("done") and time.monotonic() < deadline:
        sessions += 1
        with _spawn(f"session {sessions}", round_dir, "session") as session:
            _expect_lock(session, owns_writes=False)
            named = session.expect("done")["owner_pids"]
            session.finish()
            if named != [owner_pid]:
                raise session.problem(f"named owner(s) {named}, expected {owner_pid}")
    return sessions


def _wait_for_take_over(inbox):
    """Owner 1 is gone: a follower's next call takes over; wait for it to serve."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while True:
        try:
            follower, event = inbox.get(timeout=max(deadline - time.monotonic(), 0))
        except queue.Empty:
            raise _RoundProblem("no follower took over after owner 1 exited") from None
        if event["event"] in ("promoted", "error"):
            probes = follower.expect("owner-ready")["probe_exit_codes"]  # raises on the follower's error event
            if probes != [0]:  # one probe that got inside the index and found it healthy
                raise follower.problem(f"its preflight probes exited with {probes}, expected [0]")
            return follower, event["pid"]


def _stop_followers(followers, owner_pids):
    """The follower that did not take over must have refused all along, naming both owners."""
    for follower in followers:
        follower.close_stdin()
        report = follower.expect("done", EXIT_TIMEOUT_SECONDS)
        follower.finish()
        named = set(report["owner_pids"]) - {None}
        if "promoted" in follower.seen or named != owner_pids:  # expect("done") consumed every earlier event
            raise follower.problem(f"named owners {sorted(named)}, expected {sorted(owner_pids)}")


def _check_openings(round_dir, owner_pids):
    """Every Chroma client of the round belongs to one of its owners, and both owners opened one."""
    log = round_dir / "data" / OPENINGS_LOG
    openings = [line.split() for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    openers = {int(pid) for pid, role in openings if role == OWNER_OPENING}
    if any(role != OWNER_OPENING for _, role in openings) or openers != owner_pids:
        raise _RoundProblem(f"Chroma was opened by {openings}; only the owners {sorted(owner_pids)} may")


def _reopen(round_dir, owner_count):
    """Fresh processes open what the others left behind: the check that readers during writes failed."""
    expected = {"count": owner_count, "hits": min(QUERY_RESULTS, owner_count), "embeddings": SAMPLE_IDS}
    for number in range(1, FRESH_PROCESSES + 1):
        with _spawn(f"fresh process {number}", round_dir, "reopen") as fresh:
            figures = fresh.expect("done")
            fresh.finish()
            observed = {key: figures[key] for key in expected}
            if observed != expected:
                raise fresh.problem(f"reopened {observed}, expected {expected}")


def _run_round(round_dir):
    """One round with a change of owner; returns its figures, or raises _RoundProblem."""
    inbox = queue.Queue()
    with ExitStack() as running:
        first = running.enter_context(_spawn("owner 1", round_dir, "owner", OWNER_CHURN_SECONDS))
        first_pid = _expect_lock(first, owns_writes=True)
        first.expect("seeded")
        followers = _start_followers(running, round_dir, inbox)
        first.send(GO)
        sessions = _sessions_until_done(first, first_pid, round_dir)
        first_cycles = first.expect("done", OWNER_CHURN_SECONDS + STARTUP_TIMEOUT_SECONDS)["cycles"]
        first.finish()  # its process exits: only now may a follower take over
        second, second_pid = _wait_for_take_over(inbox)
        second.send(GO)
        sessions += _sessions_until_done(second, second_pid, round_dir)
        figures = second.expect("done", OWNER_CHURN_SECONDS + STARTUP_TIMEOUT_SECONDS)
        _stop_followers([follower for follower in followers if follower is not second], {first_pid, second_pid})
        second.finish()
    _check_openings(round_dir, {first_pid, second_pid})
    _reopen(round_dir, figures["count"])
    return {"count": figures["count"], "churn_cycles": first_cycles + figures["cycles"], "sessions": sessions}


def _round_count():
    rounds = int(os.environ.get(ROUNDS_ENV, DEFAULT_ROUNDS))
    if rounds < 1:
        raise ValueError(f"{ROUNDS_ENV} must be at least 1, got {rounds}")
    return rounds


@pytest.mark.parametrize("value", ["0", "-1"])
def test_given_nonpositive_soak_rounds_when_configured_then_the_soak_fails_before_spawning(value, monkeypatch):
    monkeypatch.setenv(ROUNDS_ENV, value)
    with pytest.raises(ValueError, match=ROUNDS_ENV):
        _round_count()


def test_given_writes_and_a_change_of_owner_when_followers_call_tools_then_only_owners_open_chroma_and_it_reopens(
    tmp_path,
):
    rounds = _round_count()
    for number in range(1, rounds + 1):
        round_dir, started = tmp_path / f"round-{number}", time.monotonic()
        try:
            figures = _run_round(round_dir)
        except _RoundProblem as problem:
            pytest.fail(f"round {number}/{rounds}: {problem}", pytrace=False)
        print(f"[SOAK] round {number}/{rounds}: {time.monotonic() - started:.1f}s, {figures}")
        shutil.rmtree(round_dir, ignore_errors=True)  # a passing round's index only costs disk space
