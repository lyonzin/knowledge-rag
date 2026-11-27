"""Real stdio servers on one data_dir: only the owner opens the index; a follower takes over after it.

Each server runs tests/support/single_writer_server.py: the real transport,
writer claim and tool gate over a fake index that logs each write, and each
opening, with the PID. Probes run the real prologue, which takes the index
access guard, over a body that imports no Chroma (support/harmless_probe.py).
No model, no Chroma, no user data.
"""

from __future__ import annotations

import json
import queue
import subprocess
import time
from contextlib import ExitStack

import pytest
from support import harmless_probe
from support.mcp_stdio import EXIT_TIMEOUT_SECONDS, ROOT, McpSession, isolated_env, stdio_server
from support.tool_calls import TOOL_CALLS, is_refusal

from mcp_server import _index_access, instance_lock, preflight

SERVER = ROOT / "tests" / "support" / "single_writer_server.py"
WRITES_LOG = "writes.log"
OPEN_ENTRY = "open"
RELEASE_PREFLIGHT_FILE = "release-preflight"
RELEASE_EXIT_FILE = "release-exit"
HOLD_PREFLIGHT_ENV = "KNOWLEDGE_RAG_TEST_HOLD_PREFLIGHT"
HOLD_EXIT_ENV = "KNOWLEDGE_RAG_TEST_HOLD_EXIT"
HARMLESS_PROBE_ENV = "KNOWLEDGE_RAG_TEST_HARMLESS_PROBE"
PROBE_TIMEOUT_ENV = "KNOWLEDGE_RAG_TEST_PROBE_TIMEOUT"
SHORT_PROBE_TIMEOUT_SECONDS = 3
RELEASE_EARLIER_PROBE_FILE = "release-earlier-probe"
NEVER_RELEASED_FILE = "never-released"
PROBING_AGAIN = "[PREFLIGHT] another process is inside the index; probing again"
READY_WAIT_SECONDS = 30
POLL_SECONDS = 0.1
RETRY_AFTER_SECONDS = 5
SEARCH = TOOL_CALLS["search_knowledge"]
ADD = TOOL_CALLS["add_document"]


class _Session(McpSession):
    """A server on the test's data_dir that knows its PID and retries "repairing"."""

    def __init__(self, process, stdout, stderr):
        super().__init__(process, stdout, "single-writer")
        self.stderr = stderr
        self.pid = int(self.call_text("server_pid", {}))

    def wait_for_stderr(self, text):
        """Block until the server logs a line containing ``text``."""
        deadline = time.monotonic() + READY_WAIT_SECONDS
        while time.monotonic() < deadline:
            try:
                line = self.stderr.get(timeout=max(deadline - time.monotonic(), 0))
            except queue.Empty:
                break
            if line is None:
                break
            if text in line:
                return
        raise AssertionError(f"the server never logged {text!r}")

    def call(self, name, arguments):
        return json.loads(self.call_text(name, arguments))

    def call_when_ready(self, name, arguments):
        """Retry while the answer is "repairing", as a client honouring retry_after would."""
        deadline = time.monotonic() + READY_WAIT_SECONDS
        answer = self.call(name, arguments)
        while answer.get("status") == "repairing" and time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
            answer = self.call(name, arguments)
        return answer

    def end_transport(self):
        """Close stdin: the transport ends, the process may still live on."""
        self.process.stdin.close()

    def stop(self):
        self.end_transport()
        self.process.wait(timeout=EXIT_TIMEOUT_SECONDS)


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def start_server(tmp_path):
    """Start servers sharing one data_dir; every one of them stops after the test."""
    with ExitStack() as running:

        def start(hold_preflight=False, hold_exit=False, **extra_env):
            holds = {HOLD_PREFLIGHT_ENV: str(int(hold_preflight)), HOLD_EXIT_ENV: str(int(hold_exit))}
            env = isolated_env(tmp_path, **holds, **extra_env)
            return _Session(*running.enter_context(stdio_server(SERVER, env)))

        yield start


@pytest.fixture
def start_probe(tmp_path, monkeypatch):
    """Start probes as an owner does, the real prologue first, on the test's data_dir; all end after the test."""
    started = []
    directory = tmp_path / "data"
    directory.mkdir(exist_ok=True)
    monkeypatch.setattr(preflight.config, "data_dir", directory)
    monkeypatch.setattr(preflight.config, "chroma_dir", directory / "chroma_db")
    monkeypatch.setattr(preflight, "_PROBE_BODY", harmless_probe.BODY)
    command = preflight._probe_command(preflight._capture_target())

    def start(release=""):
        env = isolated_env(tmp_path, **{harmless_probe.RELEASE_ENV: release})
        started.append(subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL))
        return started[-1]

    yield start
    for probe in started:
        if probe.poll() is None:
            probe.kill()
        probe.wait(timeout=EXIT_TIMEOUT_SECONDS)


def _log_entries(data_dir):
    log = data_dir / WRITES_LOG
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return [(int(pid), entry) for pid, entry in (line.split(" ", 1) for line in lines)]


def _writers_of(data_dir, entry):
    return [pid for pid, logged in _log_entries(data_dir) if logged == entry]


def _wait_for_entry(data_dir, entry):
    deadline = time.monotonic() + READY_WAIT_SECONDS
    while not _writers_of(data_dir, entry):
        assert time.monotonic() < deadline, f"nothing logged {entry!r}"
        time.sleep(POLL_SECONDS)


def test_given_two_stdio_servers_on_one_data_dir_when_both_add_a_document_then_exactly_one_writes(
    start_server, data_dir
):
    owner = start_server()
    follower = start_server()

    written = owner.call_when_ready("add_document", ADD)
    refused = follower.call("add_document", ADD)

    observed = (written["status"], refused["status"], refused["owner_pid"], _writers_of(data_dir, "add_document"))
    assert observed == ("success", "error", owner.pid, [owner.pid])


def test_given_a_follower_when_each_tool_is_called_then_all_refuse_and_it_never_opens_or_writes(start_server, data_dir):
    owner = start_server()
    follower = start_server()
    owner.call_when_ready("search_knowledge", SEARCH)

    answers = {name: follower.call(name, arguments) for name, arguments in TOOL_CALLS.items()}

    refused = {name: is_refusal(answer) and answer["owner_pid"] == owner.pid for name, answer in answers.items()}
    touched = [entry for pid, entry in _log_entries(data_dir) if pid == follower.pid]
    assert (refused, touched) == (dict.fromkeys(TOOL_CALLS, True), [])


def test_given_a_writer_blocked_in_preflight_when_initialize_is_sent_then_it_answers_and_tools_return_repairing(
    start_server,
):
    owner = start_server(hold_preflight=True)  # the handshake completes inside start_server

    answer = owner.call("search_knowledge", SEARCH)

    assert (answer["status"], answer["retry_after_seconds"]) == ("repairing", RETRY_AFTER_SECONDS)


def test_given_the_owner_exited_when_its_follower_is_called_then_it_answers_repairing_then_serves_and_writes(
    start_server, data_dir
):
    owner = start_server()
    follower = start_server()
    owner.call_when_ready("search_knowledge", SEARCH)
    owner.stop()

    first = follower.call("search_knowledge", SEARCH)
    served = follower.call_when_ready("search_knowledge", SEARCH)
    written = follower.call("add_document", ADD)

    observed = (first["status"], served["status"], written["status"], _writers_of(data_dir, "add_document"))
    assert observed == ("repairing", "success", "success", [follower.pid])


def test_given_two_followers_when_the_owner_exits_then_exactly_one_takes_over(start_server):
    owner = start_server()
    followers = [start_server(), start_server()]
    owner.call_when_ready("search_knowledge", SEARCH)
    owner.stop()

    started = [follower.start_call("search_knowledge", SEARCH) for follower in followers]
    statuses = [json.loads(follower.finish_call(call))["status"] for follower, call in zip(followers, started)]

    winners = [follower.pid for follower, status in zip(followers, statuses) if status == "repairing"]
    losers = [follower for follower, status in zip(followers, statuses) if status != "repairing"]
    named = [loser.call("search_knowledge", SEARCH)["owner_pid"] for loser in losers]
    assert (sorted(statuses), named) == (["error", "repairing"], winners)


def test_given_the_owner_exited_when_a_new_server_starts_first_then_the_old_follower_names_the_new_owner(
    start_server, data_dir
):
    first_owner = start_server()
    old_follower = start_server()
    first_owner.call_when_ready("search_knowledge", SEARCH)
    first_owner.stop()

    new_owner = start_server()
    written = new_owner.call_when_ready("add_document", ADD)
    refused = old_follower.call("add_document", ADD)

    observed = (written["status"], refused["status"], refused["owner_pid"], refused["owner_running"])
    assert (observed, _writers_of(data_dir, "add_document")) == (
        ("success", "error", new_owner.pid, True),
        [new_owner.pid],
    )


def test_given_an_owner_whose_transport_ended_when_its_process_still_lives_then_no_follower_takes_over_until_it_dies(
    start_server, data_dir
):
    owner = start_server(hold_exit=True)
    owner.call_when_ready("search_knowledge", SEARCH)
    owner.end_transport()  # main() returns; the process lives on, as a daemon thread still writing would
    follower = start_server()
    while_alive = follower.call("search_knowledge", SEARCH)

    (data_dir / RELEASE_EXIT_FILE).touch()
    owner.process.wait(timeout=EXIT_TIMEOUT_SECONDS)
    after_death = follower.call("search_knowledge", SEARCH)

    observed = (is_refusal(while_alive), while_alive["owner_pid"], after_death["status"])
    assert observed == (True, owner.pid, "repairing")


def test_given_a_probe_of_an_earlier_owner_inside_the_index_when_a_new_owner_starts_then_it_opens_after_that_probe_left(
    start_server, start_probe, data_dir
):
    """An owner that died abruptly can leave its probe running; no maintenance or opening may overlap it.

    The new owner's own probe finds the index busy, which validates nothing: it
    probes again, and opens only after one of its probes got inside.
    """
    earlier_probe = start_probe(release=RELEASE_EARLIER_PROBE_FILE)
    _wait_for_entry(data_dir, harmless_probe.INSIDE)
    owner = start_server(**{HARMLESS_PROBE_ENV: "1"})
    owner.wait_for_stderr(PROBING_AGAIN)  # its own probe found the index busy
    while_inside = (owner.call("search_knowledge", SEARCH)["status"], _writers_of(data_dir, OPEN_ENTRY))

    (data_dir / RELEASE_EARLIER_PROBE_FILE).touch()
    answer = owner.call_when_ready("search_knowledge", SEARCH)

    entries = [entry for _pid, entry in _log_entries(data_dir)]
    assert (while_inside, answer["status"], earlier_probe.wait(EXIT_TIMEOUT_SECONDS), entries) == (
        ("repairing", []),
        "success",
        0,
        [harmless_probe.INSIDE, harmless_probe.LEAVING, harmless_probe.INSIDE, harmless_probe.LEAVING, OPEN_ENTRY],
    )


def test_given_an_owner_using_the_index_when_a_probe_starts_late_then_it_exits_busy_and_never_touches_it(
    start_server, start_probe, data_dir
):
    owner = start_server(**{HARMLESS_PROBE_ENV: "1"})
    owner.call_when_ready("search_knowledge", SEARCH)
    before = _log_entries(data_dir)

    late_probe = start_probe()

    assert (late_probe.wait(EXIT_TIMEOUT_SECONDS), _log_entries(data_dir)) == (_index_access.BUSY_EXIT_CODE, before)


def test_given_its_probe_hangs_past_the_timeout_when_the_owner_starts_then_it_opens_only_once_that_probe_is_gone(
    start_server, data_dir
):
    """A timeout is inconclusive: no quarantine, and the opening waits for the probe to be gone, however it ends."""
    hung = {HARMLESS_PROBE_ENV: "1", PROBE_TIMEOUT_ENV: str(SHORT_PROBE_TIMEOUT_SECONDS)}
    owner = start_server(**hung, **{harmless_probe.RELEASE_ENV: NEVER_RELEASED_FILE})

    answer = owner.call_when_ready("search_knowledge", SEARCH)

    entries = _log_entries(data_dir)
    probe_pid = next(pid for pid, entry in entries if entry == harmless_probe.INSIDE)
    observed = (answer["status"], [entry for _pid, entry in entries], instance_lock._pid_is_running(probe_pid))
    assert (observed, (data_dir / "backups").exists()) == (
        ("success", [harmless_probe.INSIDE, OPEN_ENTRY], False),
        False,
    )
