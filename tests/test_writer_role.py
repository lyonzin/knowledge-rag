"""What a server process may do on its data_dir, and what tools answer meanwhile.

Unit tests of ``mcp_server._writer_role`` on plain functions: no server import,
a fresh RoleState and an empty data_dir per test (tests/conftest.py releases
any claim afterwards). Another process owning data_dir is played by
``instance_lock.writer_lock()``, which locks the guard through its own descriptor.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from mcp_server import _index_access, _writer_role, instance_lock

REPAIR_BLOCKED_EXIT_CODE = 75  # preflight could not move a damaged index aside
CONCURRENT_CALLS = 2
CLAIM_WINDOW_SECONDS = 0.5  # how long a slowed claim waits for a second, concurrent one


@pytest.fixture
def state(monkeypatch, tmp_path):
    """A fresh role over an empty data_dir; each test sets the role it needs."""
    monkeypatch.delenv(instance_lock.ENV_VAR, raising=False)
    monkeypatch.setattr(instance_lock.config, "data_dir", tmp_path)
    fresh = _writer_role.RoleState()
    monkeypatch.setattr(_writer_role, "STATE", fresh)
    return fresh


def _reads_the_index():
    """A tool body that reaches the index the way get_orchestrator() does."""
    _writer_role.raise_unless_index_ready()
    return "served"


def _recording_tool(calls):
    def tool(**arguments):
        calls.append(arguments)
        return "served"

    return tool


def test_given_the_writer_startup_running_when_a_tool_runs_then_it_returns_repairing_with_retry_after(state):
    state.ready.clear()

    answer = json.loads(_writer_role.index_tool(_reads_the_index)())

    assert (answer["status"], answer["retry_after_seconds"]) == ("repairing", _writer_role.RETRY_AFTER_SECONDS)


def test_given_a_failed_startup_check_when_a_tool_runs_then_it_returns_error_with_the_reason(state):
    state.failure = "The startup check of the index failed: OSError('disk full')"

    answer = json.loads(_writer_role.index_tool(_reads_the_index)())

    assert answer == {"status": "error", "message": state.failure}


def test_given_a_free_data_dir_when_claiming_writes_then_this_process_owns_them_and_records_its_pid(state, tmp_path):
    owns = _writer_role.claim_writes()

    assert (owns, instance_lock._read_lock_pid(tmp_path / instance_lock.LOCK_FILENAME)) == (True, os.getpid())


def test_given_a_claim_when_another_owner_tries_the_guard_then_it_is_still_held(state):
    _writer_role.claim_writes()

    with instance_lock.writer_lock() as other_owns:
        pass

    assert other_owns is False


def test_given_another_owner_when_claiming_writes_then_this_process_follows(state):
    with instance_lock.writer_lock():
        owns = _writer_role.claim_writes()

    assert owns is False


def test_given_single_instance_opt_in_when_the_guard_is_busy_then_claiming_raises(state, monkeypatch):
    monkeypatch.setenv(instance_lock.ENV_VAR, "1")
    with instance_lock.writer_lock():
        with pytest.raises(instance_lock.AlreadyRunningError):
            _writer_role.claim_writes()


def test_given_a_live_legacy_pid_file_when_claiming_writes_then_this_process_follows(state, tmp_path):
    (tmp_path / instance_lock.LOCK_FILENAME).write_text(f"{os.getpid()}\n", encoding="utf-8")

    assert _writer_role.claim_writes() is False


def test_given_a_guarded_pid_file_whose_pid_names_another_live_process_when_claiming_then_this_process_owns(
    state, tmp_path
):
    """The OS guard decides: a PID file written under it is stale once the guard is free, live PID or not."""
    lock_path = tmp_path / instance_lock.LOCK_FILENAME
    lock_path.write_text(f"{os.getppid()}\n{instance_lock.GUARDED_MARK}\n", encoding="utf-8")

    owns = _writer_role.claim_writes()

    lines = lock_path.read_text(encoding="utf-8").splitlines()
    assert (owns, lines) == (True, [str(os.getpid()), instance_lock.GUARDED_MARK])


def test_given_a_follower_whose_owner_runs_when_a_tool_runs_then_it_refuses_and_never_runs_the_tool(state):
    calls = []
    _writer_role.become_follower(lambda: pytest.fail("the owner is alive: no take-over"))

    with instance_lock.writer_lock():
        answer = json.loads(_writer_role.index_tool(_recording_tool(calls))(filepath="notes.md"))

    observed = (answer["status"], answer["owner_pid"], answer["owner_running"], answer["hint"], calls)
    assert observed == ("error", os.getpid(), True, _writer_role.HTTP_HINT, [])


def test_given_a_follower_whose_owner_exited_when_a_tool_runs_then_it_takes_over_and_answers_repairing(state):
    startups = []
    _writer_role.become_follower(lambda: startups.append("owner startup"))

    answer = json.loads(_writer_role.index_tool(_reads_the_index)())

    observed = (answer["status"], answer["retry_after_seconds"], startups, state.is_follower)
    assert observed == ("repairing", _writer_role.RETRY_AFTER_SECONDS, ["owner startup"], False)


def test_given_two_calls_at_once_in_a_follower_whose_owner_exited_then_only_one_takes_over(state, monkeypatch):
    startups, claims = [], []
    real_claim = _writer_role.claim_writes

    def slow_claim():
        claims.append(threading.get_ident())  # without the promotion lock, the other call lands here too
        deadline = time.monotonic() + CLAIM_WINDOW_SECONDS
        while len(claims) < CONCURRENT_CALLS and time.monotonic() < deadline:
            time.sleep(0.01)
        return real_claim()

    monkeypatch.setattr(_writer_role, "claim_writes", slow_claim)
    _writer_role.become_follower(lambda: startups.append("owner startup"))
    tool, barrier = _writer_role.index_tool(_reads_the_index), threading.Barrier(CONCURRENT_CALLS)
    answers = []
    workers = [threading.Thread(target=lambda: (barrier.wait(), answers.append(tool()))) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert (startups, sorted(json.loads(answer)["status"] for answer in answers)) == (
        ["owner startup"],
        ["repairing"] * 2,
    )


def test_given_a_follower_when_its_index_gate_runs_then_it_refuses_to_open_the_index(state):
    _writer_role.become_follower(lambda: None)

    with pytest.raises(_writer_role.IndexNotReady) as refused:
        _writer_role.raise_unless_index_ready()

    assert refused.value.status == _writer_role.ERROR


def test_given_a_take_over_whose_startup_cannot_start_when_a_tool_runs_then_it_answers_the_error(state):
    def broken_startup():
        raise RuntimeError("thread limit reached")

    _writer_role.become_follower(broken_startup)

    answer = json.loads(_writer_role.index_tool(_reads_the_index)())

    assert (answer["status"], "thread limit reached" in answer["message"]) == ("error", True)


def test_given_preflight_exits_75_when_the_startup_check_runs_then_the_failure_is_recorded_not_raised(state):
    def blocked_repair():
        raise SystemExit(REPAIR_BLOCKED_EXIT_CODE)

    passed = _writer_role.run_startup_check(blocked_repair)

    # No other session can hold the files then: the message must not send the user after them.
    assert (passed, state.failure, "session" in state.failure) == (False, _writer_role._REPAIR_BLOCKED, False)


def test_given_a_process_inside_the_index_past_the_wait_when_the_startup_check_runs_then_tools_answer_why(state):
    def stuck_probe_inside():
        raise _index_access.IndexAccessBusy("a stuck probe holds the index")

    passed = _writer_role.run_startup_check(stuck_probe_inside)
    answer = json.loads(_writer_role.index_tool(_reads_the_index)())

    assert (passed, answer["status"], answer["message"]) == (False, "error", "a stuck probe holds the index")


@pytest.mark.parametrize("failure", [SystemExit(78), RuntimeError("probe crashed")], ids=["other-exit", "exception"])
def test_given_any_other_preflight_failure_when_the_startup_check_runs_then_tools_answer_error(state, failure):
    def failing_check():
        raise failure

    _writer_role.run_startup_check(failing_check)
    answer = json.loads(_writer_role.index_tool(_reads_the_index)())

    assert answer["status"] == _writer_role.ERROR
