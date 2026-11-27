"""A follower answers each of the 13 tools with the refusal and never builds an orchestrator.

The owner is played by ``instance_lock.writer_lock()``, which locks the guard
through its own descriptor; tests/conftest.py releases any claim afterwards.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest
from support.tool_calls import TOOL_CALLS, is_refusal

from mcp_server import _writer_role, instance_lock, server


@pytest.fixture
def follower_session(monkeypatch, tmp_path):
    """This process as main() leaves a follower; yields the orchestrator class it must not build."""
    monkeypatch.delenv(instance_lock.ENV_VAR, raising=False)
    monkeypatch.setattr(server.config, "data_dir", tmp_path)
    monkeypatch.setattr(server.config, "rate_limit_enabled", False)
    monkeypatch.setattr(_writer_role, "STATE", _writer_role.RoleState())
    monkeypatch.setattr(server, "_orchestrator", None)
    orchestrator_class = MagicMock()
    monkeypatch.setattr(server, "KnowledgeOrchestrator", orchestrator_class)
    return orchestrator_class


@pytest.mark.parametrize("tool_name", sorted(TOOL_CALLS))
def test_given_a_follower_when_any_tool_is_called_then_it_refuses_without_building_the_orchestrator(
    follower_session, tool_name
):
    _writer_role.become_follower(lambda: pytest.fail("the owner is alive: no take-over"))

    with instance_lock.writer_lock():
        answer = json.loads(getattr(server, tool_name)(**TOOL_CALLS[tool_name]))

    assert (is_refusal(answer), answer["owner_pid"], follower_session.call_count) == (True, os.getpid(), 0)


def test_given_a_follower_whose_owner_exited_when_a_tool_is_called_then_it_answers_repairing_and_starts_the_owner(
    follower_session,
):
    startups = []
    _writer_role.become_follower(lambda: startups.append("owner startup"))

    answer = json.loads(server.search_knowledge(**TOOL_CALLS["search_knowledge"]))

    observed = (answer["status"], startups, _writer_role.STATE.is_follower, follower_session.call_count)
    assert observed == ("repairing", ["owner startup"], False, 0)
