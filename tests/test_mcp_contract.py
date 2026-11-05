"""MCP tool contract gate (scripts/check_mcp_contract.py): breaking rules, additions and the real server."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_mcp_contract import compare_contracts, main

ROOT = Path(__file__).resolve().parents[1]
DUMP_TIMEOUT_SECONDS = 180
# A function that loses its @mcp.tool() leaves tools/list while getattr(server, name) still finds it.
REGISTERED_TOOLS = frozenset(
    {
        "add_document",
        "add_from_url",
        "evaluate_retrieval",
        "get_document",
        "get_index_stats",
        "get_reindex_status",
        "list_categories",
        "list_documents",
        "reindex_documents",
        "remove_document",
        "search_knowledge",
        "search_similar",
        "update_document",
    }
)
REQUIRED = object()


def parameter(name: str, default: object = REQUIRED, type_: str = "integer") -> dict[str, object]:
    """One parameter as ``--dump`` writes it; required unless it has a default."""
    if default is REQUIRED:
        return {"name": name, "type": type_, "required": True}
    return {"name": name, "type": type_, "required": False, "default": default}


FILEPATH = parameter("filepath", type_="string")
CATEGORY = parameter("category", None, "null|string")
LIMIT = parameter("limit", 100)
OFFSET = parameter("offset", 0)
BASE = {"get_document": [FILEPATH], "list_documents": [CATEGORY, LIMIT, OFFSET]}

BREAKING_CASES = [
    pytest.param({"list_documents": [CATEGORY, LIMIT, OFFSET]}, "tool removed: get_document", id="tool-removed"),
    pytest.param(
        {**BASE, "list_documents": [CATEGORY, LIMIT]},
        "list_documents.offset: parameter removed or renamed",
        id="parameter-removed",
    ),
    pytest.param(
        {**BASE, "list_documents": [CATEGORY, LIMIT, parameter("start", 0)]},
        "list_documents.offset: parameter removed or renamed",
        id="parameter-renamed",
    ),
    pytest.param(
        {**BASE, "list_documents": [CATEGORY, parameter("limit"), OFFSET]},
        "list_documents.limit: became required",
        id="became-required",
    ),
    pytest.param(
        {**BASE, "get_document": [FILEPATH, parameter("encoding", type_="string")]},
        "get_document.encoding: new required parameter",
        id="new-required-parameter",
    ),
    pytest.param(
        {**BASE, "list_documents": [CATEGORY, parameter("limit", 100, "number"), OFFSET]},
        "list_documents.limit: type changed: integer -> number",
        id="type-changed",
    ),
    pytest.param(
        {**BASE, "list_documents": [CATEGORY, parameter("limit", 10), OFFSET]},
        "list_documents.limit: default changed: 100 -> 10",
        id="default-changed",
    ),
    pytest.param(
        {**BASE, "list_documents": [parameter("category", type_="null|string"), LIMIT, OFFSET]},
        "list_documents.category: default removed (was null)",
        id="default-removed",
    ),
    pytest.param(
        {**BASE, "list_documents": [CATEGORY, OFFSET, LIMIT]},
        "list_documents: parameter order changed: ['category', 'limit', 'offset'] -> ['category', 'offset', 'limit']",
        id="order-changed",
    ),
]


@pytest.mark.parametrize(("head", "expected"), BREAKING_CASES)
def test_given_a_breaking_change_when_comparing_then_it_is_reported(head, expected):
    breaks, _ = compare_contracts(BASE, head)
    assert expected in breaks


def test_given_only_additions_when_comparing_then_nothing_breaks_and_they_are_listed():
    head = {**BASE, "list_documents": [CATEGORY, LIMIT, OFFSET, parameter("sort", "path", "string")], "ping": []}
    assert compare_contracts(BASE, head) == ([], ["tool added: ping", "list_documents.sort: new optional parameter"])


def test_given_identical_contracts_when_comparing_then_nothing_is_reported():
    assert compare_contracts(BASE, BASE) == ([], [])


@pytest.mark.parametrize(
    ("head_text", "exit_code"),
    [
        pytest.param(json.dumps({**BASE, "ping": []}), 0, id="addition"),
        pytest.param(json.dumps({"get_document": [FILEPATH]}), 1, id="tool-removed"),
        pytest.param("not json", 2, id="not-json"),
        pytest.param(json.dumps({"version": 1, "modules": {}}), 2, id="not-a-dump"),
    ],
)
def test_given_two_dump_files_when_comparing_then_the_exit_code_tells_the_outcome(tmp_path, head_text, exit_code):
    base_file, head_file = tmp_path / "base.json", tmp_path / "head.json"
    base_file.write_text(json.dumps(BASE), encoding="utf-8")
    head_file.write_text(head_text, encoding="utf-8")
    assert main(["--compare", str(base_file), str(head_file)]) == exit_code


def test_given_this_tree_when_dumping_the_contract_then_every_tool_is_registered(tmp_path):
    """The dump lists tools/list over MCP and ignores the caller's KNOWLEDGE_RAG_DIR."""
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    # Config() raises on an invalid regex, so reading the caller's config.yaml would fail the dump.
    (caller_dir / "config.yaml").write_text('search:\n  lexical_fast_path:\n    patterns: ["("]\n', encoding="utf-8")
    dump_file = tmp_path / "contract.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_mcp_contract.py"), "--dump", str(dump_file)],
        cwd=ROOT,
        env={**os.environ, "KNOWLEDGE_RAG_DIR": str(caller_dir)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=DUMP_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stderr
    missing = REGISTERED_TOOLS - set(json.loads(dump_file.read_text(encoding="utf-8")))
    assert not missing, f"Not registered as MCP tools: {sorted(missing)}"
