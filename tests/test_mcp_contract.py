"""MCP tool contract gate (scripts/check_mcp_contract.py): schema reading, breaking rules and the real server."""

from __future__ import annotations

import asyncio
import enum
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Literal, Optional

import pytest
from mcp.server import MCPServer
from pydantic import BaseModel, Field

import scripts.check_mcp_contract as contract_gate
from scripts.check_mcp_contract import compare_contracts, main, read_contract, schema_type, tool_parameters

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
INTEGER = {"type": "integer"}


def parameter(name: str, default: object = REQUIRED, schema: dict | None = None) -> dict[str, object]:
    """One parameter as ``--dump`` writes it; required unless it has a default."""
    schema = schema or INTEGER
    entry = {"name": name, "type": schema_type(schema), "required": default is REQUIRED, "schema": schema}
    if default is not REQUIRED:
        entry["default"] = default
    return entry


def window_tool(*required_fields: str) -> list[dict[str, object]]:
    """A tool whose parameter is a model reached through ``$ref``, as pydantic writes it."""
    model = {
        "properties": {"start": {"title": "Start", "type": "integer"}, "end": {"title": "End", "type": "integer"}},
        "required": list(required_fields),
        "title": "Window",
        "type": "object",
    }
    schema = {"$defs": {"Window": model}, "properties": {"window": {"$ref": "#/$defs/Window"}}, "required": ["window"]}
    return tool_parameters(schema)


FILEPATH = parameter("filepath", schema={"type": "string"})
CATEGORY = parameter("category", None, {"anyOf": [{"type": "string"}, {"type": "null"}]})
LIMIT = parameter("limit", 100)
OFFSET = parameter("offset", 0)
METHOD = parameter("method", "auto", {"enum": ["auto", "fts5"], "type": "string"})
TOP = parameter("top", 5, {"minimum": 1, "type": "integer"})
BASE = {
    "get_document": [FILEPATH],
    "list_documents": [CATEGORY, LIMIT, OFFSET],
    "read_window": window_tool("start"),
    "search": [METHOD, TOP],
}

BREAKING_CASES = [
    pytest.param(
        {name: params for name, params in BASE.items() if name != "get_document"},
        "tool removed: get_document",
        id="tool-removed",
    ),
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
        {**BASE, "get_document": [FILEPATH, parameter("encoding", schema={"type": "string"})]},
        "get_document.encoding: new required parameter",
        id="new-required-parameter",
    ),
    pytest.param(
        {**BASE, "list_documents": [CATEGORY, parameter("limit", 100, {"type": "number"}), OFFSET]},
        'list_documents.limit: schema changed: integer {"type": "integer"} -> number {"type": "number"}',
        id="type-changed",
    ),
    pytest.param(
        {**BASE, "list_documents": [CATEGORY, parameter("limit", 100, {"anyOf": [INTEGER, {"type": "null"}]}), OFFSET]},
        'list_documents.limit: schema changed: integer {"type": "integer"} -> '
        'integer|null {"anyOf": [{"type": "integer"}, {"type": "null"}]}',
        id="widening-fails-closed",
    ),
    pytest.param(
        {**BASE, "search": [parameter("method", "auto", {"enum": ["auto"], "type": "string"}), TOP]},
        'search.method: schema changed: string {"enum": ["auto", "fts5"], "type": "string"} -> '
        'string {"enum": ["auto"], "type": "string"}',
        id="literal-value-removed",
    ),
    pytest.param(
        {**BASE, "search": [METHOD, parameter("top", 5, {"minimum": 2, "type": "integer"})]},
        'search.top: schema changed: integer {"minimum": 1, "type": "integer"} -> '
        'integer {"minimum": 2, "type": "integer"}',
        id="constraint-changed",
    ),
    pytest.param(
        {**BASE, "read_window": window_tool("start", "end")},
        'read_window.window: schema changed: object {"properties": {"end": {"type": "integer"}, "start": '
        '{"type": "integer"}}, "required": ["start"], "type": "object"} -> object {"properties": {"end": '
        '{"type": "integer"}, "start": {"type": "integer"}}, "required": ["start", "end"], "type": "object"}',
        id="ref-model-changed",
    ),
    pytest.param(
        {**BASE, "list_documents": [CATEGORY, parameter("limit", 10), OFFSET]},
        "list_documents.limit: default changed: 100 -> 10",
        id="default-changed",
    ),
    pytest.param(
        {**BASE, "list_documents": [CATEGORY, LIMIT, parameter("offset", False)]},
        "list_documents.offset: default changed: 0 -> false",
        id="default-json-form",
    ),
    pytest.param(
        {**BASE, "list_documents": [parameter("category", schema=CATEGORY["schema"]), LIMIT, OFFSET]},
        "list_documents.category: default removed (was null)",
        id="default-removed",
    ),
    pytest.param(
        {**BASE, "list_documents": [CATEGORY, OFFSET, LIMIT]},
        "list_documents: parameter positions changed (new parameters go last): "
        "['category', 'limit', 'offset'] -> ['category', 'offset', 'limit']",
        id="order-changed",
    ),
    pytest.param(
        {**BASE, "list_documents": [CATEGORY, parameter("sort", "path", {"type": "string"}), LIMIT, OFFSET]},
        "list_documents: parameter positions changed (new parameters go last): "
        "['category', 'limit', 'offset'] -> ['category', 'sort', 'limit', 'offset']",
        id="parameter-inserted-in-the-middle",
    ),
]


@pytest.mark.parametrize(("head", "expected"), BREAKING_CASES)
def test_given_a_breaking_change_when_comparing_then_it_is_reported(head, expected):
    breaks, _ = compare_contracts(BASE, head)
    assert expected in breaks


def test_given_only_additions_when_comparing_then_nothing_breaks_and_they_are_listed():
    sort = parameter("sort", "path", {"type": "string"})
    head = {**BASE, "list_documents": [CATEGORY, LIMIT, OFFSET, sort], "ping": []}
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
        pytest.param("{}", 2, id="empty-contract"),
    ],
)
def test_given_two_dump_files_when_comparing_then_the_exit_code_tells_the_outcome(tmp_path, head_text, exit_code):
    base_file, head_file = tmp_path / "base.json", tmp_path / "head.json"
    base_file.write_text(json.dumps(BASE), encoding="utf-8")
    head_file.write_text(head_text, encoding="utf-8")
    assert main(["--compare", str(base_file), str(head_file)]) == exit_code


def test_given_a_server_without_tools_when_dumping_then_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(contract_gate, "_collect_contract", lambda: {})
    target = tmp_path / "contract.json"
    assert (main(["--dump", str(target)]), target.exists()) == (2, False)


class _Mode(enum.Enum):
    AUTO = "auto"


class _Window(BaseModel):
    start: int
    title: str = "x"


def _probe_server() -> MCPServer:
    """A server whose one tool covers the schema shapes pydantic writes for MCP parameters."""
    probe = MCPServer("contract-probe")

    @probe.tool()
    def lookup(
        folders: list[str],
        limit: Optional[int] = None,
        method: Literal["auto", "fts5"] = "auto",
        top: Annotated[int, Field(ge=1, description="Ignored by the contract")] = 5,
        mode: _Mode = _Mode.AUTO,
        window: Optional[_Window] = None,
    ) -> str:
        return ""

    return probe


def test_given_a_server_when_reading_its_contract_then_schemas_are_canonical():
    """``$ref`` is resolved (with its sibling default), annotations go, a field named ``title`` stays."""
    window = {
        "properties": {"start": INTEGER, "title": {"default": "x", "type": "string"}},
        "required": ["start"],
        "type": "object",
    }
    assert asyncio.run(read_contract(_probe_server())) == {
        "lookup": [
            parameter("folders", schema={"items": {"type": "string"}, "type": "array"}),
            parameter("limit", None, {"anyOf": [INTEGER, {"type": "null"}]}),
            parameter("method", "auto", {"enum": ["auto", "fts5"], "type": "string"}),
            parameter("top", 5, {"minimum": 1, "type": "integer"}),
            parameter("mode", "auto", {"enum": ["auto"], "type": "string"}),
            parameter("window", None, {"anyOf": [window, {"type": "null"}]}),
        ]
    }


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
