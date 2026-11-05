"""Detect breaking changes to the MCP tool contract of knowledge-rag.

MCP clients call tools by name with JSON arguments, so what they depend on is
what the server advertises in ``tools/list``: the tool names, each tool's
parameter names in order, each parameter's JSON-schema type, which parameters
are required and their default values. Descriptions are not part of it.

    python scripts/check_mcp_contract.py --dump PATH
        Import ``mcp_server`` from the checkout that contains this script, list
        its registered tools through the MCP SDK and write their contract to
        PATH. The import runs with a temporary KNOWLEDGE_RAG_DIR and with
        HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE set, so the user's config and
        index are never read and no model is downloaded.

    python scripts/check_mcp_contract.py --compare BASE HEAD
        Compare two dumps and exit:
            0  = HEAD keeps BASE's contract (new tools and new optional
                 parameters are allowed and listed)
            1  = breaking change: a tool removed, a parameter removed or
                 renamed, a parameter that is now required (new or existing),
                 a changed type, a changed or removed default, or a change in
                 the relative order of the parameters both sides have
            2  = a file is missing or is not a dump

In CI, BASE is dumped from master's code: this script is copied into a
``HEAD^1`` worktree and run there, so a pull request cannot approve its own
break by editing an expected list. An intentional break needs the
``skip-api-surface`` label, set by a maintainer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from mcp.client import Client

REPO_ROOT = Path(__file__).resolve().parent.parent
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
PARAMETER_KEYS = frozenset({"name", "type", "required"})
BYPASS_LABEL = "skip-api-surface"
EXIT_OK = 0
EXIT_BREAKING = 1
EXIT_UNREADABLE = 2

Parameter = dict[str, Any]
Contract = dict[str, list[Parameter]]


def schema_type(schema: dict[str, Any]) -> str:
    """Render a parameter's JSON-schema type, e.g. ``integer`` or ``array[string]|null``."""
    branches = schema.get("anyOf") or schema.get("oneOf")
    if branches:
        return "|".join(sorted(schema_type(branch) for branch in branches))
    kind = str(schema.get("type", "any"))
    if kind == "array":
        return f"array[{schema_type(schema.get('items', {}))}]"
    return kind


def tool_parameters(input_schema: dict[str, Any]) -> list[Parameter]:
    """Return a tool's parameters in declaration order, with type, required flag and default."""
    required = set(input_schema.get("required", []))
    parameters = []
    for name, schema in input_schema.get("properties", {}).items():
        parameter: Parameter = {"name": name, "type": schema_type(schema), "required": name in required}
        if "default" in schema:
            parameter["default"] = schema["default"]
        parameters.append(parameter)
    return parameters


async def _list_tools(server: Any) -> list[Any]:
    """Every tool the server advertises over MCP, across pages."""
    tools: list[Any] = []
    cursor: str | None = None
    async with Client(server) as client:
        while True:
            page = await client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                return tools


def collect_contract() -> Contract:
    """Import this checkout's MCP server in isolation and return the contract of its registered tools."""
    with tempfile.TemporaryDirectory(prefix="kr-contract-", ignore_cleanup_errors=True) as base_dir:
        # config.py resolves its base directory and reads config.yaml when it is imported.
        os.environ.update(OFFLINE_ENV, KNOWLEDGE_RAG_DIR=base_dir)
        # Ahead of an installed copy, so a HEAD^1 worktree dumps its own code.
        sys.path.insert(0, str(REPO_ROOT))
        from mcp_server import server

        imported = Path(server.__file__).resolve()
        if not imported.is_relative_to(REPO_ROOT):
            raise RuntimeError(f"Imported {imported} instead of the mcp_server in {REPO_ROOT}")
        tools = asyncio.run(_list_tools(server.mcp))
    return {tool.name: tool_parameters(tool.input_schema) for tool in tools}


def _compare_parameter(label: str, base: Parameter, head: Parameter) -> list[str]:
    """Breaking changes between two versions of one parameter."""
    breaks = []
    if head["type"] != base["type"]:
        breaks.append(f"{label}: type changed: {base['type']} -> {head['type']}")
    if head["required"] and not base["required"]:
        breaks.append(f"{label}: became required")
    if "default" in base and "default" not in head:
        breaks.append(f"{label}: default removed (was {json.dumps(base['default'])})")
    elif "default" in base and head["default"] != base["default"]:
        breaks.append(f"{label}: default changed: {json.dumps(base['default'])} -> {json.dumps(head['default'])}")
    return breaks


def _compare_tool(tool: str, base: list[Parameter], head: list[Parameter]) -> tuple[list[str], list[str]]:
    """Breaking changes and additions between two versions of one tool's parameters."""
    base_by_name = {parameter["name"]: parameter for parameter in base}
    head_by_name = {parameter["name"]: parameter for parameter in head}
    breaks = [f"{tool}.{name}: parameter removed or renamed" for name in base_by_name if name not in head_by_name]
    additions = []
    for name, parameter in head_by_name.items():
        if name in base_by_name:
            breaks += _compare_parameter(f"{tool}.{name}", base_by_name[name], parameter)
        elif parameter["required"]:
            breaks.append(f"{tool}.{name}: new required parameter")
        else:
            additions.append(f"{tool}.{name}: new optional parameter")
    old_order = [name for name in base_by_name if name in head_by_name]
    new_order = [name for name in head_by_name if name in base_by_name]
    if old_order != new_order:
        breaks.append(f"{tool}: parameter order changed: {old_order} -> {new_order}")
    return breaks, additions


def compare_contracts(base: Contract, head: Contract) -> tuple[list[str], list[str]]:
    """Return the breaking changes and the additions from BASE's tool contract to HEAD's."""
    breaks = [f"tool removed: {tool}" for tool in base if tool not in head]
    additions = [f"tool added: {tool}" for tool in head if tool not in base]
    for tool in (name for name in base if name in head):
        tool_breaks, tool_additions = _compare_tool(tool, base[tool], head[tool])
        breaks += tool_breaks
        additions += tool_additions
    return breaks, additions


def _is_parameter_list(value: Any) -> bool:
    """True for a list of parameters shaped as ``tool_parameters`` writes them."""
    return isinstance(value, list) and all(isinstance(item, dict) and PARAMETER_KEYS <= item.keys() for item in value)


def load_contract(path: Path) -> Contract:
    """Read a file written by ``--dump``; raise ``ValueError`` when it is not one."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(_is_parameter_list(value) for value in data.values()):
        raise ValueError(f"{path} is not an MCP tool contract written by --dump")
    contract: Contract = data
    return contract


def _dump(path: Path) -> int:
    """Write this checkout's tool contract to PATH."""
    contract = collect_contract()
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] Contract of {len(contract)} MCP tools written to {path}")
    return EXIT_OK


def _compare(base_path: Path, head_path: Path) -> int:
    """Print what changed between two dumps and return the exit code."""
    try:
        base, head = load_contract(base_path), load_contract(head_path)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] Cannot read an MCP tool contract: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE
    breaks, additions = compare_contracts(base, head)
    for addition in additions:
        print(f"  + {addition}")
    if not breaks:
        print(f"[OK] No breaking change to the MCP tool contract ({len(additions)} addition(s)).")
        return EXIT_OK
    print("[FAIL] Breaking change(s) to the MCP tool contract:", file=sys.stderr)
    for item in breaks:
        print(f"  - {item}", file=sys.stderr)
    print(
        "\nEvery MCP client that calls these tools is affected. If the break is intentional, bump the MAJOR\n"
        f"version, add migration notes to CHANGELOG and ask a maintainer for the '{BYPASS_LABEL}' label.",
        file=sys.stderr,
    )
    return EXIT_BREAKING


def main(argv: list[str] | None = None) -> int:
    """Run ``--dump`` or ``--compare``; the module docstring lists the exit codes."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dump", type=Path, metavar="PATH", help="write this checkout's tool contract to PATH")
    mode.add_argument("--compare", type=Path, nargs=2, metavar=("BASE", "HEAD"), help="compare two dumps")
    args = parser.parse_args(argv)
    if args.dump:
        return _dump(args.dump)
    return _compare(*args.compare)


if __name__ == "__main__":
    sys.exit(main())
