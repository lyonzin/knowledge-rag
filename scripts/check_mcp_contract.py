"""Detect breaking changes to the MCP tool contract of knowledge-rag.

MCP clients call tools by name with JSON arguments, and Python code calls the
same functions positionally, so the contract is what the server advertises in
``tools/list``: the tool names, each tool's parameters in order, each
parameter's schema, which parameters are required and their default values.
Titles and descriptions are not part of it.

    python scripts/check_mcp_contract.py --dump PATH
        Import ``mcp_server`` from the checkout that contains this script, list
        its registered tools through the MCP SDK and write their contract to
        PATH. The import runs with a temporary KNOWLEDGE_RAG_DIR and with
        HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE set, so the user's config and
        index are never read and no model is downloaded.

    python scripts/check_mcp_contract.py --compare BASE HEAD
        Compare two dumps and exit:
            0  = HEAD keeps BASE's contract (new tools, and new optional
                 parameters after the existing ones, are listed)
            1  = breaking change: a tool removed, a parameter removed or
                 renamed, a parameter that is now required (new or existing),
                 any change to a parameter's schema, a changed or removed
                 default, or an existing parameter that moved
            2  = a file is missing, is not a dump or lists no tools; ``--dump``
                 also exits 2 when the server registers no tools

Schemas are compared as canonical JSON, with ``$ref`` resolved and ``title`` /
``description`` dropped, and defaults as JSON, so ``1`` differs from ``1.0``
and ``0`` from ``false``. The check fails closed: a widening such as ``int``
to ``Optional[int]`` breaks no caller but still needs the label.

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
PARAMETER_KEYS = frozenset({"name", "type", "required", "schema"})
# Keywords that document a schema without constraining the values it accepts.
ANNOTATION_KEYWORDS = frozenset({"title", "description"})
DEFS_PREFIX = "#/$defs/"
BYPASS_LABEL = "skip-api-surface"
EXIT_OK = 0
EXIT_BREAKING = 1
EXIT_INVALID = 2

Parameter = dict[str, Any]
Contract = dict[str, list[Parameter]]


def _json(value: Any) -> str:
    """Canonical JSON: equal values give equal text, while ``1``/``1.0`` and ``0``/``false`` differ."""
    return json.dumps(value, sort_keys=True)


def schema_type(schema: dict[str, Any]) -> str:
    """Render a canonical schema's type for messages, e.g. ``integer`` or ``array[string]|null``."""
    branches = schema.get("anyOf") or schema.get("oneOf")
    if branches:
        return "|".join(sorted(schema_type(branch) for branch in branches))
    kind = str(schema.get("type", "any"))
    if kind == "array":
        return f"array[{schema_type(schema.get('items', {}))}]"
    return kind


def _resolve(node: Any, defs: dict[str, Any], resolving: frozenset[str], names: bool = False) -> Any:
    """Copy ``node`` with each ``$ref`` replaced by its definition and the annotation keywords dropped.

    ``names`` marks a ``properties`` mapping, whose keys are field names (a field
    may be called ``title``), not keywords. ``resolving`` keeps a model that
    contains itself as a ``$ref`` instead of recursing forever.
    """
    if isinstance(node, list):
        return [_resolve(item, defs, resolving) for item in node]
    if not isinstance(node, dict):
        return node
    ref = None if names else node.get("$ref")
    if isinstance(ref, str) and ref.startswith(DEFS_PREFIX) and ref not in resolving:
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        return _resolve({**defs[ref.removeprefix(DEFS_PREFIX)], **siblings}, defs, resolving | {ref})
    return {
        key: _resolve(value, defs, resolving, names=not names and key == "properties")
        for key, value in node.items()
        if names or key not in ANNOTATION_KEYWORDS
    }


def canonical_schema(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """A parameter's schema as callers see it, without its default, which is compared apart."""
    canonical: dict[str, Any] = _resolve(schema, defs, frozenset())
    canonical.pop("default", None)
    return canonical


def tool_parameters(input_schema: dict[str, Any]) -> list[Parameter]:
    """Return a tool's parameters in declaration order: canonical schema, readable type, required flag, default."""
    defs = input_schema.get("$defs", {})
    required = set(input_schema.get("required", []))
    parameters = []
    for name, raw in input_schema.get("properties", {}).items():
        schema = canonical_schema(raw, defs)
        parameter: Parameter = {"name": name, "type": schema_type(schema), "required": name in required}
        parameter["schema"] = schema
        if "default" in raw:
            parameter["default"] = raw["default"]
        parameters.append(parameter)
    return parameters


async def read_contract(server: Any) -> Contract:
    """Return the contract of every tool ``server`` lists over MCP, across pages."""
    contract: Contract = {}
    cursor: str | None = None
    async with Client(server) as client:
        while True:
            page = await client.list_tools(cursor=cursor)
            contract.update((tool.name, tool_parameters(tool.input_schema)) for tool in page.tools)
            cursor = page.next_cursor
            if cursor is None:
                return contract


def _collect_contract() -> Contract:
    """Import this checkout's MCP server in isolation and read its contract.

    Meant for the ``--dump`` process: it sets ``os.environ`` and ``sys.path`` for
    the import and does not restore them.
    """
    with tempfile.TemporaryDirectory(prefix="kr-contract-", ignore_cleanup_errors=True) as base_dir:
        # config.py resolves its base directory and reads config.yaml when it is imported.
        os.environ.update(OFFLINE_ENV, KNOWLEDGE_RAG_DIR=base_dir)
        # Ahead of an installed copy, so a HEAD^1 worktree dumps its own code.
        sys.path.insert(0, str(REPO_ROOT))
        from mcp_server import server

        imported = Path(server.__file__).resolve()
        if not imported.is_relative_to(REPO_ROOT):
            raise RuntimeError(f"Imported {imported} instead of the mcp_server in {REPO_ROOT}")
        return asyncio.run(read_contract(server.mcp))


def _compare_parameter(label: str, base: Parameter, head: Parameter) -> list[str]:
    """Breaking changes between two versions of one parameter."""
    breaks = []
    if _json(head["schema"]) != _json(base["schema"]):
        breaks.append(
            f"{label}: schema changed: {base['type']} {_json(base['schema'])} -> {head['type']} {_json(head['schema'])}"
        )
    if head["required"] and not base["required"]:
        breaks.append(f"{label}: became required")
    if "default" in base and "default" not in head:
        breaks.append(f"{label}: default removed (was {_json(base['default'])})")
    elif "default" in base and _json(head["default"]) != _json(base["default"]):
        breaks.append(f"{label}: default changed: {_json(base['default'])} -> {_json(head['default'])}")
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
    # Python callers bind arguments by position: existing parameters stay first, in their order.
    kept = [name for name in base_by_name if name in head_by_name]
    if list(head_by_name)[: len(kept)] != kept:
        breaks.append(f"{tool}: parameter positions changed (new parameters go last): {kept} -> {list(head_by_name)}")
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
    """Read a file written by ``--dump``; raise ``ValueError`` when it is not one or lists no tools."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data or not all(_is_parameter_list(value) for value in data.values()):
        raise ValueError(f"{path} is not an MCP tool contract written by --dump, or it lists no tools")
    contract: Contract = data
    return contract


def _dump(path: Path) -> int:
    """Write this checkout's tool contract to PATH; an empty one is refused."""
    contract = _collect_contract()
    if not contract:
        print("[ERROR] The MCP server registers no tools, so there is no contract to write.", file=sys.stderr)
        return EXIT_INVALID
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] Contract of {len(contract)} MCP tools written to {path}")
    return EXIT_OK


def _compare(base_path: Path, head_path: Path) -> int:
    """Print what changed between two dumps and return the exit code."""
    try:
        base, head = load_contract(base_path), load_contract(head_path)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] Cannot read an MCP tool contract: {exc}", file=sys.stderr)
        return EXIT_INVALID
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
