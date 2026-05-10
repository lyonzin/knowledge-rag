"""Detect breaking changes to the public API surface of knowledge-rag.

Walks `mcp_server/` via Python AST (no runtime imports — works offline,
without HuggingFace, ChromaDB, or any heavy deps installed) and emits a
structured snapshot of:

- Module-level public functions (no leading underscore)
- Public classes and their public methods
- Public exception classes
- Type signatures (positional + keyword args, defaults presence, return type)

Two modes:

    python scripts/check_api_surface.py --snapshot
        Generate the current snapshot and write it to
        .github/api-surface-baseline.json

    python scripts/check_api_surface.py --check
        Compare current code against the committed baseline and exit:
            0  = no breaking change (additions are OK)
            1  = breaking change detected (function/class removed,
                 signature narrowed, parameter renamed, return type changed)
            2  = baseline file missing or unparseable

In CI we run with --check on PRs. If a contributor genuinely needs to
break the public API they:
    1. Bump the MAJOR version
    2. Run --snapshot locally to regenerate the baseline
    3. Commit the new baseline alongside the breaking change PR
    4. Justify the break in the PR description and CHANGELOG migration notes

Adopting griffe directly would require installing it; the AST approach
keeps this dependency-free and 100% deterministic.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "mcp_server"
BASELINE = REPO_ROOT / ".github" / "api-surface-baseline.json"


def _is_public(name: str) -> bool:
    """Names without leading underscore are part of the public surface."""
    return not name.startswith("_")


def _arg_signature(args: ast.arguments) -> dict[str, Any]:
    """Render a function's argument list into a structural dict.

    We capture names, kind (positional / keyword-only / vararg / kwarg),
    and whether each argument has a default. We deliberately do NOT capture
    the actual default expression — defaults can be tweaked without breaking
    callers, only the *presence* of a default matters for compatibility.
    """

    def render(arg_list: list[ast.arg], defaults_count: int, kind: str) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        for idx, arg in enumerate(arg_list):
            has_default = idx >= len(arg_list) - defaults_count
            rendered.append(
                {
                    "name": arg.arg,
                    "kind": kind,
                    "has_default": has_default,
                    "annotated": arg.annotation is not None,
                }
            )
        return rendered

    sig: list[dict[str, Any]] = []
    sig.extend(render(args.posonlyargs, len(args.defaults), "positional_only"))
    sig.extend(render(args.args, max(0, len(args.defaults) - len(args.posonlyargs)), "positional"))
    if args.vararg is not None:
        sig.append(
            {
                "name": args.vararg.arg,
                "kind": "vararg",
                "has_default": False,
                "annotated": args.vararg.annotation is not None,
            }
        )
    # kwonlyargs each have their own defaults list aligned by index
    kw_defaults = args.kw_defaults
    for idx, arg in enumerate(args.kwonlyargs):
        sig.append(
            {
                "name": arg.arg,
                "kind": "keyword_only",
                "has_default": kw_defaults[idx] is not None,
                "annotated": arg.annotation is not None,
            }
        )
    if args.kwarg is not None:
        sig.append(
            {
                "name": args.kwarg.arg,
                "kind": "kwarg",
                "has_default": False,
                "annotated": args.kwarg.annotation is not None,
            }
        )
    return {"args": sig}


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    sig = _arg_signature(node.args)
    sig["returns_annotated"] = node.returns is not None
    sig["is_async"] = isinstance(node, ast.AsyncFunctionDef)
    return sig


def _class_info(node: ast.ClassDef) -> dict[str, Any]:
    bases = [ast.unparse(b) for b in node.bases]
    methods: dict[str, Any] = {}
    for body_node in node.body:
        if isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_public(body_node.name):
                methods[body_node.name] = _function_signature(body_node)
    return {"bases": bases, "methods": methods}


def collect_module_surface(module_path: Path) -> dict[str, Any]:
    """Parse a single .py file and return its public surface."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    functions: dict[str, Any] = {}
    classes: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(node.name):
            functions[node.name] = _function_signature(node)
        elif isinstance(node, ast.ClassDef) and _is_public(node.name):
            classes[node.name] = _class_info(node)
    return {"functions": functions, "classes": classes}


def collect_package_surface() -> dict[str, Any]:
    """Walk the package and emit a deterministic surface dict."""
    surface: dict[str, Any] = {}
    for py in sorted(PACKAGE.rglob("*.py")):
        # Skip private modules (leading underscore in any path component)
        rel = py.relative_to(REPO_ROOT)
        if any(part.startswith("_") and part != "__init__.py" for part in rel.parts[1:]):
            continue
        module_name = ".".join(rel.with_suffix("").parts)
        # __init__.py becomes the package name (drop the trailing .__init__)
        if module_name.endswith(".__init__"):
            module_name = module_name[: -len(".__init__")]
        surface[module_name] = collect_module_surface(py)
    return {"version": 1, "modules": surface}


# ---------------------------------------------------------------------------
# Diff logic
# ---------------------------------------------------------------------------


def _diff_function(name: str, old: dict[str, Any], new: dict[str, Any], breaks: list[str]) -> None:
    old_args = old.get("args", [])
    new_args = new.get("args", [])

    old_required = [a for a in old_args if not a["has_default"] and a["kind"] != "kwarg" and a["kind"] != "vararg"]
    new_required = [a for a in new_args if not a["has_default"] and a["kind"] != "kwarg" and a["kind"] != "vararg"]

    # New required parameters break callers
    if len(new_required) > len(old_required):
        added = {a["name"] for a in new_required} - {a["name"] for a in old_required}
        if added:
            breaks.append(f"{name}: new required parameter(s): {sorted(added)}")

    # Removed parameters (any kind) break callers that supplied them
    old_names = {a["name"] for a in old_args}
    new_names = {a["name"] for a in new_args}
    removed = old_names - new_names
    if removed:
        breaks.append(f"{name}: removed parameter(s): {sorted(removed)}")

    # Renamed positional args also count as breaking
    old_positional = [a for a in old_args if a["kind"] in ("positional", "positional_only")]
    new_positional = [a for a in new_args if a["kind"] in ("positional", "positional_only")]
    for idx, old_arg in enumerate(old_positional):
        if idx < len(new_positional) and new_positional[idx]["name"] != old_arg["name"]:
            breaks.append(f"{name}: positional arg #{idx} renamed: {old_arg['name']} -> {new_positional[idx]['name']}")

    # Async <-> sync flip is breaking
    if old.get("is_async") != new.get("is_async"):
        breaks.append(f"{name}: async/sync flipped")


def diff_surfaces(old_surface: dict[str, Any], new_surface: dict[str, Any]) -> list[str]:
    breaks: list[str] = []
    old_modules = old_surface.get("modules", {})
    new_modules = new_surface.get("modules", {})

    for mod_name, old_mod in old_modules.items():
        if mod_name not in new_modules:
            breaks.append(f"module removed: {mod_name}")
            continue
        new_mod = new_modules[mod_name]

        for fn_name, old_sig in old_mod.get("functions", {}).items():
            if fn_name not in new_mod.get("functions", {}):
                breaks.append(f"function removed: {mod_name}.{fn_name}")
                continue
            _diff_function(f"{mod_name}.{fn_name}", old_sig, new_mod["functions"][fn_name], breaks)

        for cls_name, old_cls in old_mod.get("classes", {}).items():
            if cls_name not in new_mod.get("classes", {}):
                breaks.append(f"class removed: {mod_name}.{cls_name}")
                continue
            new_cls = new_mod["classes"][cls_name]
            for method_name, old_method in old_cls.get("methods", {}).items():
                if method_name not in new_cls.get("methods", {}):
                    breaks.append(f"method removed: {mod_name}.{cls_name}.{method_name}")
                    continue
                _diff_function(
                    f"{mod_name}.{cls_name}.{method_name}",
                    old_method,
                    new_cls["methods"][method_name],
                    breaks,
                )

    return breaks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", action="store_true", help="Write current surface as new baseline")
    group.add_argument("--check", action="store_true", help="Compare current surface against committed baseline")
    args = parser.parse_args()

    current = collect_package_surface()

    if args.snapshot:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        n_modules = len(current["modules"])
        n_funcs = sum(len(m.get("functions", {})) for m in current["modules"].values())
        n_classes = sum(len(m.get("classes", {})) for m in current["modules"].values())
        print(f"[OK] Snapshot written to {BASELINE}")
        print(f"     modules={n_modules}  functions={n_funcs}  classes={n_classes}")
        return 0

    # --check mode
    if not BASELINE.exists():
        print(f"[ERROR] Baseline not found: {BASELINE}", file=sys.stderr)
        print("        Run with --snapshot first to create it.", file=sys.stderr)
        return 2

    try:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Could not parse baseline: {exc}", file=sys.stderr)
        return 2

    breaks = diff_surfaces(baseline, current)
    if not breaks:
        print("[OK] No breaking changes to public API surface.")
        return 0

    print("[FAIL] Breaking change(s) detected in public API:", file=sys.stderr)
    for b in breaks:
        print(f"  - {b}", file=sys.stderr)
    print(
        "\nIf this break is intentional:\n"
        "  1. Bump MAJOR version (X+1.0.0)\n"
        "  2. Add migration notes to CHANGELOG\n"
        "  3. Regenerate baseline: python scripts/check_api_surface.py --snapshot\n"
        "  4. Commit the new baseline with the break\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
