"""Compatibility gate regressions using real parsed Python signatures."""

from __future__ import annotations

import ast

import pytest

from scripts.check_api_surface import _arg_signature, _diff_function, _function_signature


def signature(arguments: str) -> dict:
    node = ast.parse(f"def example({arguments}): pass").body[0]
    assert isinstance(node, ast.FunctionDef)
    return _function_signature(node)


def changes(old: str, new: str) -> list[str]:
    result = []
    _diff_function("example", signature(old), signature(new), result)
    return result


@pytest.mark.parametrize(
    ("arguments", "defaults"),
    [
        ("a, /, b=1", [False, True]),
        ("a, b=1, /, c=2", [False, True, True]),
        ("a=1, /, b=2", [True, True]),
        ("a, /, b, c=1", [False, False, True]),
        ("a=1, /", [True]),
        ("a, /, *, b=1, c", [False, True, False]),
    ],
)
def test_defaults_align_with_all_positional_parameters(arguments, defaults):
    node = ast.parse(f"def example({arguments}): pass").body[0]
    assert isinstance(node, ast.FunctionDef)
    assert [argument["has_default"] for argument in _arg_signature(node.args)["args"]] == defaults


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("a=1", "a"),
        ("*, a=1", "*, a"),
        ("a=1, /", "a, /"),
        ("a", "a, /"),
        ("a", "*, a"),
        ("a, /", "*, a"),
        ("*, a", "a, /"),
        ("*values", "values=()"),
        ("**options", "options=None"),
        ("a, *values", "a, b=None, *values"),
        ("a=1, b=2", "b=2, a=1"),
        ("a=1", "a=1, *, required"),
    ],
)
def test_gate_rejects_narrowed_call_contract(old, new):
    assert changes(old, new), (old, new)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("a", "a=1"),
        ("a", "a, b=None"),
        ("a, /", "a"),
        ("*, a", "a"),
        ("a, /, b=1", "a, /, b=1, *, c=None"),
        ("a", "a, *values, **options"),
    ],
)
def test_gate_allows_compatible_widening(old, new):
    assert changes(old, new) == []


def test_same_parameter_name_losing_default_is_reported():
    assert "parameter became required: count" in changes("count=5", "count")[0]
