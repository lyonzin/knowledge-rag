"""Compatibility gate regressions using real parsed Python signatures."""

from __future__ import annotations

import ast

import pytest

from scripts.check_api_surface import _arg_signature, _class_info, _diff_function, _function_signature, diff_surfaces


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


def constructor_changes(old: str, new: str) -> list[str]:
    """Breaking changes between the constructors of two versions of class ``C``."""

    def constructor(source: str) -> dict:
        node = ast.parse(source).body[0]
        assert isinstance(node, ast.ClassDef)
        return _class_info(node)["methods"]["__init__"]

    result = []
    _diff_function("C.__init__", constructor(old), constructor(new), result)
    return result


def dataclass_source(*fields: str, decorator: str = "@dataclass") -> str:
    """Source of a dataclass ``C`` with one field per argument."""
    return f"{decorator}\nclass C:\n" + "".join(f"    {line}\n" for line in fields)


@pytest.mark.parametrize(
    ("fields", "breaks"),
    [
        pytest.param(["a: int"], True, id="field-removed"),
        pytest.param(["a: int", "b: int"], True, id="default-removed"),
        pytest.param(["a: int", "z: int = 0", "b: int = 1"], True, id="inserted-before-existing"),
        pytest.param(
            ["a: int", "b: int = 1", "c: list = field(kw_only=True)"], True, id="appended-keyword-only-required"
        ),
        pytest.param(["a: int", "b: int = 1", "c: list = field(default_factory=list)"], False, id="appended-default"),
    ],
)
def test_dataclass_fields_follow_the_constructor_rules(fields, breaks):
    old = dataclass_source("a: int", "b: int = 1")
    assert bool(constructor_changes(old, dataclass_source(*fields))) is breaks


def test_constructor_that_now_requires_an_argument_is_reported():
    old = "class C:\n    def __init__(self): pass\n"
    new = "class C:\n    def __init__(self, store): pass\n"
    assert constructor_changes(old, new) == ["C.__init__: new required parameter(s): ['store']"]


def generated_init(source: str) -> list[tuple[str, str, bool]] | None:
    """``(name, kind, has_default)`` after ``self`` in the ``__init__`` recorded for ``C``; None if there is none."""
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.ClassDef)
    init = _class_info(node)["methods"].get("__init__")
    return None if init is None else [(a["name"], a["kind"], a["has_default"]) for a in init["args"][1:]]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(dataclass_source("a: int", decorator="@dataclass(init=False)"), None, id="decorator-init-false"),
        pytest.param(
            dataclass_source("a: int", "b: int = 1", decorator="@dataclass(kw_only=True)"),
            [("a", "keyword_only", False), ("b", "keyword_only", True)],
            id="decorator-kw-only",
        ),
        pytest.param(
            dataclass_source("a: int", "b: int = field(default=1, init=False)"),
            [("a", "positional", False)],
            id="field-init-false",
        ),
        pytest.param(
            dataclass_source("a: int = 0", "b: list = field(kw_only=True)"),
            [("a", "positional", True), ("b", "keyword_only", False)],
            id="field-kw-only-required",
        ),
        pytest.param(
            dataclass_source("a: int = field(kw_only=True)", "b: int"),
            [("b", "positional", False), ("a", "keyword_only", False)],
            id="keyword-only-fields-go-last",
        ),
        pytest.param(
            dataclass_source("a: int", "_: KW_ONLY", "b: int"),
            [("a", "positional", False), ("b", "keyword_only", False)],
            id="kw-only-sentinel",
        ),
        pytest.param(
            dataclass_source("a: int", "LIMIT: ClassVar[int] = 3"), [("a", "positional", False)], id="classvar"
        ),
    ],
)
def test_generated_constructor_follows_dataclass_semantics(source, expected):
    assert generated_init(source) == expected


def surface_of(source: str) -> dict:
    """A one-module surface holding the classes defined in ``source``."""
    classes = {node.name: _class_info(node) for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}
    return {"modules": {"m": {"functions": {}, "classes": classes}}}


@pytest.mark.parametrize(
    ("init", "expected"),
    [
        pytest.param(
            "def __init__(self, model): pass",
            ["m.EmbeddingError.__init__: new constructor requires parameter(s): ['model']"],
            id="requires-an-argument",
        ),
        pytest.param("def __init__(self, model=None): pass", [], id="optional-arguments-only"),
    ],
)
def test_new_constructor_over_an_inherited_one_must_not_require_arguments(init, expected):
    old = "class EmbeddingError(RuntimeError):\n    pass\n"
    new = f"class EmbeddingError(RuntimeError):\n    {init}\n"
    assert diff_surfaces(surface_of(old), surface_of(new)) == expected
