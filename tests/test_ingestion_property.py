"""Property-based fuzz tests for the document parsers.

Innovation layer: Hypothesis generates thousands of inputs the team
would never write by hand and runs the parser against each. The
contract we want is simple: parsing must not crash on any reasonable
input the user might throw at us.

Each property test focuses on one parser and one shape of evil input.
The asserts are deliberately weak (no crash, chunks are strings,
chunks list is bounded) — Hypothesis searches for inputs that violate
exactly that, so we catch *catastrophic* failures, not subtle ones.

Subtle correctness lives in unit tests; property tests guard the floor.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("hypothesis", reason="hypothesis required for property-based fuzz tests")

from hypothesis import HealthCheck, given, settings  # noqa: E402  — imported after the importorskip guard
from hypothesis import strategies as st  # noqa: E402  — imported after the importorskip guard

from mcp_server.ingestion import DocumentParser  # noqa: E402  — imported after the importorskip guard

# Shared settings: keep CI fast (50 examples per property), suppress the
# function-scoped fixture warning (we use tmp_path which is fine here).
HSETTINGS = settings(
    max_examples=50,
    deadline=2_000,  # ms per example
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# ---------------------------------------------------------------------------
# Markdown parser — random heading depths, nested code, special chars
# ---------------------------------------------------------------------------

# Grammar of "things that look like markdown to a robot":
markdown_lines = st.one_of(
    st.text(alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"), max_size=80),
    st.builds(lambda d, t: f"{'#' * d} {t}", st.integers(min_value=1, max_value=6), st.text(max_size=40)),
    st.builds(lambda t: f"```\n{t}\n```", st.text(max_size=200)),
    st.builds(lambda t: f"[link]({t})", st.text(min_size=1, max_size=40)),
    st.just(""),
    st.just("---"),
    st.just("[" * 50),  # the literal nightmare: many open brackets
)
random_markdown = st.lists(markdown_lines, max_size=30).map("\n".join)


@given(content=random_markdown)
@HSETTINGS
def test_markdown_parser_never_crashes(content, tmp_path):
    """No markdown input must raise unhandled exceptions."""
    path = tmp_path / "fuzz.md"
    path.write_text(content, encoding="utf-8")

    parser = DocumentParser()
    try:
        doc = parser.parse_file(path)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Parser crashed on input:\n---\n{content[:200]}...\n---\n{type(exc).__name__}: {exc}")

    if doc is None:
        return  # parser legitimately rejected empty/whitespace content

    # Every chunk must be a string (no None, no bytes leaking through)
    for chunk in doc.chunks:
        assert isinstance(chunk.content, str), f"Chunk content is not a string: {type(chunk.content)}"


# ---------------------------------------------------------------------------
# JSON parser — random nesting depth and key/value types
# ---------------------------------------------------------------------------


# Hypothesis recursive strategy for arbitrary JSON-compatible structures
@st.composite
def json_value(draw, depth=0):
    if depth > 4:
        return draw(st.one_of(st.text(max_size=20), st.integers(), st.floats(allow_nan=False, allow_infinity=False)))
    return draw(
        st.one_of(
            st.text(max_size=20),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.none(),
            st.lists(json_value(depth=depth + 1), max_size=5),
            st.dictionaries(st.text(min_size=1, max_size=10), json_value(depth=depth + 1), max_size=5),
        )
    )


@given(payload=st.dictionaries(st.text(min_size=1, max_size=10), json_value(), min_size=1, max_size=8))
@HSETTINGS
def test_json_parser_never_crashes(payload, tmp_path):
    """Arbitrary nested JSON must parse without exceptions."""
    path = tmp_path / "fuzz.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    parser = DocumentParser()
    try:
        doc = parser.parse_file(path)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"JSON parser crashed on payload of {len(str(payload))} chars: {exc}")

    if doc is None:
        return
    assert all(isinstance(c.content, str) for c in doc.chunks)


# ---------------------------------------------------------------------------
# CSV parser — embedded tabs, nulls, weird quoting
# ---------------------------------------------------------------------------

csv_field = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs",),
        blacklist_characters="\x00",  # CSV can break on raw NUL
    ),
    max_size=20,
)


@given(rows=st.lists(st.lists(csv_field, min_size=1, max_size=4), min_size=1, max_size=20))
@HSETTINGS
def test_csv_parser_never_crashes(rows, tmp_path):
    """Bizarre CSV content (tabs, special chars) must not crash the parser."""
    path = tmp_path / "fuzz.csv"
    text = "\n".join(",".join(field.replace(",", " ") for field in row) for row in rows)
    path.write_text(text, encoding="utf-8")

    parser = DocumentParser()
    try:
        doc = parser.parse_file(path)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"CSV parser crashed: {exc}")

    if doc is None:
        return
    assert all(isinstance(c.content, str) for c in doc.chunks)


# ---------------------------------------------------------------------------
# Text parser — large random blobs
# ---------------------------------------------------------------------------


@given(content=st.text(max_size=5000))
@HSETTINGS
def test_text_parser_never_crashes(content, tmp_path):
    """Arbitrary text up to 5KB must parse without exceptions."""
    path = tmp_path / "fuzz.txt"
    path.write_text(content, encoding="utf-8")

    parser = DocumentParser()
    try:
        doc = parser.parse_file(path)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Text parser crashed on {len(content)}-char input: {exc}")

    if doc is None:
        return
    # Sum of chunk lengths must be <= input length + reasonable overhead
    total = sum(len(c.content) for c in doc.chunks)
    assert total <= len(content) * 3, f"Chunks expanded content suspiciously: {len(content)} -> {total}"
