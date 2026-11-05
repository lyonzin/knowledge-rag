"""Test-count gate: count the selected tests, not the deselected ones."""

from __future__ import annotations

import pytest

from scripts.check_test_count import parse_collected_count


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        pytest.param(
            "tests/test_a.py::test_one\ntests/test_a.py::test_two\n\n"
            "1312/1317 tests collected (5 deselected) in 3.21s\n",
            1312,
            id="deselected-tests-are-not-counted",
        ),
        pytest.param("144 tests collected in 1.23s", 144, id="plain-summary"),
        pytest.param("1 test collected", 1, id="singular-summary"),
        pytest.param("tests/test_a.py::test_one\n", None, id="no-summary-line"),
    ],
)
def test_parse_collected_count(output: str, expected: int | None) -> None:
    assert parse_collected_count(output) == expected
