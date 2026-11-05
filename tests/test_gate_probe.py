"""Temporary probe for #269, removed in the next commit.

It fails in a single matrix cell (macOS, Python 3.13) to show that one failing
cell is enough to turn Quality Gate Summary red.
"""

import sys

import pytest

IS_PROBE_CELL = sys.platform == "darwin" and sys.version_info[:2] == (3, 13)


@pytest.mark.skipif(not IS_PROBE_CELL, reason="the #269 probe fails in one matrix cell only")
def test_one_failing_matrix_cell_blocks_the_merge():
    raise AssertionError("deliberate failure for #269; this file is removed in the next commit")
