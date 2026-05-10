"""Verifies the pytest atexit cleanup_numbered_dir Windows-flake patch.

The patch lives in ``tests/conftest.py``; this test guards against accidental
removal of the wrapper or upstream changes to the function signature that
would silently break the workaround.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import _pytest.pathlib as _pp
import conftest


def test_cleanup_numbered_dir_is_wrapped():
    """conftest.py must replace _pytest.pathlib.cleanup_numbered_dir with a safe wrapper."""
    assert _pp.cleanup_numbered_dir.__name__ == "_safe_cleanup_numbered_dir", (
        "tests/conftest.py is no longer wrapping pytest's cleanup_numbered_dir; "
        "the Windows atexit OSError flake will resurface."
    )


def test_safe_wrapper_swallows_oserror():
    """The wrapper must NOT propagate OSError raised by the real cleanup."""

    def boom(*args, **kwargs):
        raise OSError("simulated tmp_path cleanup race")

    with patch.object(conftest, "_ORIGINAL_CLEANUP_NUMBERED_DIR", boom):
        # Calling the wrapper must not raise
        result = _pp.cleanup_numbered_dir(root=None, prefix="x", keep=0, consider_lock_dead_if_created_before=0.0)
        assert result is None


def test_safe_wrapper_propagates_non_oserror():
    """Non-OSError exceptions must still propagate so real bugs are not hidden."""

    def boom(*args, **kwargs):
        raise ValueError("real bug")

    with patch.object(conftest, "_ORIGINAL_CLEANUP_NUMBERED_DIR", boom):
        try:
            _pp.cleanup_numbered_dir(root=None, prefix="x", keep=0, consider_lock_dead_if_created_before=0.0)
        except ValueError as e:
            assert "real bug" in str(e)
        else:
            raise AssertionError("Wrapper swallowed a non-OSError exception")


def test_real_cleanup_signature_unchanged():
    """If pytest changes the signature, the wrapper would silently break.

    This guard fires on pytest upgrades that alter the function so we know to
    revisit the patch (or, ideally, remove it because pytest fixed the bug).
    """
    real = conftest._ORIGINAL_CLEANUP_NUMBERED_DIR
    assert real is not None, "conftest failed to capture the real cleanup_numbered_dir"
    params = list(inspect.signature(real).parameters.keys())
    expected = ["root", "prefix", "keep", "consider_lock_dead_if_created_before"]
    assert params == expected, (
        f"pytest cleanup_numbered_dir signature changed: got {params}, expected {expected}. "
        "Revisit the conftest.py wrapper."
    )
