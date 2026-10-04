"""Portable path-adapter contracts; containment remains the security boundary."""

from pathlib import PurePosixPath

import pytest

from mcp_server import filesystem
from mcp_server.security import PathEscapeError, validate_path_within


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"C:\documents\nested\source.md", r"\\?\C:\documents\nested\source.md"),
        (r"\\server\share\nested\source.md", r"\\?\UNC\server\share\nested\source.md"),
        (r"C:\documents\unused\..\source.md", r"\\?\C:\documents\source.md"),
        (r"C:/documents/./source.md", r"\\?\C:\documents\source.md"),
        (r"\\?\C:\documents\source.md", r"\\?\C:\documents\source.md"),
        (r"\\?\UNC\server\share\source.md", r"\\?\UNC\server\share\source.md"),
    ],
)
def test_win32_namespace_keeps_drive_unc_and_normalized_target(source, expected):
    assert filesystem._extend_windows_path(source) == expected


@pytest.mark.parametrize("source", [r"C:\documents\source.md", r"\\server\share\source.md"])
def test_win32_namespace_roundtrip_preserves_public_path(source):
    assert filesystem._public_windows_path(filesystem._extend_windows_path(source)) == source


def test_posix_paths_are_returned_unchanged(monkeypatch):
    monkeypatch.setattr(filesystem.sys, "platform", "linux")
    source = PurePosixPath("/documents") / ("x" * 240) / "source.md"
    assert filesystem.io_path(source) is source


def test_short_local_paths_are_returned_unchanged(tmp_path):
    source = tmp_path / "source.md"
    assert filesystem.io_path(source) is source


def test_containment_still_rejects_parent_traversal_and_nul(tmp_path):
    for source in ("../outside.md", "source.md\x00hidden"):
        with pytest.raises(PathEscapeError):
            validate_path_within(tmp_path, source)


def test_normal_local_resolution_keeps_canonical_identity(tmp_path):
    source = tmp_path / "unused" / ".." / "source.md"
    assert filesystem.resolve_path(source) == source.resolve()
