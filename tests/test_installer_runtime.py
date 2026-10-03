"""Installer process selection and dry-run guarantees without network access."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import install


def test_python_detection_keeps_windows_paths_with_spaces(tmp_path, monkeypatch):
    """An executable path is one argv element, not a whitespace-split command."""
    local = str(tmp_path / "User Name")
    candidate = Path(f"{local}\\Programs\\Python\\Python312\\python.exe")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.touch()
    monkeypatch.setenv("LOCALAPPDATA", local)
    monkeypatch.setattr(install, "IS_WINDOWS", True)
    monkeypatch.setattr(install, "IS_MACOS", False)
    monkeypatch.setattr(install, "IS_LINUX", False)
    monkeypatch.setattr(install.shutil, "which", lambda value: None)
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        assert Path(command[0]) == candidate
        assert len(command) == 3
        return str(candidate).encode() if "sys.executable" in command[-1] else b"3.12\n"

    monkeypatch.setattr(install.subprocess, "check_output", probe)
    assert install.find_python() == candidate
    assert len(calls) == 2


@pytest.mark.parametrize("version", ["3.11", "3.12", "3.13"])
def test_python_detection_accepts_tested_versions(tmp_path, monkeypatch, version):
    candidate = tmp_path / "python"
    candidate.touch()
    monkeypatch.setattr(install, "IS_WINDOWS", False)
    monkeypatch.setattr(install, "IS_MACOS", False)
    monkeypatch.setattr(install, "IS_LINUX", False)
    monkeypatch.setattr(install.shutil, "which", lambda value: str(candidate))

    def probe(command, **kwargs):
        return str(candidate).encode() if "sys.executable" in command[-1] else version.encode()

    monkeypatch.setattr(install.subprocess, "check_output", probe)
    assert install.find_python() == candidate


def test_python_detection_does_not_claim_untested_314(tmp_path, monkeypatch):
    candidate = tmp_path / "python"
    candidate.touch()
    monkeypatch.setattr(install.shutil, "which", lambda value: str(candidate))
    monkeypatch.setattr(install.subprocess, "check_output", lambda *a, **kw: b"3.14")
    with pytest.raises(SystemExit) as caught:
        install.find_python()
    assert caught.value.code == 1


def test_selected_interpreter_creates_venv(tmp_path, monkeypatch):
    selected = tmp_path / "Python Selected" / "python.exe"
    calls = []
    monkeypatch.setattr(install.subprocess, "check_call", lambda argv, **kw: calls.append(argv))
    result = install.setup_venv(tmp_path, selected, False, False, None)
    assert calls[0] == [str(selected), "-m", "venv", str(tmp_path / "venv")]
    assert all(call[0] == str(result) for call in calls[1:])


def test_existing_venv_is_not_recreated(tmp_path, monkeypatch):
    executable = install.venv_python_path(tmp_path / "venv")
    executable.parent.mkdir(parents=True)
    executable.touch()
    calls = []
    monkeypatch.setattr(install.subprocess, "check_call", lambda argv, **kw: calls.append(argv))
    install.setup_venv(tmp_path, Path(sys.executable), False, False, None)
    assert not any(call[1:3] == ["-m", "venv"] for call in calls)


def test_source_install_registers_package_from_separate_checkout(tmp_path, monkeypatch):
    source = tmp_path / "source checkout"
    source.mkdir()
    (source / "pyproject.toml").touch()
    (source / "mcp_server").mkdir()
    (source / "mcp_server/server.py").touch()
    target = tmp_path / "installation"
    monkeypatch.chdir(source)
    calls = []
    monkeypatch.setattr(install.subprocess, "check_call", lambda argv, **kw: calls.append(argv))
    executable = install.setup_venv(target, Path(sys.executable), False, True, None)
    assert calls[-1] == [str(executable), "-m", "pip", "install", "--editable", str(source), "--quiet"]


def test_invalid_source_does_not_destroy_existing_venv(tmp_path, monkeypatch):
    target = tmp_path / "installation"
    executable = install.venv_python_path(target / "venv")
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"existing interpreter")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as caught:
        install.setup_venv(target, Path(sys.executable), True, True, None)
    assert caught.value.code == 1
    assert executable.read_bytes() == b"existing interpreter"


@pytest.mark.parametrize("source_flags", [[], ["--from-source"]])
def test_main_dry_run_creates_no_directories_or_files(tmp_path, monkeypatch, source_flags):
    target = tmp_path / "installation not created"
    config = tmp_path / "client not created" / "settings.json"
    client = install.Client("test", "Test client", lambda: config, "mcpServers", install._spec_mcp_servers)
    monkeypatch.setattr(install, "CLIENTS", [client])
    monkeypatch.setattr(install, "find_python", lambda: Path(sys.executable))
    monkeypatch.setattr(install, "python_version_short", lambda exe: "3.12")

    def unexpected(*args, **kwargs):
        pytest.fail("dry-run must not execute installation subprocesses")

    monkeypatch.setattr(install.subprocess, "check_call", unexpected)
    assert install.main(["--dry-run", "--for", "test", "--install-path", str(target), *source_flags]) == 0
    assert list(tmp_path.iterdir()) == []
