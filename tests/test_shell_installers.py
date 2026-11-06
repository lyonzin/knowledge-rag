"""Run installer shells with local fixtures; no downloads or global installs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mcp_server import skills_cli

ROOT = Path(__file__).resolve().parents[1]
SKILLS_SH = ROOT / "skills" / "install.sh"
SKILLS_PS1 = ROOT / "skills" / "install.ps1"
POWERSHELL_FLAGS = ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
# Arguments given to a wrapper, and the ones it must pass to knowledge-rag-skills.
FORWARDING = [
    pytest.param(
        ["--target", "dir with spaces", "--yes"],
        ["install", "--target", "dir with spaces", "--yes"],
        id="install-by-default",
    ),
    pytest.param(["uninstall", "--dry-run"], ["uninstall", "--dry-run"], id="subcommand-kept"),
]
# Commands on PATH, then the exit code and advice a wrapper must give without knowledge-rag-skills.
UNUSABLE = [
    pytest.param((), 127, "pip install knowledge-rag", id="not-installed"),
    pytest.param(("knowledge-rag",), 1, "pip install -U knowledge-rag", id="release-5.0.0-or-older"),
]


def _shells(*names: str) -> list:
    """Shells on PATH, as test parameters. C:/Windows/System32/bash.exe is left out: it runs scripts inside WSL."""
    found = [path for path in map(shutil.which, names) if path and Path(path).parent.name.lower() != "system32"]
    skipped = pytest.param(None, marks=pytest.mark.skip(reason=f"none of {names} is installed"), id="missing")
    return [pytest.param(path, id=Path(path).stem.lower()) for path in found] or [skipped]


def _path_with(folder: Path, *names: str) -> Path:
    """A PATH folder whose stand-ins for ``names`` print each argument on its own line.

    Each gets a script, for sh and pwsh, and a .cmd, which Windows PowerShell needs.
    """
    folder.mkdir()
    printer = folder / "printer.py"
    printer.write_text("import sys\n\nprint('\\n'.join(sys.argv[1:]))\n", encoding="utf-8")
    for name in names:
        script = folder / name
        script.write_text(
            '#!/bin/sh\nfor arg in "$@"; do printf \'%s\\n\' "$arg"; done\n', encoding="utf-8", newline="\n"
        )
        script.chmod(0o755)
        (folder / f"{name}.cmd").write_text(f'@"{sys.executable}" "{printer}" %*\n', encoding="utf-8")
    return folder


def _run_with_path(command: list, path: Path) -> subprocess.CompletedProcess:
    """Run with PATH set to ``path`` alone, so a real knowledge-rag can never answer."""
    return subprocess.run(
        command,
        env=dict(os.environ, PATH=str(path), HOME=str(path), USERPROFILE=str(path)),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("shell", _shells("sh", "bash"))
@pytest.mark.parametrize(("arguments", "expected"), FORWARDING)
def test_sh_wrapper_forwards_arguments_to_knowledge_rag_skills(tmp_path, shell, arguments, expected):
    path = _path_with(tmp_path / "bin", "knowledge-rag-skills", "knowledge-rag")
    result = _run_with_path([shell, str(SKILLS_SH), *arguments], path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == expected


@pytest.mark.parametrize("shell", _shells("sh", "bash"))
def test_sh_wrapper_prefixes_install_with_no_arguments(tmp_path, shell):
    path = _path_with(tmp_path / "bin", "knowledge-rag-skills", "knowledge-rag")
    result = _run_with_path([shell, str(SKILLS_SH)], path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["install"]


@pytest.mark.parametrize("shell", _shells("sh", "bash"))
@pytest.mark.parametrize(("commands", "code", "advice"), UNUSABLE)
def test_sh_wrapper_explains_what_to_install_and_never_runs_an_old_knowledge_rag(
    tmp_path, shell, commands, code, advice
):
    result = _run_with_path([shell, str(SKILLS_SH), "status"], _path_with(tmp_path / "bin", *commands))
    assert (result.returncode, result.stdout) == (code, "")
    assert advice in result.stderr


@pytest.mark.parametrize("shell", _shells("powershell", "pwsh"))
@pytest.mark.parametrize(("arguments", "expected"), FORWARDING)
def test_powershell_wrapper_forwards_arguments_to_knowledge_rag_skills(tmp_path, shell, arguments, expected):
    path = _path_with(tmp_path / "bin", "knowledge-rag-skills", "knowledge-rag")
    result = _run_with_path([shell, *POWERSHELL_FLAGS, str(SKILLS_PS1), *arguments], path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == expected


@pytest.mark.parametrize("shell", _shells("powershell", "pwsh"))
def test_powershell_wrapper_prefixes_install_with_no_arguments(tmp_path, shell):
    path = _path_with(tmp_path / "bin", "knowledge-rag-skills", "knowledge-rag")
    result = _run_with_path([shell, *POWERSHELL_FLAGS, str(SKILLS_PS1)], path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["install"]


@pytest.mark.parametrize("shell", _shells("powershell", "pwsh"))
@pytest.mark.parametrize(("commands", "code", "advice"), UNUSABLE)
def test_powershell_wrapper_explains_what_to_install_and_never_runs_an_old_knowledge_rag(
    tmp_path, shell, commands, code, advice
):
    command = [shell, *POWERSHELL_FLAGS, str(SKILLS_PS1), "status"]
    result = _run_with_path(command, _path_with(tmp_path / "bin", *commands))
    assert (result.returncode, result.stdout) == (code, "")
    assert advice in result.stderr


@pytest.mark.parametrize("shell", _shells("sh", "bash", "powershell", "pwsh"))
@pytest.mark.parametrize("code", [7, 130])
def test_wrappers_propagate_the_installer_exit_code(tmp_path, shell, code):
    path = _path_with(tmp_path / "bin", "knowledge-rag-skills")
    script = path / "knowledge-rag-skills"
    script.write_text(f'#!/bin/sh\nexit "{code}"\n', encoding="utf-8", newline="\n")
    script.chmod(0o755)
    (path / "knowledge-rag-skills.cmd").write_text(f"@exit /b {code}\r\n", encoding="utf-8")
    if Path(shell).stem in ("powershell", "pwsh"):
        result = _run_with_path([shell, *POWERSHELL_FLAGS, str(SKILLS_PS1), "status"], path)
    else:
        result = _run_with_path([shell, str(SKILLS_SH), "status"], path)
    assert result.returncode == code, result.stdout + result.stderr


def test_powershell_wrapper_is_ascii_for_windows_powershell_5():
    """Windows PowerShell 5.1 reads a script without a BOM in the ANSI code page."""
    assert SKILLS_PS1.read_bytes().isascii()


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell 5.1 argument passing")
def test_powershell_5_forwards_an_argument_ending_in_a_backslash(tmp_path):
    shell = shutil.which("powershell")
    if shell is None:
        pytest.skip("Windows PowerShell is unavailable")
    target = str(tmp_path / "space name") + "\\"
    path = _path_with(tmp_path / "bin", "knowledge-rag-skills")
    result = _run_with_path([shell, *POWERSHELL_FLAGS, str(SKILLS_PS1), "status", "--target", target], path)
    assert result.returncode == 0, result.stderr
    received = result.stdout.splitlines()[-1]
    # 5.1 doubling may deliver a second backslash; both normalize to the same folder.
    assert os.path.normpath(received) == os.path.normpath(target)


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is unavailable")
def test_powershell_wrapper_runs_piped_through_invoke_expression_without_leaking(tmp_path):
    path = _path_with(tmp_path / "bin", "knowledge-rag-skills", "knowledge-rag")
    command = f"Get-Content -Raw -Path '{SKILLS_PS1}' | Invoke-Expression; 'leak=' + (Test-Path variable:forwarded)"
    result = _run_with_path([shutil.which("pwsh"), "-NoProfile", "-Command", command], path)
    assert result.returncode == 0, result.stderr
    assert "leak=False" in result.stdout


def test_subcommands_do_not_drift_from_the_wrappers():
    parser = skills_cli.build_parser()
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    python_commands = set(subparsers.choices)
    sh_line = next(
        line for line in SKILLS_SH.read_text(encoding="utf-8").splitlines() if "install | uninstall | status" in line
    )
    sh_commands = set(re.findall(r"\b(install|uninstall|status)\b", sh_line))
    ps_line = next(line for line in SKILLS_PS1.read_text(encoding="utf-8").splitlines() if "-cnotin @(" in line)
    ps_commands = set(re.findall(r"\b(install|uninstall|status)\b", ps_line))
    assert python_commands == sh_commands == ps_commands == {"install", "uninstall", "status"}


@pytest.mark.skipif(os.name != "nt", reason="Exercise the Windows PowerShell 5.1 wrapper")
def test_powershell_forwards_python_arguments_with_spaces(tmp_path):
    shell = shutil.which("powershell")
    if shell is None:
        pytest.skip("Windows PowerShell is unavailable")
    wrapper = tmp_path / "install.ps1"
    shutil.copyfile(ROOT / "install.ps1", wrapper)
    (tmp_path / "install.py").write_text("import json, sys; print(json.dumps(sys.argv[1:]))", encoding="utf-8")
    result = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-File",
            str(wrapper),
            "-SkipPython",
            "--dry-run",
            "--install-path",
            str(tmp_path / "space name"),
        ],
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == ["--dry-run", "--install-path", str(tmp_path / "space name")]
