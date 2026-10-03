"""Run installer shells with local fixtures; no downloads or global installs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _bash() -> str:
    executable = shutil.which("bash")
    if not executable and os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        executable = str(candidate) if candidate.is_file() else None
    if not executable:
        pytest.skip("Bash is not installed on this test host")
    return executable


def _skill_fixture(tmp_path, *, exit_code=0, payload="replacement"):
    script = tmp_path / "install.sh"
    script.write_text((ROOT / "skills/install.sh").read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    tools = tmp_path / "bin"
    tools.mkdir()
    downloader = tools / "curl"
    downloader.write_text(
        '#!/usr/bin/env bash\nwhile [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then target="$2"; shift 2; else shift; fi\ndone\n'
        f"printf '%s' '{payload}' > \"$target\"\nexit {exit_code}\n",
        encoding="utf-8",
        newline="\n",
    )
    downloader.chmod(0o755)
    # Git Bash prepends its own bin directory to PATH on startup. Bind a shell
    # function instead so tests can never accidentally call the real downloader.
    startup = tmp_path / "test-shell-env.sh"
    startup.write_text('curl() { "$AUDIT_FAKE_CURL" "$@"; }\nwget() { return 99; }\n', encoding="utf-8", newline="\n")
    environment = dict(os.environ, BASH_ENV=startup.as_posix(), AUDIT_FAKE_CURL=downloader.as_posix())
    return script, environment


def _run_skill(script, environment, target, *extra):
    return subprocess.run(
        [_bash(), str(script), "--target", target.as_posix(), "--only", "rag-check-first", *extra],
        env=environment,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(("exit_code", "payload"), [(22, "partial download"), (0, "")])
def test_remote_skill_failure_preserves_existing_bytes(tmp_path, exit_code, payload):
    script, environment = _skill_fixture(tmp_path, exit_code=exit_code, payload=payload)
    target = tmp_path / "skills with spaces"
    target.mkdir()
    destination = target / "rag-check-first.md"
    destination.write_bytes(b"existing user skill\n")
    result = _run_skill(script, environment, target)
    assert result.returncode == 2, result.stderr + result.stdout
    assert destination.read_bytes() == b"existing user skill\n"
    assert list(target.iterdir()) == [destination]


def test_remote_skill_success_replaces_complete_file(tmp_path):
    script, environment = _skill_fixture(tmp_path)
    target = tmp_path / "skills with spaces"
    target.mkdir()
    destination = target / "rag-check-first.md"
    destination.write_text("old", encoding="utf-8")
    result = _run_skill(script, environment, target)
    assert result.returncode == 0, result.stderr + result.stdout
    assert destination.read_text(encoding="utf-8") == "replacement"
    assert list(target.iterdir()) == [destination]


def test_skill_dry_run_does_not_create_target(tmp_path):
    script, environment = _skill_fixture(tmp_path)
    target = tmp_path / "not created"
    result = _run_skill(script, environment, target, "--dry-run")
    assert result.returncode == 0, result.stderr + result.stdout
    assert not target.exists()


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
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout.splitlines()[-1]) == ["--dry-run", "--install-path", str(tmp_path / "space name")]
