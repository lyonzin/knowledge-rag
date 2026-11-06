"""Platform-specific behaviour of the skills installer: Windows links, and a real terminal prompt."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from mcp_server.skills_bundle import SKILL_FILE, bundled_skills

ROOT = Path(__file__).resolve().parents[1]
SKILLS = bundled_skills()


def _junction(link: Path, target: Path) -> None:
    """mklink /J needs no elevation; skip when it is unavailable."""
    if os.name != "nt":
        pytest.skip("junctions exist on Windows only")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        pytest.skip(f"mklink /J failed: {result.stderr.strip()}")


def _is_junction(path: Path) -> bool:
    return os.name == "nt" and getattr(os.lstat(path), "st_reparse_tag", 0) == stat.IO_REPARSE_TAG_MOUNT_POINT


@pytest.mark.skipif(os.name != "nt", reason="Windows junctions")
def test_install_writes_nothing_through_a_junction(tmp_path, capsys):
    from mcp_server import skills_cli

    target = tmp_path / "skills dir"
    target.mkdir()
    real = tmp_path / "real skill"
    real.mkdir()
    (real / SKILL_FILE).write_text("user content", encoding="utf-8")
    _junction(target / SKILLS[0].name, real)
    code = skills_cli.main(["install", "--target", str(target), "--yes"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert (real / SKILL_FILE).read_text(encoding="utf-8") == "user content"
    assert _is_junction(target / SKILLS[0].name)


@pytest.mark.skipif(os.name != "nt", reason="Windows junctions")
def test_uninstall_leaves_a_junction_target_intact(tmp_path, capsys):
    from mcp_server import skills_cli

    target = tmp_path / "skills dir"
    target.mkdir()
    real = tmp_path / "real skill"
    real.mkdir()
    (real / SKILL_FILE).write_text("user content", encoding="utf-8")
    _junction(target / SKILLS[0].name, real)
    code = skills_cli.main(["uninstall", "--target", str(target), "--yes"])
    assert code == 0, capsys.readouterr().out
    assert (real / SKILL_FILE).read_text(encoding="utf-8") == "user content"
    assert _is_junction(target / SKILLS[0].name)


@pytest.mark.skipif(os.name != "nt", reason="Windows junctions")
def test_install_and_uninstall_resolve_a_linked_skills_root(tmp_path, capsys):
    from mcp_server import skills_cli

    real = tmp_path / "real skills"
    real.mkdir()
    link = tmp_path / "linked skills"
    _junction(link, real)
    code = skills_cli.main(["install", "--target", str(link), "--yes"])
    assert code == 0, capsys.readouterr().out
    assert (real / "rag-check-first" / SKILL_FILE).is_file()
    assert _is_junction(link)
    code = skills_cli.main(["uninstall", "--target", str(link), "--yes"])
    assert code == 0, capsys.readouterr().out
    assert list(real.iterdir()) == []
    assert _is_junction(link)


@pytest.mark.skipif(os.name != "nt", reason="Windows symlinks")
def test_install_and_uninstall_leave_a_linked_skill_file_untouched(tmp_path, capsys):
    from mcp_server import skills_cli

    target = tmp_path / "skills dir"
    (target / SKILLS[0].name).mkdir(parents=True)
    real = tmp_path / "real.md"
    real.write_text("user content", encoding="utf-8")
    result = subprocess.run(
        ["cmd", "/c", "mklink", str(target / SKILLS[0].name / SKILL_FILE), str(real)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.skip(f"file symlinks need extra rights here: {result.stderr.strip()}")
    for command in ("install", "uninstall"):
        code = skills_cli.main([command, "--target", str(target), "--yes"])
        assert code == 0, capsys.readouterr().out
        assert real.read_text(encoding="utf-8") == "user content"
    assert target.joinpath(SKILLS[0].name, SKILL_FILE).is_symlink()


@pytest.mark.skipif(os.name == "nt", reason="pty exists on POSIX")
def test_prompt_reads_an_answer_from_a_terminal(tmp_path):
    import pty

    target = tmp_path / "skills dir"
    env = {key: value for key, value in os.environ.items() if key != "KNOWLEDGE_RAG_DIR"}
    env.update(
        PYTHONPATH=str(ROOT),
        KNOWLEDGE_RAG_DIR=str(tmp_path / "base"),
        HOME=str(tmp_path),
        USERPROFILE=str(tmp_path),
    )
    pid, master = pty.fork()
    if pid == 0:
        os.execve(
            sys.executable,
            [sys.executable, "-m", "mcp_server.skills_cli", "install", "--target", str(target)],
            env,
        )
    output = b""
    while b"[Y/n]" not in output:
        chunk = os.read(master, 4096)
        if not chunk:
            break
        output += chunk
    assert b"[Y/n]" in output, output
    os.write(master, b"y\n")
    while True:
        try:
            chunk = os.read(master, 4096)
        except OSError:
            break
        if not chunk:
            break
        output += chunk
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, output.decode(errors="replace")
    assert (target / "rag-check-first" / SKILL_FILE).is_file()
