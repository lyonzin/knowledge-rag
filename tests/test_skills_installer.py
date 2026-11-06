"""`knowledge-rag skills`: detection, folder layout, re-runs and confirmation, on fake homes only."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mcp_server import assistants, skills_cli
from mcp_server.skills_bundle import MANIFEST_FILE, bundled_skills

ROOT = Path(__file__).resolve().parents[1]
SKILLS = bundled_skills()
NAMES = sorted(skill.name for skill in SKILLS)


def _home(path, env=None, commands=()):
    return assistants.Home(path, env or {}, lambda name: f"/opt/bin/{name}" if name in commands else None)


def _detected(home):
    return [harness.name for harness in assistants.detect(home)]


def _harness(name):
    return next(harness for harness in assistants.HARNESSES if harness.name == name)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A profile path with a space; no terminal and no answers unless a test sets them."""
    profile = tmp_path / "Ana Maria"
    profile.mkdir()
    monkeypatch.setattr(skills_cli, "user_homes", lambda: [_home(profile)])
    monkeypatch.setattr(skills_cli, "_stdin_is_terminal", lambda: False)
    monkeypatch.setattr("builtins.input", lambda question: pytest.fail(f"asked without a terminal: {question}"))
    return profile


def _run(capsys, *argv):
    code = skills_cli.main(list(argv))
    return code, capsys.readouterr().out


def _answer(monkeypatch, reply):
    """Make stdin a terminal that gives ``reply`` to every question; return the questions asked."""
    asked = []
    monkeypatch.setattr(skills_cli, "_stdin_is_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", lambda question: asked.append(question) or reply)
    return asked


def test_every_bundled_skill_folder_is_named_like_its_frontmatter():
    assert len(NAMES) == len(set(NAMES)) >= 10
    for skill in SKILLS:
        text = skill.content.decode("utf-8").replace("\r\n", "\n")
        assert text.startswith("---\n") and f"\nname: {skill.name}\n" in text, skill.name


def test_config_folders_identify_assistants_in_a_profile_with_spaces(tmp_path):
    profile = tmp_path / "Ana Maria"
    for marker in (".claude", ".codex", ".config/opencode"):
        (profile / marker).mkdir(parents=True)
    (profile / ".aider.conf.yml").write_text("model: local\n", encoding="utf-8")
    assert _detected(_home(profile)) == ["claude-code", "codex", "opencode", "aider"]


def test_executables_on_path_identify_assistants(tmp_path):
    assert _detected(_home(tmp_path, commands={"gemini", "cline"})) == ["gemini", "cline"]


def test_a_profile_without_path_is_judged_by_config_folders_only(tmp_path):
    assert assistants.detect(assistants.Home(tmp_path, {}, None)) == []


def test_environment_overrides_move_claude_code_and_reveal_codex(tmp_path):
    profile = tmp_path / "profile"
    custom = tmp_path / "claude work"
    codex_home = tmp_path / "codex home"
    codex_home.mkdir()
    env = {"CLAUDE_CONFIG_DIR": str(custom), "CODEX_HOME": str(codex_home)}
    assert assistants.skills_folder(_harness("claude-code"), _home(profile, env)) == custom / "skills"
    assert assistants.skills_folder(_harness("claude-code"), _home(profile)) == profile / ".claude" / "skills"
    assert _detected(_home(profile, env)) == ["codex"]


@pytest.mark.skipif(os.name != "nt", reason="USERPROFILE decides the profile on Windows only")
def test_windows_profile_is_userprofile_even_when_home_points_into_onedrive(tmp_path, monkeypatch):
    profile = tmp_path / "Users" / "Ana Maria"
    monkeypatch.setenv("USERPROFILE", str(profile))
    monkeypatch.setenv("HOME", str(profile / "OneDrive" / "Documents"))
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    assert [home.path for home in assistants.user_homes()] == [profile]


def test_wsl_also_offers_the_windows_profile(tmp_path, monkeypatch):
    profile = tmp_path / "mnt" / "c" / "Users" / "Ana Maria"
    profile.mkdir(parents=True)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command[0])
        output = "C:\\Users\\Ana Maria\r\n" if command[0] == "cmd.exe" else f"{profile}\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(assistants.subprocess, "run", fake_run)
    windows = assistants.user_homes()[-1]
    assert (windows.path, windows.which, windows.suffix) == (profile, None, " (Windows)")
    assert calls == ["cmd.exe", "wslpath"]


def test_wsl_without_windows_interop_scans_the_linux_profile_only(monkeypatch):
    def missing(command, **kwargs):
        raise FileNotFoundError(command[0])

    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(assistants.subprocess, "run", missing)
    assert len(assistants.user_homes()) == 1


def test_install_writes_one_skill_md_per_folder_and_a_manifest(home, capsys):
    (home / ".claude").mkdir()
    code, out = _run(capsys, "install", "--yes")
    target = home / ".claude" / "skills"
    assert code == 0, out
    assert sorted(entry.name for entry in target.iterdir()) == sorted([*NAMES, MANIFEST_FILE])
    for skill in SKILLS:
        assert (target / skill.name / "SKILL.md").read_bytes() == skill.content
    manifest = json.loads((target / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["skills"] == {skill.name: skill.digest for skill in SKILLS}


def test_assistants_that_share_a_folder_get_one_question(home, capsys, monkeypatch):
    for marker in (".codex", ".cursor"):
        (home / marker).mkdir()
    asked = _answer(monkeypatch, "y")
    code, out = _run(capsys, "install")
    assert code == 0, out
    assert asked == ["Install the knowledge-rag skills for Codex CLI, Cursor? [Y/n] "]
    assert (home / ".agents" / "skills" / "rag-check-first" / "SKILL.md").is_file()


def test_declined_question_writes_nothing(home, capsys, monkeypatch):
    (home / ".claude").mkdir()
    _answer(monkeypatch, "n")
    code, out = _run(capsys, "install")
    assert code == 0, out
    assert not (home / ".claude" / "skills").exists()


def test_ctrl_c_at_a_question_stops_the_run(home, monkeypatch):
    for marker in (".claude", ".codex"):
        (home / marker).mkdir()
    monkeypatch.setattr(skills_cli, "_stdin_is_terminal", lambda: True)

    def interrupt(question):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    with pytest.raises(SystemExit) as caught:
        skills_cli.main(["install"])
    assert caught.value.code == 130
    assert not (home / ".claude" / "skills").exists() and not (home / ".agents").exists()


def test_without_a_terminal_install_prints_the_plan_and_needs_yes(home, capsys):
    (home / ".claude").mkdir()
    code, out = _run(capsys, "install")
    assert code == 1
    assert f"{len(SKILLS)} not installed" in out and "--yes" in out
    assert not (home / ".claude" / "skills").exists()


def test_dry_run_changes_nothing(home, capsys):
    (home / ".claude").mkdir()
    code, out = _run(capsys, "install", "--dry-run")
    assert code == 0 and "Dry run" in out
    assert not (home / ".claude" / "skills").exists()


def test_second_install_is_a_no_op_that_needs_no_confirmation(home, capsys):
    target = home / "skills dir"
    assert _run(capsys, "install", "--target", str(target), "--yes")[0] == 0
    before = {path: path.stat().st_mtime_ns for path in target.rglob("*")}
    code, out = _run(capsys, "install", "--target", str(target))
    assert code == 0, out
    assert f"{len(SKILLS)} up to date" in out
    assert {path: path.stat().st_mtime_ns for path in target.rglob("*")} == before


def test_install_updates_unedited_copies_and_keeps_edited_and_untracked_ones(home, capsys):
    target = home / "skills dir"
    older, edited, untracked = SKILLS[0], SKILLS[1], SKILLS[2]
    copies = {older: b"older release\n", edited: b"my own notes\n", untracked: b"from another installer\n"}
    for skill, text in copies.items():
        (target / skill.name).mkdir(parents=True)
        (target / skill.name / "SKILL.md").write_bytes(text)
    recorded = {older.name: b"older release\n", edited.name: b"what was installed\n"}
    manifest = {"skills": {name: hashlib.sha256(text).hexdigest() for name, text in recorded.items()}}
    (target / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    code, out = _run(capsys, "install", "--target", str(target), "--yes")
    assert code == 0, out
    assert (target / older.name / "SKILL.md").read_bytes() == older.content
    for skill in (edited, untracked):
        assert (target / skill.name / "SKILL.md").read_bytes() == copies[skill]
    assert f"edited since installed (kept): {edited.name}" in out
    assert f"different copy not installed from here (kept): {untracked.name}" in out


def test_uninstall_removes_only_unedited_folders_that_hold_nothing_else(home, capsys):
    target = home / "skills dir"
    assert _run(capsys, "install", "--target", str(target), "--yes")[0] == 0
    edited, crowded = SKILLS[0], SKILLS[1]
    (target / edited.name / "SKILL.md").write_text("my version", encoding="utf-8")
    (target / crowded.name / "notes.md").write_text("mine", encoding="utf-8")
    code, out = _run(capsys, "uninstall", "--target", str(target), "--yes")
    assert code == 0, out
    assert sorted(entry.name for entry in target.iterdir()) == sorted([edited.name, crowded.name, MANIFEST_FILE])
    assert (target / crowded.name / "SKILL.md").read_bytes() == crowded.content


def test_uninstall_finds_installed_skills_after_the_assistant_is_gone(home, capsys):
    (home / ".codex").mkdir()
    assert _run(capsys, "install", "--yes")[0] == 0
    (home / ".codex").rmdir()
    code, out = _run(capsys, "uninstall", "--yes")
    assert code == 0, out
    assert list((home / ".agents" / "skills").iterdir()) == []


def test_status_reports_and_writes_nothing(home, capsys):
    (home / ".claude").mkdir()
    code, out = _run(capsys, "status")
    assert code == 0
    assert "Claude Code: " in out and f"{len(SKILLS)} not installed" in out
    assert not (home / ".claude" / "skills").exists()


def test_symlinked_skill_folder_is_left_alone(home, capsys, tmp_path):
    target = home / "skills dir"
    target.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    try:
        (target / SKILLS[0].name).symlink_to(elsewhere, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks needs extra rights on this host")
    code, out = _run(capsys, "install", "--target", str(target), "--yes")
    assert code == 0, out
    assert list(elsewhere.iterdir()) == [] and "left alone" in out


def test_skill_selection_installs_the_named_skills_only(home, capsys):
    target = home / "skills dir"
    code, out = _run(capsys, "install", "--target", str(target), "--yes", "--only", "rag-check-first,rag-cite-sources")
    assert code == 0, out
    assert sorted(entry.name for entry in target.iterdir() if entry.is_dir()) == ["rag-check-first", "rag-cite-sources"]


def test_unknown_skill_is_a_usage_error(home):
    with pytest.raises(SystemExit) as caught:
        skills_cli.main(["install", "--skill", "rag-nope"])
    assert caught.value.code == 2


def test_assistant_without_a_skills_folder_gets_a_pointer(home, capsys):
    (home / ".aider.conf.yml").write_text("", encoding="utf-8")
    code, out = _run(capsys, "install", "--yes")
    assert code == 1
    assert "Aider: no skills folder to install into" in out and "#other-assistants" in out


def test_server_main_routes_the_skills_subcommand(home, capsys, monkeypatch):
    from mcp_server import server

    monkeypatch.setattr(sys, "argv", ["knowledge-rag", "skills", "status", "--target", str(home)])
    with pytest.raises(SystemExit) as caught:
        server.main()
    assert caught.value.code == 0
    assert f"{len(SKILLS)} not installed" in capsys.readouterr().out


def test_module_entry_point_needs_yes_without_a_terminal_and_installs_with_it(tmp_path):
    """The npx wrapper runs `python -m mcp_server.server`; its exit code must survive that path."""
    target = tmp_path / "skills dir"
    env = dict(os.environ, KNOWLEDGE_RAG_DIR=str(tmp_path / "base"), PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    command = [sys.executable, "-m", "mcp_server.server", "skills", "install", "--target", str(target)]
    options = {"env": env, "stdin": subprocess.DEVNULL, "capture_output": True, "text": True, "timeout": 120}
    refused = subprocess.run(command, **options)
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert not target.exists()
    installed = subprocess.run([*command, "--yes"], **options)
    assert installed.returncode == 0, installed.stdout + installed.stderr
    assert sorted(entry.name for entry in target.iterdir()) == sorted([*NAMES, MANIFEST_FILE])
