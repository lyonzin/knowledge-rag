"""`knowledge-rag skills`: detection, folder layout, re-runs, confirmation and failure handling, on fake homes only."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mcp_server import assistants, skills_bundle, skills_cli
from mcp_server.skills_bundle import MANIFEST_FILE, SKILL_FILE, bundled_skills

ROOT = Path(__file__).resolve().parents[1]

# Oracle: the skills this release ships. Tests compare the bundle against it instead of deriving
# their expectations from the bundle itself.
EXPECTED_SKILLS = sorted(
    [
        "rag-check-first",
        "rag-cite-sources",
        "rag-code-review",
        "rag-deep-dive",
        "rag-evaluate-quality",
        "rag-index-decisions",
        "rag-onboard-context",
        "rag-security-first",
        "rag-troubleshoot",
        "rag-web-fallback",
    ]
)
SKILLS = bundled_skills()
NAMES = EXPECTED_SKILLS


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


def _manifest(target: Path) -> dict:
    return json.loads((target / MANIFEST_FILE).read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# The bundle and its content
# ─────────────────────────────────────────────────────────────────────────────


def test_bundled_skills_match_the_oracle():
    assert sorted(skill.name for skill in SKILLS) == EXPECTED_SKILLS
    assert len(EXPECTED_SKILLS) == len(set(EXPECTED_SKILLS))


def test_every_bundled_skill_folder_is_named_like_its_frontmatter():
    for skill in SKILLS:
        text = skill.content.decode("utf-8").replace("\r\n", "\n")
        assert text.startswith("---\n") and f"\nname: {skill.name}\n" in text, skill.name


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────


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


def test_claude_config_dir_value_expands_a_tilde(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    env = {"CLAUDE_CONFIG_DIR": "~/work"}
    monkeypatch.setenv("HOME", str(profile))
    monkeypatch.setenv("USERPROFILE", str(profile))
    assert assistants.skills_folder(_harness("claude-code"), _home(profile, env)) == profile / "work" / "skills"


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
        assert kwargs["stdin"] is subprocess.DEVNULL
        output = "USERPROFILE=C:\\Users\\Ana Maria\r\n\r\n" if command[0] == "cmd.exe" else f"{profile}\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(assistants.subprocess, "run", fake_run)
    windows = assistants.user_homes()[-1]
    assert (windows.path, windows.which, windows.suffix) == (profile, None, " (Windows)")
    assert calls == ["cmd.exe", "wslpath"]


def test_wsl_without_windows_interop_notes_it_and_scans_linux_only(monkeypatch, capsys):
    def missing(command, **kwargs):
        raise FileNotFoundError(command[0])

    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(assistants.subprocess, "run", missing)
    assert len(assistants.user_homes()) == 1
    assert "could not read the Windows profile" in capsys.readouterr().err


# ─────────────────────────────────────────────────────────────────────────────
# Install layout, adoption, and re-runs
# ─────────────────────────────────────────────────────────────────────────────


def test_install_writes_one_skill_md_per_folder_and_a_manifest(home, capsys):
    (home / ".claude").mkdir()
    code, out = _run(capsys, "install", "--yes")
    target = home / ".claude" / "skills"
    assert code == 0, out
    assert sorted(entry.name for entry in target.iterdir()) == sorted([*NAMES, MANIFEST_FILE])
    for skill in SKILLS:
        assert (target / skill.name / SKILL_FILE).read_bytes() == skill.content
    assert _manifest(target)["skills"] == {skill.name: skill.digest for skill in SKILLS}


def test_assistants_that_share_a_folder_get_one_question(home, capsys, monkeypatch):
    for marker in (".codex", ".cursor"):
        (home / marker).mkdir()
    asked = _answer(monkeypatch, "y")
    code, out = _run(capsys, "install")
    assert code == 0, out
    assert asked == ["Install the knowledge-rag skills for Codex CLI, Cursor? [Y/n] "]
    assert (home / ".agents" / "skills" / "rag-check-first" / SKILL_FILE).is_file()


def test_declined_question_writes_nothing(home, capsys, monkeypatch):
    (home / ".claude").mkdir()
    _answer(monkeypatch, "n")
    code, out = _run(capsys, "install")
    assert code == 0, out
    assert "Nothing was changed." in out
    assert not (home / ".claude" / "skills").exists()


def test_without_a_terminal_install_prints_the_plan_and_needs_yes(home, capsys):
    (home / ".claude").mkdir()
    code, out = _run(capsys, "install")
    assert code == 1
    assert f"{len(SKILLS)} to install" in out and "--yes" in out
    assert not (home / ".claude" / "skills").exists()


def test_dry_run_changes_nothing(home, capsys):
    (home / ".claude").mkdir()
    code, out = _run(capsys, "install", "--dry-run")
    assert code == 0 and "Dry run" in out
    assert not (home / ".claude" / "skills").exists()


def test_install_notes_a_missing_folder_it_will_create(home, capsys):
    code, out = _run(capsys, "install", "--target", str(home / "not there yet"), "--dry-run")
    assert code == 0, out
    assert "folder does not exist; it will be created" in out


def test_second_install_is_a_no_op_that_writes_nothing(home, capsys, monkeypatch):
    target = home / "skills dir"
    assert _run(capsys, "install", "--target", str(target), "--yes")[0] == 0
    real_write = skills_bundle._write_bytes
    writes = []
    monkeypatch.setattr(skills_bundle, "_write_bytes", lambda path, data: writes.append(path) or real_write(path, data))
    before = _manifest(target)
    code, out = _run(capsys, "install", "--target", str(target))
    assert code == 0, out
    assert writes == []
    assert _manifest(target) == before
    assert f"{len(SKILLS)} up to date" in out


def test_install_updates_unedited_copies_and_keeps_edited_and_untracked_ones(home, capsys):
    target = home / "skills dir"
    older, edited, untracked = SKILLS[0], SKILLS[1], SKILLS[2]
    copies = {older: b"older release\n", edited: b"my own notes\n", untracked: b"from another installer\n"}
    for skill, text in copies.items():
        (target / skill.name).mkdir(parents=True)
        (target / skill.name / SKILL_FILE).write_bytes(text)
    recorded_edited = b"what was installed\n"
    manifest = {
        "skills": {
            older.name: hashlib.sha256(b"older release\n").hexdigest(),
            edited.name: hashlib.sha256(recorded_edited).hexdigest(),
        }
    }
    (target / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    code, out = _run(capsys, "install", "--target", str(target), "--yes")
    assert code == 0, out
    assert (target / older.name / SKILL_FILE).read_bytes() == older.content
    for skill in (edited, untracked):
        assert (target / skill.name / SKILL_FILE).read_bytes() == copies[skill]
    assert f"edited (kept): {edited.name}" in out
    assert f"different copy not installed from here (kept): {untracked.name}" in out
    final = _manifest(target)["skills"]
    assert final[older.name] == older.digest
    assert final[edited.name] == manifest["skills"][edited.name]
    assert untracked.name not in final


def test_manual_copy_is_adopted_then_uninstalled(home, capsys):
    target = home / "skills dir"
    (target / SKILLS[0].name).mkdir(parents=True)
    (target / SKILLS[0].name / SKILL_FILE).write_bytes(SKILLS[0].content)
    code, out = _run(capsys, "install", "--target", str(target), "--yes", "--skill", SKILLS[0].name)
    assert code == 0, out
    assert _manifest(target)["skills"] == {SKILLS[0].name: SKILLS[0].digest}
    code, out = _run(capsys, "uninstall", "--target", str(target), "--yes", "--skill", SKILLS[0].name)
    assert code == 0, out
    assert list(target.iterdir()) == []


def test_uninstall_keeps_identical_copies_when_the_manifest_is_gone(home, capsys):
    target = home / "skills dir"
    assert _run(capsys, "install", "--target", str(target), "--yes")[0] == 0
    (target / MANIFEST_FILE).unlink()
    code, out = _run(capsys, "uninstall", "--target", str(target), "--yes")
    assert code == 0, out
    assert "identical copy, no record of installing it (kept)" in out
    assert (target / SKILLS[0].name / SKILL_FILE).is_file()


def test_install_that_writes_nothing_reports_it(home, capsys):
    target = home / "skills dir"
    for skill in SKILLS:
        (target / skill.name).mkdir(parents=True)
        (target / skill.name / SKILL_FILE).write_text("my version", encoding="utf-8")
    code, out = _run(capsys, "install", "--target", str(target), "--yes")
    assert code == 0, out
    assert f"Nothing installed: {len(SKILLS)} skills kept" in out
    assert "Start a new session" not in out


@pytest.mark.parametrize("selection", [["--skill", ""], ["--skill", ","]])
def test_an_empty_skill_selection_is_a_usage_error(home, selection):
    with pytest.raises(SystemExit) as caught:
        skills_cli.main(["install", "--target", str(home), *selection])
    assert caught.value.code == 2


# ─────────────────────────────────────────────────────────────────────────────
# Uninstall
# ─────────────────────────────────────────────────────────────────────────────


def test_uninstall_removes_only_unedited_folders_that_hold_nothing_else(home, capsys):
    target = home / "skills dir"
    assert _run(capsys, "install", "--target", str(target), "--yes")[0] == 0
    edited, crowded = SKILLS[0], SKILLS[1]
    (target / edited.name / SKILL_FILE).write_text("my version", encoding="utf-8")
    (target / crowded.name / "notes.md").write_text("mine", encoding="utf-8")
    code, out = _run(capsys, "uninstall", "--target", str(target), "--yes")
    assert code == 0, out
    assert sorted(entry.name for entry in target.iterdir()) == sorted([edited.name, crowded.name, MANIFEST_FILE])
    assert (target / crowded.name / SKILL_FILE).read_bytes() == crowded.content
    final = _manifest(target)["skills"]
    assert final[crowded.name] == crowded.digest
    assert edited.name in final  # the record stays; the copy is edited now


def test_uninstall_finds_installed_skills_after_the_assistant_is_gone(home, capsys):
    (home / ".codex").mkdir()
    assert _run(capsys, "install", "--yes")[0] == 0
    (home / ".codex").rmdir()
    code, out = _run(capsys, "uninstall", "--yes")
    assert code == 0, out
    assert list((home / ".agents" / "skills").iterdir()) == []


def test_uninstall_removes_orphan_entries_the_bundle_no_longer_has(home, capsys):
    target = home / "skills dir"
    assert _run(capsys, "install", "--target", str(target), "--yes")[0] == 0
    orphan = target / "rag-retired-skill"
    orphan.mkdir()
    (orphan / SKILL_FILE).write_text("old skill", encoding="utf-8")
    manifest = _manifest(target)
    manifest["skills"]["rag-retired-skill"] = hashlib.sha256(b"old skill").hexdigest()
    (target / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    code, out = _run(capsys, "uninstall", "--target", str(target), "--yes")
    assert code == 0, out
    assert "removed orphans no longer in the bundle" in out
    assert list(target.iterdir()) == []


def test_uninstall_prompt_defaults_to_no(home, capsys, monkeypatch):
    (home / ".claude").mkdir()
    assert _run(capsys, "install", "--yes")[0] == 0
    asked = _answer(monkeypatch, "")
    code, out = _run(capsys, "uninstall")
    assert code == 0, out
    assert asked == ["Remove the knowledge-rag skills for Claude Code? [y/N] "]
    assert "Nothing was changed." in out
    assert (home / ".claude" / "skills" / "rag-check-first" / SKILL_FILE).is_file()


# ─────────────────────────────────────────────────────────────────────────────
# Status, prompts, and misc behaviour
# ─────────────────────────────────────────────────────────────────────────────


def test_status_reports_and_writes_nothing(home, capsys):
    (home / ".claude").mkdir()
    code, out = _run(capsys, "status")
    assert code == 0
    assert "Claude Code: " in out and f"{len(SKILLS)} not installed" in out
    assert not (home / ".claude" / "skills").exists()


def test_an_invalid_answer_is_skipped(home, capsys, monkeypatch):
    (home / ".claude").mkdir()
    _answer(monkeypatch, "maybe")
    code, out = _run(capsys, "install")
    assert code == 0, out
    assert "skipped" in out
    assert not (home / ".claude" / "skills").exists()


def test_eof_at_a_prompt_changes_nothing(home, capsys, monkeypatch):
    (home / ".claude").mkdir()
    monkeypatch.setattr(skills_cli, "_stdin_is_terminal", lambda: True)

    def closed(question):
        raise EOFError

    monkeypatch.setattr("builtins.input", closed)
    code, out = _run(capsys, "install")
    assert code == 0, out
    assert "Nothing was changed." in out


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


def test_terminal_detection_accepts_a_tty(monkeypatch):
    class Tty:
        def isatty(self):
            return True

        def fileno(self):
            return 1

    monkeypatch.setattr(skills_cli.sys, "stdin", Tty())
    if os.name == "nt":
        import ctypes

        monkeypatch.setattr(ctypes.windll.kernel32, "GetConsoleMode", lambda handle, mode: 1)
    assert skills_cli._stdin_is_terminal() is True


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
    """The npx wrapper runs `python -m mcp_server.skills_cli`; its exit code must survive that path."""
    target = tmp_path / "skills dir"
    env = dict(
        os.environ,
        KNOWLEDGE_RAG_DIR=str(tmp_path / "base"),
        PYTHONPATH=str(ROOT),
        PYTHONIOENCODING="utf-8",
        HOME=str(tmp_path),
        USERPROFILE=str(tmp_path),
    )
    command = [sys.executable, "-m", "mcp_server.skills_cli", "install", "--target", str(target)]
    options = {"env": env, "stdin": subprocess.DEVNULL, "capture_output": True, "text": True, "timeout": 120}
    refused = subprocess.run(command, **options)
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert not target.exists()
    installed = subprocess.run([*command, "--yes"], **options)
    assert installed.returncode == 0, installed.stdout + installed.stderr
    assert sorted(entry.name for entry in target.iterdir()) == sorted([*NAMES, MANIFEST_FILE])


def test_non_latin_target_prints_with_the_stream_encoding(tmp_path):
    target = tmp_path / "skills 日本語"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONIOENCODING", "PYTHONUTF8", "KNOWLEDGE_RAG_DIR")
    }
    env.update(PYTHONPATH=str(ROOT), HOME=str(tmp_path), USERPROFILE=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "mcp_server.skills_cli", "status", "--target", str(target)],
        env=env,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert b"not installed" in result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# Correctness: racing edits, corrupt manifests, and mid-run failures
# ─────────────────────────────────────────────────────────────────────────────


def test_install_skips_a_skill_edited_between_planning_and_apply(home, capsys, monkeypatch):
    target = home / "skills dir"
    edited_name = SKILLS[0].name
    real_state = skills_bundle.skill_state

    def racing_state(folder, skill, manifest):
        state = real_state(folder, skill, manifest)
        if skill.name == edited_name:
            (folder / skill.name).mkdir(parents=True, exist_ok=True)
            (folder / skill.name / SKILL_FILE).write_text("edited mid-run", encoding="utf-8")
        return state

    monkeypatch.setattr(skills_bundle, "skill_state", racing_state)
    code, out = _run(capsys, "install", "--target", str(target), "--yes")
    assert code == 0, out
    assert (target / edited_name / SKILL_FILE).read_text(encoding="utf-8") == "edited mid-run"
    assert edited_name not in _manifest(target)["skills"]


def test_uninstall_skips_a_skill_edited_between_planning_and_delete(home, capsys, monkeypatch):
    target = home / "skills dir"
    assert _run(capsys, "install", "--target", str(target), "--yes")[0] == 0
    edited_name = SKILLS[0].name
    real_state = skills_bundle.skill_state

    def racing_state(folder, skill, manifest):
        state = real_state(folder, skill, manifest)
        if skill.name == edited_name:
            (folder / skill.name / SKILL_FILE).write_text("edited mid-run", encoding="utf-8")
        return state

    monkeypatch.setattr(skills_bundle, "skill_state", racing_state)
    code, out = _run(capsys, "uninstall", "--target", str(target), "--yes")
    assert code == 0, out
    assert (target / edited_name / SKILL_FILE).read_text(encoding="utf-8") == "edited mid-run"
    assert edited_name in _manifest(target)["skills"]
    assert sorted(entry.name for entry in target.iterdir() if entry.is_dir()) == [edited_name]


@pytest.mark.parametrize(
    "payload",
    [
        b'{"skills": {"rag-x": "ab',  # truncated JSON
        b"\xff\xfe\x00\x01",  # not UTF-8
        b"[]",  # top-level list
        b'{"skills": {"rag-x": 3}}',  # non-string digest
        b'{"skills": ["rag-x"]}',  # skills not an object
        b'\xef\xbb\xbf{"skills": {}}',  # UTF-8 BOM
        b"[" * 1000,  # deeper than the JSON parser allows
    ],
    ids=["truncated", "invalid-bytes", "list", "non-str-values", "skills-list", "bom", "deeply-nested"],
)
def test_a_corrupt_manifest_blocks_changes_and_is_left_as_is(home, capsys, payload):
    target = home / "skills dir"
    target.mkdir()
    manifest = target / MANIFEST_FILE
    manifest.write_bytes(payload)
    code, out = _run(capsys, "install", "--target", str(target), "--yes")
    assert code == 1, out
    assert "fix or delete" in out
    assert manifest.read_bytes() == payload
    assert list(target.iterdir()) == [manifest]


def test_a_write_failure_counts_what_was_done_and_keeps_the_other_folder_going(home, capsys, monkeypatch):
    failing = home / "failing"
    failing.mkdir()
    old = b"old release\n"
    (failing / SKILLS[0].name).mkdir()
    (failing / SKILLS[0].name / SKILL_FILE).write_bytes(old)
    (failing / MANIFEST_FILE).write_text(
        json.dumps({"skills": {SKILLS[0].name: hashlib.sha256(old).hexdigest()}}), encoding="utf-8"
    )
    working = home / "working"
    real_write = skills_bundle._write_bytes
    calls = {"count": 0}

    def failing_write(path, data):
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError(13, "Permission denied", str(path))
        return real_write(path, data)

    monkeypatch.setattr(skills_bundle, "_write_bytes", failing_write)
    code, out = _run(capsys, "install", "--target", str(failing), "--target", str(working), "--yes")
    assert code == 1, out
    assert f"failed after 0 of {len(SKILLS)}" in out
    assert (failing / SKILLS[0].name / SKILL_FILE).read_bytes() == old
    leftovers = [entry.name for entry in failing.iterdir() if entry.name.startswith(".knowledge-rag-")]
    assert leftovers == [MANIFEST_FILE]
    assert (working / SKILLS[0].name / SKILL_FILE).is_file()


def test_uninstall_writes_the_manifest_even_when_a_delete_fails(home, capsys, monkeypatch):
    target = home / "skills dir"
    assert _run(capsys, "install", "--target", str(target), "--yes")[0] == 0
    real_rmdir = Path.rmdir

    def failing_rmdir(self, *args, **kwargs):
        if self.name == SKILLS[-1].name:
            raise PermissionError(13, "Permission denied", str(self))
        return real_rmdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "rmdir", failing_rmdir)
    code, out = _run(capsys, "uninstall", "--target", str(target), "--yes")
    assert code == 1, out
    assert _manifest(target)["skills"] == {SKILLS[-1].name: SKILLS[-1].digest}


def test_a_manifest_write_failure_is_noted_on_the_primary_error(home, monkeypatch):
    plan = skills_bundle.plan_folder(home / "skills dir", SKILLS)
    monkeypatch.setattr(
        skills_bundle, "_write_bytes", lambda path, data: (_ for _ in ()).throw(PermissionError("disk gone"))
    )
    monkeypatch.setattr(skills_bundle, "_write_manifest", lambda plan_: (_ for _ in ()).throw(OSError("manifest gone")))
    with pytest.raises(skills_bundle.SkillWriteError) as caught:
        skills_bundle.install(plan, SKILLS)
    assert any("manifest gone" in note for note in caught.value.__notes__)


# ─────────────────────────────────────────────────────────────────────────────
# Target selection
# ─────────────────────────────────────────────────────────────────────────────


def test_explicit_targets_do_not_shell_out_for_wsl(home, capsys, monkeypatch):
    def explode():
        raise AssertionError("user_homes must not run for explicit targets")

    monkeypatch.setattr(skills_cli, "user_homes", explode)
    code, out = _run(capsys, "install", "--target", str(home / "skills"), "--yes")
    assert code == 0, out


def test_harness_selection_does_not_shell_out_for_wsl(home, capsys, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
    monkeypatch.setattr(skills_cli, "user_homes", lambda: pytest.fail("user_homes must not run"))
    code, out = _run(capsys, "install", "--harness", "claude-code", "--yes")
    assert code == 0, out
    assert (home / ".claude" / "skills" / "rag-check-first" / SKILL_FILE).is_file()


def test_empty_target_is_a_usage_error(home):
    with pytest.raises(SystemExit) as caught:
        skills_cli.main(["install", "--target", ""])
    assert caught.value.code == 2


@pytest.mark.parametrize("pattern", ["%KR_SKILLS_DIR%/skills"] if os.name == "nt" else ["$KR_SKILLS_DIR/skills"])
def test_target_expands_environment_variables(home, capsys, monkeypatch, pattern):
    monkeypatch.setenv("KR_SKILLS_DIR", str(home / "env dir"))
    target = home / "env dir" / "skills"
    code, out = _run(capsys, "install", "--target", pattern, "--yes")
    assert code == 0, out
    assert (target / "rag-check-first" / SKILL_FILE).is_file()
