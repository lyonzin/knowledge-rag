"""Which AI coding assistants the current user has, and where each loads user skills from.

Detection only reads: it checks for an assistant's config folder in the user
profile (``%USERPROFILE%`` on Windows, never the OneDrive-redirected Documents
folder) and for its executable on PATH. Under WSL it also scans the Windows
profile, for config folders only, because PATH there describes the Linux side.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

_AGENTS = "~/.agents/skills"  # user folder of the Agent Skills convention, read by most assistants
_WSL_CODEPAGE = 65001  # makes cmd.exe print a non-ASCII profile name as UTF-8
_WSL_TIMEOUT = 15


@dataclass(frozen=True)
class Harness:
    """An AI coding assistant: how to detect it and where it loads user skills from.

    Paths are templates: ``~/sub`` is relative to the user profile and
    ``$VAR/sub`` to an environment variable, skipped while it is unset. The
    first ``skills_dirs`` entry that resolves wins; an empty tuple means the
    assistant has no ``<name>/SKILL.md`` folder to install into.
    """

    name: str
    label: str
    markers: tuple[str, ...]
    commands: tuple[str, ...]
    skills_dirs: tuple[str, ...]


# Skills folders as each assistant's official docs give them in November 2026;
# skills/README.md links those docs.
HARNESSES: tuple[Harness, ...] = (
    Harness(
        "claude-code",
        "Claude Code",
        ("$CLAUDE_CONFIG_DIR", "~/.claude"),
        ("claude",),
        ("$CLAUDE_CONFIG_DIR/skills", "~/.claude/skills"),
    ),
    Harness("codex", "Codex CLI", ("$CODEX_HOME", "~/.codex"), ("codex",), (_AGENTS,)),
    Harness("copilot", "GitHub Copilot CLI", ("~/.copilot",), ("copilot",), (_AGENTS,)),
    Harness("gemini", "Gemini CLI", ("~/.gemini",), ("gemini",), (_AGENTS,)),
    Harness("cursor", "Cursor", ("~/.cursor",), ("cursor",), (_AGENTS,)),
    Harness("opencode", "OpenCode", ("~/.config/opencode",), ("opencode",), (_AGENTS,)),
    Harness("windsurf", "Windsurf", ("~/.codeium/windsurf",), ("windsurf",), (_AGENTS,)),
    Harness("zed", "Zed", ("~/.config/zed", "$APPDATA/Zed"), ("zed",), (_AGENTS,)),
    Harness("cline", "Cline", ("~/.cline",), ("cline",), ("~/.cline/skills",)),
    Harness("aider", "Aider", ("~/.aider.conf.yml",), ("aider",), ()),
)


@dataclass(frozen=True)
class Home:
    """A user profile to scan. ``which`` is None where PATH does not describe it."""

    path: Path
    env: Mapping[str, str]
    which: Callable[[str], str | None] | None
    suffix: str = ""


def _expand(template: str, home: Home) -> Path | None:
    """Resolve ``~/sub`` against the profile and ``$VAR/sub`` against its environment."""
    head, _, rest = template.partition("/")
    if head == "~":
        base = home.path
    elif home.env.get(head[1:]):
        base = Path(os.path.expanduser(home.env[head[1:]]))
    else:
        return None
    return base / rest if rest else base


def detect(home: Home) -> list[Harness]:
    """Assistants with a config folder in ``home`` or, where PATH applies, an executable on it."""
    return [harness for harness in HARNESSES if _is_present(harness, home)]


def _is_present(harness: Harness, home: Home) -> bool:
    paths = [_expand(marker, home) for marker in harness.markers]
    if any(path is not None and path.exists() for path in paths):
        return True
    which = home.which
    return which is not None and any(which(command) for command in harness.commands)


def skills_folder(harness: Harness, home: Home) -> Path | None:
    """The folder ``harness`` loads user skills from in ``home``, or None when it has none."""
    for template in harness.skills_dirs:
        folder = _expand(template, home)
        if folder is not None:
            return folder
    return None


def user_homes() -> list[Home]:
    """The current user profile and, under WSL, the Windows profile."""
    homes = [Home(Path.home(), os.environ, shutil.which)]
    windows = _wsl_windows_profile()
    if windows is not None:
        homes.append(Home(windows, {}, None, " (Windows)"))
    return homes


def _wsl_windows_profile() -> Path | None:
    """Under WSL, the Windows user profile as a Linux path, such as /mnt/c/Users/Ana."""
    if not os.environ.get("WSL_DISTRO_NAME"):
        return None
    profile = None
    try:
        # `set USERPROFILE` prints the value verbatim; `echo %USERPROFILE%` would let a profile
        # name with & or | break the command. Only the last non-empty line is the answer.
        windows = _run(["cmd.exe", "/d", "/c", f"chcp {_WSL_CODEPAGE} >nul & set USERPROFILE"])
        lines = [line.strip() for line in windows.splitlines() if line.strip()]
        windows = lines[-1] if lines else ""
        linux = _run(["wslpath", "-u", windows])
        candidate = Path(linux)
        if windows and "%" not in windows and candidate.is_dir():
            profile = candidate
    except (OSError, subprocess.SubprocessError):
        profile = None
    if profile is None:
        print(
            "[WARN] could not read the Windows profile under WSL; use --target to point at its skills folder",
            file=sys.stderr,
        )
    return profile


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=_WSL_TIMEOUT,
        check=True,
    )
    return completed.stdout.strip()
