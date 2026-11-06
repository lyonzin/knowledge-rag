"""``knowledge-rag skills install|uninstall|status``: put the bundled skills where assistants load them.

Without ``--harness``, ``--target`` or ``--project`` it acts on the skills
folder of every assistant it detects, plus any folder that already holds the
skills, and asks once per folder. When stdin is not a terminal it prints the
plan and changes nothing unless ``--yes`` is given.

Example::

    knowledge-rag skills install --dry-run
    knowledge-rag skills install --harness claude-code --yes
    knowledge-rag skills status
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from . import skills_bundle as bundle
from .assistants import HARNESSES, Home, detect, skills_folder, user_homes

# Outcome of one folder.
DONE, DECLINED, NEEDS_YES, FAILED = "done", "declined", "needs-yes", "failed"

_MANUAL_SETUP = "https://github.com/lyonzin/knowledge-rag/blob/master/skills/README.md#other-assistants"
_WORDS = {
    bundle.MISSING: "not installed",
    bundle.OUTDATED: "older version",
    bundle.CURRENT: "up to date",
    bundle.EDITED: "edited since installed (kept)",
    bundle.UNTRACKED: "different copy not installed from here (kept)",
    bundle.FOREIGN: "not a plain folder (left alone)",
}
_KEPT = (bundle.EDITED, bundle.UNTRACKED)
_NAMED = (*_KEPT, bundle.FOREIGN)  # states whose skills are listed by name

Targets = dict[Path, list[str]]  # skills folder -> the assistants (labels) that read it


def _note(label: str) -> str:
    return f"{label}: no skills folder to install into; see {_MANUAL_SETUP}"


def _detected_targets(homes: Sequence[Home], skills: Sequence[bundle.Skill]) -> tuple[Targets, list[str]]:
    """Folders a detected assistant reads, plus folders that already hold the skills."""
    folders: Targets = {}
    notes: list[str] = []
    for home in homes:
        detected = detect(home)
        for harness in HARNESSES:
            folder = skills_folder(harness, home)
            label = harness.label + home.suffix
            if folder is None and harness in detected:
                notes.append(_note(label))
            elif folder is not None and harness in detected:
                folders.setdefault(folder, []).append(label)
            elif folder is not None and bundle.holds_skills(folder, skills):
                folders.setdefault(folder, [])
    return folders, notes


def _explicit_targets(args: argparse.Namespace, home: Home) -> tuple[Targets, list[str]]:
    """Folders given with --target or --project, plus the folders of the --harness assistants."""
    folders: Targets = {Path(os.path.abspath(Path(raw).expanduser())): [] for raw in args.target or []}
    if args.project:
        folders.setdefault(Path.cwd() / ".claude" / "skills", []).append("Claude Code (this project)")
    notes = []
    for harness in [harness for harness in HARNESSES if harness.name in (args.harness or [])]:
        folder = skills_folder(harness, home)
        if folder is None:
            notes.append(_note(harness.label))
        else:
            folders.setdefault(folder, []).append(harness.label)
    return folders, notes


def _summary(plan: bundle.Plan) -> str:
    """Count the skills per state, naming the ones the user should look at."""
    lines = []
    for state, words in _WORDS.items():
        names = [name for name, value in plan.states.items() if value == state]
        if names:
            lines.append(f"  {len(names)} {words}" + (f": {', '.join(names)}" if state in _NAMED else ""))
    return "\n".join(lines)


def _has_work(command: str, plan: bundle.Plan) -> bool:
    wanted = (bundle.MISSING, bundle.OUTDATED) if command == "install" else (bundle.CURRENT, bundle.OUTDATED)
    return any(state in wanted for state in plan.states.values())


def _stdin_is_terminal() -> bool:
    """True for a real terminal. On Windows the NUL device passes isatty(), so ask the console API too."""
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            return False
        if sys.platform == "win32":
            import ctypes
            import msvcrt

            mode = ctypes.c_uint32()
            handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
        else:
            return True
    except (OSError, ValueError):  # a replaced stdin without a usable descriptor
        return False


def _confirm(command: str, folder: Path, labels: Sequence[str]) -> bool:
    verb = "Install" if command == "install" else "Remove"
    where = f"for {', '.join(labels)}" if labels else f"in {folder}"
    try:
        answer = input(f"{verb} the knowledge-rag skills {where}? [Y/n] ")
    except EOFError:
        return False
    except KeyboardInterrupt:  # Ctrl+C stops the whole run, not just this folder
        print()
        raise SystemExit(130) from None
    return answer.strip().lower() in ("", "y", "yes")


def _apply(command: str, plan: bundle.Plan, skills: Sequence[bundle.Skill]) -> str:
    """Change one folder, reporting an OS error instead of stopping at it."""
    try:
        if command == "install":
            bundle.install(plan, skills)
            print("  done")
        else:
            kept = bundle.uninstall(plan, skills)
            print("  done" + (f"; kept, because they hold other files: {', '.join(kept)}" if kept else ""))
    except OSError as exc:
        print(f"  failed: {exc}")
        return FAILED
    return DONE


def _handle(args: argparse.Namespace, folder: Path, labels: Sequence[str], skills: Sequence[bundle.Skill]) -> str:
    """Report one folder, then apply the command to it once confirmed."""
    print(f"\n{', '.join(labels) or 'Skills folder'}: {folder}")
    try:
        plan = bundle.plan_folder(folder, skills)
    except OSError as exc:
        print(f"  cannot read it: {exc}")
        return FAILED
    print(_summary(plan))
    if args.command == "install" and any(state in _KEPT for state in plan.states.values()):
        print("  To install this version over a kept skill, delete its folder first.")
    if args.command == "status" or args.dry_run or not _has_work(args.command, plan):
        return DONE
    if not args.yes and not _stdin_is_terminal():
        return NEEDS_YES
    if not args.yes and not _confirm(args.command, folder, labels):
        return DECLINED
    return _apply(args.command, plan, skills)


def _select(
    parser: argparse.ArgumentParser, skills: list[bundle.Skill], requested: list[str] | None
) -> list[bundle.Skill]:
    if not requested:
        return skills
    wanted = {name.strip() for value in requested for name in value.split(",") if name.strip()}
    unknown = sorted(wanted - {skill.name for skill in skills})
    if unknown:
        parser.error(f"unknown skill {', '.join(unknown)}; available: {', '.join(skill.name for skill in skills)}")
    return [skill for skill in skills if skill.name in wanted]


def build_parser() -> argparse.ArgumentParser:
    """Command line of ``knowledge-rag skills``."""
    parser = argparse.ArgumentParser(
        prog="knowledge-rag skills",
        description="Install the knowledge-rag skills for the AI coding assistants of the current user.",
        epilog="Docs: https://github.com/lyonzin/knowledge-rag/blob/master/skills/README.md",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, summary in (
        ("install", "install or update the skills, asking once per skills folder"),
        ("uninstall", "remove the skills installed from here that nobody edited"),
        ("status", "show what is installed where, changing nothing"),
    ):
        command = commands.add_parser(name, help=summary, description=summary)
        _add_selection(command)
        if name == "status":
            command.set_defaults(yes=False, dry_run=False)
            continue
        command.add_argument("-y", "--yes", action="store_true", help="do not ask; needed when stdin is not a terminal")
        command.add_argument("--dry-run", action="store_true", help="print the plan and change nothing")
    return parser


def _add_selection(command: argparse.ArgumentParser) -> None:
    names = [harness.name for harness in HARNESSES]
    command.add_argument(
        "--harness", action="append", choices=names, metavar="NAME", help=f"this assistant only: {', '.join(names)}"
    )
    command.add_argument("--target", action="append", metavar="DIR", help="this skills folder only")
    command.add_argument("--project", action="store_true", help="./.claude/skills in the current folder only")
    command.add_argument("--skill", "--only", action="append", metavar="NAME", help="these skills only (a,b or repeat)")


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``knowledge-rag skills``; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        skills = _select(parser, bundle.bundled_skills(), args.skill)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    homes = user_homes()
    explicit = bool(args.target or args.harness or args.project)
    folders, notes = _explicit_targets(args, homes[0]) if explicit else _detected_targets(homes, skills)
    for note in notes:
        print(note)
    if not folders:
        print("No skills folder to act on. Name an assistant with --harness or a folder with --target.")
        return 1 if args.command == "install" else 0
    return _finish(args, [_handle(args, folder, labels, skills) for folder, labels in folders.items()])


def _finish(args: argparse.Namespace, outcomes: Sequence[str]) -> int:
    if NEEDS_YES in outcomes:
        print(f"\nNothing was changed: stdin is not a terminal. Re-run with --yes to {args.command} as planned above.")
        return 1
    if args.dry_run:
        print("\nDry run: nothing was written.")
    elif args.command == "install" and DONE in outcomes:
        print("\nStart a new session in each assistant to load the skills.")
    return 1 if FAILED in outcomes else 0


if __name__ == "__main__":  # `python -m mcp_server.skills_cli`, which the npm wrapper runs
    raise SystemExit(main())
