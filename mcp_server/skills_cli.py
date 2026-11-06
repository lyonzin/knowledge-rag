"""``knowledge-rag skills install|uninstall|status``: put the bundled skills where assistants load them.

Without ``--harness``, ``--target`` or ``--project`` it acts on the skills folder of every assistant it
detects, plus any folder that already holds the skills, and asks once per folder. When stdin is not a
terminal it prints the plan and changes nothing unless ``--yes`` is given.

Example::

    knowledge-rag skills install --dry-run
    knowledge-rag skills install --harness claude-code --yes
    knowledge-rag skills status
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from . import skills_bundle as bundle
from .assistants import HARNESSES, Home, detect, skills_folder, user_homes

# Outcome of one folder.
DONE, DECLINED, NEEDS_YES, FAILED = "done", "declined", "needs-yes", "failed"

_EXIT_INTERRUPTED = 130  # what Ctrl+C at a prompt returns
_MANUAL_SETUP = "https://github.com/lyonzin/knowledge-rag/blob/master/skills/README.md#other-assistants"
_WORDS = {
    "install": {
        bundle.MISSING: "to install",
        bundle.OUTDATED: "to update",
        bundle.CURRENT: "up to date",
        bundle.MANUAL: "identical copy, will be recorded (manual install)",
        bundle.EDITED: "edited (kept)",
        bundle.UNTRACKED: "different copy not installed from here (kept)",
        bundle.FOREIGN: "not a plain folder (left alone)",
    },
    "uninstall": {
        bundle.MISSING: "not installed",
        bundle.OUTDATED: "to remove",
        bundle.CURRENT: "to remove",
        bundle.MANUAL: "identical copy, no record of installing it (kept)",
        bundle.EDITED: "edited (kept)",
        bundle.UNTRACKED: "different copy not installed from here (kept)",
        bundle.FOREIGN: "not a plain folder (left alone)",
    },
    "status": {
        bundle.MISSING: "not installed",
        bundle.OUTDATED: "older version (install updates it)",
        bundle.CURRENT: "up to date",
        bundle.MANUAL: "identical copy, no record of installing it",
        bundle.EDITED: "edited since installed",
        bundle.UNTRACKED: "different copy not installed from here",
        bundle.FOREIGN: "not a plain folder",
    },
}
_KEPT = (bundle.EDITED, bundle.UNTRACKED, bundle.FOREIGN)  # install never touches these
_NAMED = (bundle.MANUAL, bundle.EDITED, bundle.UNTRACKED, bundle.FOREIGN)  # listed by name in summaries

Targets = dict[Path, list[str]]  # skills folder -> the assistants (labels) that read it
Result = tuple[str, int, int]  # outcome, skills applied, skills kept


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


def _local_home() -> Home:
    """The current profile; explicit targets must not shell out to the Windows side."""
    return Home(Path.home(), os.environ, shutil.which)


def _explicit_targets(
    parser: argparse.ArgumentParser, args: argparse.Namespace, home: Home
) -> tuple[Targets, list[str]]:
    """Folders given with --target or --project, plus the folders of the --harness assistants."""
    folders: Targets = {}
    for raw in args.target or []:
        expanded = os.path.expanduser(os.path.expandvars(raw)).strip()
        if not expanded:
            parser.error("--target needs a non-empty path")
        folders.setdefault(Path(os.path.abspath(expanded)), [])
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


def _summary(command: str, plan: bundle.Plan) -> str:
    """Count the skills per state, naming the ones the user should look at."""
    lines = []
    for state, words in _WORDS[command].items():
        names = [name for name, value in plan.states.items() if value == state]
        if names:
            lines.append(f"  {len(names)} {words}" + (f": {', '.join(names)}" if state in _NAMED else ""))
    return "\n".join(lines)


def _has_work(command: str, plan: bundle.Plan, bundled: set[str]) -> bool:
    if command == "install":
        return any(state in (bundle.MISSING, bundle.OUTDATED, bundle.MANUAL) for state in plan.states.values())
    return any(state in (bundle.CURRENT, bundle.OUTDATED) for state in plan.states.values()) or any(
        name not in bundled for name in plan.manifest
    )


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


def _flush_terminal_input() -> None:
    """POSIX: drop keystrokes typed before the question, so they cannot answer it."""
    if sys.platform == "win32":  # termios does not exist there; mypy narrows this branch per --platform
        return
    else:
        try:
            import termios

            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except (ImportError, OSError, ValueError):
            pass


def _confirm(command: str, folder: Path, labels: Sequence[str]) -> bool | None:
    """Ask yes or no; None when the answer was neither. Install defaults to yes, removal to no."""
    verb = "Install" if command == "install" else "Remove"
    where = f"for {', '.join(labels)}" if labels else f"in {folder}"
    hint = "[Y/n]" if command == "install" else "[y/N]"
    _flush_terminal_input()
    try:
        answer = input(f"{verb} the knowledge-rag skills {where}? {hint} ")
    except KeyboardInterrupt:  # Ctrl+C stops the whole run, not just this folder
        print()
        raise SystemExit(_EXIT_INTERRUPTED) from None
    lowered = answer.strip().lower()
    if not lowered:
        return command == "install"
    if lowered in ("y", "yes"):
        return True
    if lowered in ("n", "no"):
        return False
    return None


def _apply(command: str, plan: bundle.Plan, skills: Sequence[bundle.Skill]) -> Result:
    """Change one folder, reporting an OS error instead of stopping at it."""
    kept = sum(state in _KEPT for state in plan.states.values())
    try:
        if command == "install":
            written = bundle.install(plan, skills)
            adopted = sum(plan.states[skill.name] == bundle.MANUAL for skill in skills)
            print("  done")
            return DONE, len(written) + adopted, kept
        kept_list, orphans, removed = bundle.uninstall(plan, skills)
        notes = []
        if kept_list:
            notes.append(f"kept, because they hold other files: {', '.join(kept_list)}")
        if orphans:
            notes.append(f"removed orphans no longer in the bundle: {', '.join(orphans)}")
        print("  done" + (f"; {'; '.join(notes)}" if notes else ""))
        return DONE, len(removed) + len(orphans), kept
    except OSError as exc:
        print(f"  failed: {exc}")
        return FAILED, 0, kept


def _handle(args: argparse.Namespace, folder: Path, labels: Sequence[str], skills: Sequence[bundle.Skill]) -> Result:
    """Report one folder, then apply the command to it once confirmed."""
    folder = Path(os.path.realpath(folder))  # the prompt must show the resolved destination
    print(f"\n{', '.join(labels) or 'Skills folder'}: {folder}")
    try:
        plan = bundle.plan_folder(folder, skills)
    except OSError as exc:
        print(f"  cannot read it: {exc}")
        return FAILED, 0, 0
    print(_summary(args.command, plan))
    if args.command == "install" and not folder.exists():
        print("  folder does not exist; it will be created")
    if args.command == "install" and any(state in _KEPT for state in plan.states.values()):
        print("  To install this version over a kept skill, delete its folder first.")
    kept = sum(state in _KEPT for state in plan.states.values())
    bundled = {skill.name for skill in skills}
    if args.command == "status" or args.dry_run or not _has_work(args.command, plan, bundled):
        return DONE, 0, kept
    if not args.yes and not _stdin_is_terminal():
        return NEEDS_YES, 0, 0
    if not args.yes:
        try:
            answer = _confirm(args.command, folder, labels)
        except EOFError:
            print("  Nothing was changed.")
            return DECLINED, 0, 0
        if answer is None:
            print("  skipped")
            return DECLINED, 0, 0
        if not answer:
            print("  Nothing was changed.")
            return DECLINED, 0, 0
    return _apply(args.command, plan, skills)


def _select(
    parser: argparse.ArgumentParser, skills: list[bundle.Skill], requested: list[str] | None
) -> list[bundle.Skill]:
    if not requested:
        return skills
    entries = [name for value in requested for name in value.split(",")]
    if any(not name.strip() for name in entries):
        parser.error("--skill needs skill names; found an empty one")
    wanted = {name.strip() for name in entries}
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


def _reconfigure_streams() -> None:
    """Keep going when a folder name cannot be encoded in the output stream's encoding."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (OSError, ValueError):
            pass


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``knowledge-rag skills``; returns the process exit code."""
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "skills":  # the npm wrapper runs `-m mcp_server.skills_cli skills ...`
        argv = argv[1:]
    _reconfigure_streams()
    args = parser.parse_args(argv)
    try:
        skills = _select(parser, bundle.bundled_skills(), args.skill)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    explicit = bool(args.target or args.harness or args.project)
    if explicit:
        folders, notes = _explicit_targets(parser, args, _local_home())
    else:
        folders, notes = _detected_targets(user_homes(), skills)
    for note in notes:
        print(note)
    if not folders:
        print("No skills folder to act on. Name an assistant with --harness or a folder with --target.")
        return 1 if args.command == "install" else 0
    results = [_handle(args, folder, labels, skills) for folder, labels in folders.items()]
    return _finish(args, results)


def _finish(args: argparse.Namespace, results: Sequence[Result]) -> int:
    outcomes = [result[0] for result in results]
    if NEEDS_YES in outcomes:
        print(f"\nNothing was changed: stdin is not a terminal. Re-run with --yes to {args.command} as planned above.")
        return 1
    if args.dry_run:
        print("\nDry run: nothing was written.")
        return 1 if FAILED in outcomes else 0
    applied = sum(result[1] for result in results)
    kept = sum(result[2] for result in results)
    if args.command == "install":
        if applied == 0 and kept:
            print(f"\nNothing installed: {kept} skills kept")
        elif applied > 0:
            print("\nStart a new session in each assistant to load the skills.")
    return 1 if FAILED in outcomes else 0


if __name__ == "__main__":  # `python -m mcp_server.skills_cli`, which the npm wrapper runs
    raise SystemExit(main())
