"""The bundled skills and the files they occupy in one assistant's skills folder.

Each skills folder gets a ``.knowledge-rag-skills.json`` manifest with the
SHA-256 of every ``SKILL.md`` installed from here. A copy is replaced only
while it still matches the manifest, so a skill the user edited is kept, and
removal deletes only folders that hold nothing but their ``SKILL.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .file_transaction import replace_with_retry

SKILL_FILE = "SKILL.md"
MANIFEST_FILE = ".knowledge-rag-skills.json"
_MODE_0644 = 0o644

# State of one bundled skill inside one skills folder. OUTDATED copies still match what was
# installed, so replacing them loses nothing; EDITED ones changed since; UNTRACKED ones differ and
# never came from here; MANUAL ones match the bundle but have no manifest record; FOREIGN entries
# are links or not plain folders. Only missing, outdated and manual copies are ever touched.
MISSING, CURRENT, OUTDATED = "missing", "current", "outdated"
EDITED, UNTRACKED, MANUAL, FOREIGN = "edited", "untracked", "manual", "foreign"

# Files that do not stop a skill folder from being removable: our temp files and OS metadata.
_IGNORED_ENTRIES = (".DS_Store", "Thumbs.db", "desktop.ini")

_MANIFEST_NOTE = "Written by `knowledge-rag skills`: SHA-256 of each SKILL.md it installed; updates replace only copies that still match."


class SkillWriteError(OSError):
    """A skill write or delete failed; the message carries the per-skill accounting."""


@dataclass(frozen=True)
class Skill:
    """One bundled skill: its folder name and the bytes of its SKILL.md."""

    name: str
    content: bytes

    @property
    def digest(self) -> str:
        """SHA-256 of the bundled SKILL.md."""
        return hashlib.sha256(self.content).hexdigest()


@dataclass
class Plan:
    """One skills folder: the state of each selected skill and the folder's manifest."""

    folder: Path
    states: dict[str, str]
    manifest: dict[str, str]


def bundled_skills() -> list[Skill]:
    """The skills shipped with knowledge-rag, sorted by name.

    Wheels carry them in ``mcp_server/data/skills``; a source checkout reads
    the repository's ``skills/`` folder.
    """
    package = Path(__file__).resolve().parent
    for root in (package / "data" / "skills", package.parent / "skills"):
        found = [Skill(path.parent.name, path.read_bytes()) for path in root.glob(f"*/*/{SKILL_FILE}")]
        if found:
            return sorted(found, key=lambda skill: skill.name)
    raise FileNotFoundError(f"no bundled skills in {package / 'data' / 'skills'}; reinstall knowledge-rag")


def read_manifest(folder: Path) -> dict[str, str]:
    """Skill name to SHA-256 of the copy installed from here; empty when the file is absent.

    A present file that is not a plain ``{"skills": {name: digest}}`` object raises an OSError:
    nothing must ever be written over a manifest the installer cannot read.
    """
    path = folder / MANIFEST_FILE
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise OSError(f"cannot read {path}: {exc}") from exc
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise OSError(f"{path} is not valid JSON; fix or delete the file ({exc})") from exc
    entries = document.get("skills") if isinstance(document, dict) else None
    if not isinstance(entries, dict):
        raise OSError(f"{path} does not have a 'skills' object; fix or delete the file")
    for name, digest in entries.items():
        if not (isinstance(name, str) and isinstance(digest, str)):
            raise OSError(f"{path} has an entry that is not a name/digest pair; fix or delete the file")
    return dict(entries)


def _is_link(path: Path) -> bool:
    """True for symlinks and, on Windows, for junctions; OneDrive placeholders stay normal files."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    if os.name == "nt":
        tag = getattr(info, "st_reparse_tag", 0)
        return tag in (
            getattr(stat, "IO_REPARSE_TAG_SYMLINK", -1),
            getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", -2),
        )
    return False


def skill_state(folder: Path, skill: Skill, manifest: Mapping[str, str]) -> str:
    """Classify ``skill`` in ``folder``; links and other non-regular entries are foreign.

    CURRENT and OUTDATED both mean the manifest records the copy's digest, so they are
    the only states uninstall may delete; MANUAL means an identical copy nobody recorded.
    """
    skill_dir = folder / skill.name
    path = skill_dir / SKILL_FILE
    if _is_link(skill_dir) or _is_link(path) or (skill_dir.exists() and not skill_dir.is_dir()):
        return FOREIGN
    if not path.exists():
        return MISSING
    if not path.is_file():
        return FOREIGN
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    recorded = manifest.get(skill.name)
    if digest == skill.digest:
        return CURRENT if recorded == digest else MANUAL
    if recorded is None:
        return UNTRACKED
    return OUTDATED if recorded == digest else EDITED


def plan_folder(folder: Path, skills: Sequence[Skill]) -> Plan:
    """Read the manifest and classify every skill; OSError if the folder cannot be read."""
    manifest = read_manifest(folder)
    return Plan(folder, {skill.name: skill_state(folder, skill, manifest) for skill in skills}, manifest)


def holds_skills(folder: Path, skills: Sequence[Skill]) -> bool:
    """True when ``folder`` has a manifest or a folder named after one of ``skills``."""
    return (folder / MANIFEST_FILE).is_file() or any((folder / skill.name / SKILL_FILE).exists() for skill in skills)


def install(plan: Plan, skills: Sequence[Skill]) -> list[str]:
    """Write missing and outdated skills and record manual copies; return the names written.

    The manifest records each change even when a later write fails; a failed write raises
    SkillWriteError with the per-skill accounting.
    """
    written: list[str] = []
    pending = [skill.name for skill in skills]
    try:
        for skill in skills:
            pending.remove(skill.name)
            # Re-check right before the write: the folder may have changed since the plan.
            state = skill_state(plan.folder, skill, plan.manifest)
            if state in (MISSING, OUTDATED):
                _write_bytes(plan.folder / skill.name / SKILL_FILE, skill.content)
                written.append(skill.name)
            if state in (MISSING, OUTDATED, CURRENT, MANUAL):
                plan.manifest[skill.name] = skill.digest
    except OSError as exc:
        raise SkillWriteError(
            f"failed after {len(written)} of {len(skills)} "
            f"(written: {', '.join(written) or 'none'}; not attempted: {', '.join(pending) or 'none'}): {exc}"
        ) from exc
    finally:
        _write_manifest_safely(plan)
    return written


def uninstall(plan: Plan, skills: Sequence[Skill]) -> tuple[list[str], list[str], list[str]]:
    """Delete unedited skill folders the manifest owns; return kept, orphans removed and skills removed.

    Orphaned entries the bundle no longer ships are removed too, and their records. The manifest
    is written even when a delete fails, which then raises SkillWriteError.
    """
    bundled = {skill.name for skill in skills}
    kept: list[str] = []
    orphans: list[str] = []
    removed: list[str] = []
    try:
        for name in list(plan.manifest):
            if name in bundled:
                continue
            if name != Path(name).name:  # a record this installer cannot have made
                plan.manifest.pop(name, None)
                continue
            skill_dir = plan.folder / name
            if not skill_dir.is_dir():
                plan.manifest.pop(name, None)
                continue
            if _folder_only_holds_skill_md(skill_dir):
                _delete_skill_folder(skill_dir)
                orphans.append(name)
            plan.manifest.pop(name, None)
        for skill in skills:
            # Re-check right before the delete: the folder may have changed since the plan.
            state = skill_state(plan.folder, skill, plan.manifest)
            if state == MISSING:
                plan.manifest.pop(skill.name, None)
            if state not in (CURRENT, OUTDATED):
                continue
            skill_dir = plan.folder / skill.name
            if not _folder_only_holds_skill_md(skill_dir):
                kept.append(skill.name)
                continue
            _delete_skill_folder(skill_dir)
            plan.manifest.pop(skill.name, None)
            removed.append(skill.name)
    except OSError as exc:
        raise SkillWriteError(f"failed during uninstall: {exc}") from exc
    finally:
        _write_manifest_safely(plan)
    return kept, orphans, removed


def _delete_skill_folder(skill_dir: Path) -> None:
    (skill_dir / SKILL_FILE).unlink()
    skill_dir.rmdir()


def _folder_only_holds_skill_md(skill_dir: Path) -> bool:
    """True when the folder has nothing but its SKILL.md, temp files and OS metadata files."""
    for entry in skill_dir.iterdir():
        if entry.name == SKILL_FILE:
            continue
        if entry.name in _IGNORED_ENTRIES or entry.name.startswith(".knowledge-rag-"):
            continue
        return False
    return True


def _write_manifest_safely(plan: Plan) -> None:
    """Write the manifest; if it fails during another error, annotate that error instead of masking it."""
    outer = sys.exc_info()[1]  # the exception propagating through the caller's finally, if any
    try:
        _write_manifest(plan)
    except OSError as exc:
        if outer is not None and isinstance(outer, BaseException):
            outer.add_note(f"the manifest {plan.folder / MANIFEST_FILE} could not be written either: {exc}")
        else:
            raise


def _write_manifest(plan: Plan) -> None:
    path = plan.folder / MANIFEST_FILE
    if not plan.manifest:
        path.unlink(missing_ok=True)
        return
    document = {"about": _MANIFEST_NOTE, "skills": dict(sorted(plan.manifest.items()))}
    _write_bytes(path, (json.dumps(document, indent=2) + "\n").encode("utf-8"))


def _write_bytes(path: Path, data: bytes) -> None:
    """Write beside the destination and rename over it, so no reader sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".knowledge-rag-")
    try:
        with os.fdopen(handle, "wb") as stream:
            if hasattr(os, "fchmod"):
                try:
                    os.fchmod(stream.fileno(), _MODE_0644)  # mkstemp creates 0600
                except OSError:
                    pass
            stream.write(data)
        replace_with_retry(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
