"""The bundled skills and the files they occupy in one assistant's skills folder.

Each skills folder gets a ``.knowledge-rag-skills.json`` manifest with the
SHA-256 of every ``SKILL.md`` installed from here. A copy is replaced only
while it still matches the manifest, so a skill the user edited is kept, and
removal deletes only unedited folders that hold nothing but their ``SKILL.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .file_transaction import replace_with_retry

SKILL_FILE = "SKILL.md"
MANIFEST_FILE = ".knowledge-rag-skills.json"

# State of one bundled skill inside one skills folder. Outdated copies still
# match what was installed, so replacing them loses nothing; edited ones
# changed since; untracked ones differ and never came from here (another
# installer, a manual copy); foreign entries are not plain folders. Only
# missing and outdated copies are ever written.
MISSING, CURRENT, OUTDATED = "missing", "current", "outdated"
EDITED, UNTRACKED, FOREIGN = "edited", "untracked", "foreign"

_MANIFEST_NOTE = "Written by `knowledge-rag skills`: SHA-256 of each SKILL.md it installed; updates replace only copies that still match."


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
    """Skill name to SHA-256 of the copy installed from here; empty when absent or unreadable."""
    try:
        document = json.loads((folder / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = document.get("skills") if isinstance(document, dict) else None
    if not isinstance(entries, dict):
        return {}
    return {name: digest for name, digest in entries.items() if isinstance(name, str) and isinstance(digest, str)}


def skill_state(folder: Path, skill: Skill, manifest: Mapping[str, str]) -> str:
    """Classify ``skill`` in ``folder``; symlinks and other non-regular entries are foreign."""
    skill_dir = folder / skill.name
    path = skill_dir / SKILL_FILE
    if skill_dir.is_symlink() or path.is_symlink() or (skill_dir.exists() and not skill_dir.is_dir()):
        return FOREIGN
    if not path.exists():
        return MISSING
    if not path.is_file():
        return FOREIGN
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    recorded = manifest.get(skill.name)
    if digest == skill.digest:
        return CURRENT
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


def install(plan: Plan, skills: Sequence[Skill]) -> None:
    """Write missing and outdated skills; the manifest records each write even if a later one fails."""
    try:
        for skill in skills:
            state = plan.states[skill.name]
            if state in (MISSING, OUTDATED):
                _write_bytes(plan.folder / skill.name / SKILL_FILE, skill.content)
            if state in (MISSING, OUTDATED, CURRENT):
                plan.manifest[skill.name] = skill.digest
    finally:
        _write_manifest(plan)


def uninstall(plan: Plan, skills: Sequence[Skill]) -> list[str]:
    """Delete unedited skill folders; return the ones kept because they hold other files."""
    kept = []
    for skill in skills:
        state = plan.states[skill.name]
        if state == MISSING:
            plan.manifest.pop(skill.name, None)
        if state not in (CURRENT, OUTDATED):
            continue
        skill_dir = plan.folder / skill.name
        if [entry.name for entry in skill_dir.iterdir()] != [SKILL_FILE]:
            kept.append(skill.name)
            continue
        (skill_dir / SKILL_FILE).unlink()
        skill_dir.rmdir()
        plan.manifest.pop(skill.name, None)
    _write_manifest(plan)
    return kept


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
            stream.write(data)
        os.chmod(temporary, 0o644)  # mkstemp creates 0600; skills are ordinary readable files
        replace_with_retry(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
