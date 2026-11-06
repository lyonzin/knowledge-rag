"""Source checkout and wheel configuration export the same preset content."""

import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("preset", sorted((ROOT / "presets").glob("*.yaml")), ids=lambda path: path.stem)
def test_every_preset_is_bundled_and_force_included(preset):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    relative = preset.relative_to(ROOT).as_posix()
    bundled = f"mcp_server/data/{preset.name}"
    assert includes[relative] == bundled
    assert (ROOT / bundled).read_text(encoding="utf-8") == preset.read_text(encoding="utf-8")


def test_bundled_config_matches_source_template():
    assert (ROOT / "mcp_server/data/config.example.yaml").read_text(encoding="utf-8") == (
        ROOT / "config.example.yaml"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("from_sdist", [False, True], ids=["direct-wheel", "sdist-wheel"])
def test_built_wheel_contains_each_canonical_resource_once(tmp_path, from_sdist):
    from hatchling.builders.sdist import SdistBuilder
    from hatchling.builders.wheel import WheelBuilder

    expected = {"config.example.yaml": (ROOT / "config.example.yaml").read_bytes()}
    expected.update({preset.name: preset.read_bytes() for preset in (ROOT / "presets").glob("*.yaml")})
    # `knowledge-rag skills install` reads these from the installed package.
    skills = (ROOT / "skills").glob("*/*/SKILL.md")
    expected.update({f"skills/{skill.relative_to(ROOT / 'skills').as_posix()}": skill.read_bytes() for skill in skills})
    source = ROOT
    if from_sdist:
        sdist = next(SdistBuilder(str(ROOT)).build(directory=str(tmp_path / "sdist")))
        with tarfile.open(sdist) as archive:
            archive.extractall(tmp_path / "unpacked", filter="data")
        source = next((tmp_path / "unpacked").iterdir())

    wheel = next(WheelBuilder(str(source)).build(directory=str(tmp_path / "wheel"), versions=["standard"]))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "Wheel archive contains duplicate paths"
        resources = {name for name in names if name.startswith("mcp_server/data/")}
        assert resources == {f"mcp_server/data/{name}" for name in expected}
        for name, content in expected.items():
            assert archive.read(f"mcp_server/data/{name}") == content
        entry_points = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        assert "knowledge-rag-skills = mcp_server.skills_cli:main" in archive.read(entry_points).decode("utf-8")


def test_extracted_wheel_provides_the_skills_for_the_installer(tmp_path):
    from hatchling.builders.wheel import WheelBuilder

    wheel = next(WheelBuilder(str(ROOT)).build(directory=str(tmp_path / "wheel"), versions=["standard"]))
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(tmp_path / "site")
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "site"), KNOWLEDGE_RAG_DIR=str(tmp_path / "base"))
    result = subprocess.run(
        [sys.executable, "-c", "from mcp_server.skills_bundle import bundled_skills; print(len(bundled_skills()))"],
        cwd=tmp_path / "site",
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "10"


def test_skill_sources_keep_lf_line_endings_on_any_checkout():
    """The installer hashes what it installs; CRLF conversions would make manifests differ per checkout."""
    assert b"\r" not in (ROOT / "skills" / "install.sh").read_bytes()


def test_gitignore_keeps_bundled_presets_versionable_and_runtime_data_private(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is needed to validate checkout ignore rules")
    subprocess.run([git, "init", "--quiet", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
    resources = [f"mcp_server/data/{preset.name}" for preset in (ROOT / "presets").glob("*.yaml")]
    resources.append("mcp_server/data/config.example.yaml")
    runtime = "data/chroma_db/chroma.sqlite3"
    for name in [*resources, runtime]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch()

    ignored_resources = subprocess.run(
        [git, "check-ignore", "--no-index", *resources], cwd=tmp_path, capture_output=True, text=True
    )
    assert ignored_resources.returncode == 1, ignored_resources.stdout + ignored_resources.stderr
    ignored_runtime = subprocess.run(
        [git, "check-ignore", "--no-index", runtime], cwd=tmp_path, capture_output=True, text=True
    )
    assert ignored_runtime.returncode == 0, ignored_runtime.stderr
    assert ignored_runtime.stdout.strip() == runtime
