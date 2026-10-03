"""Source checkout and wheel configuration export the same preset content."""

import shutil
import subprocess
import tomllib
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
