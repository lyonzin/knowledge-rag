"""Source checkout and wheel configuration export the same preset content."""

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
