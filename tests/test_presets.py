"""Validate every preset YAML in presets/ loads cleanly.

Pillar 4: Versatility — guarantee the 5 shipped configuration profiles
(``cybersecurity``, ``developer``, ``research``, ``general``, ``multilingual``)
remain valid and structurally complete after every PR. A regression here
breaks freshly installed users who pick a preset out of the box.

We validate the YAML structurally rather than instantiating ``Config``
directly because Config has runtime side effects (path resolution,
data_dir creation) that depend on the host filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PRESETS_DIR = Path(__file__).parent.parent / "presets"

# Required top-level sections every preset must declare for compatibility
REQUIRED_TOP_LEVEL = {"models", "documents"}

# Dim expectations per named profile (v4.8.0). Presets that declare a
# profile are validated against the profile's dim; presets without a
# profile default to the historical 384D shipped by ``compact``.
_PROFILE_DIM = {
    "compact": 384,
    "quality": 1024,
    "multilingual": 1024,
}


@pytest.fixture(params=sorted(PRESETS_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def preset_yaml(request):
    """Yield each preset YAML path so each preset gets its own test run."""
    return request.param


def test_preset_yaml_parses(preset_yaml):
    """The YAML must parse without raising and produce a top-level mapping."""
    data = yaml.safe_load(preset_yaml.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{preset_yaml.name}: top-level must be a mapping"


def test_preset_has_required_top_level_sections(preset_yaml):
    """Every preset must declare the contract sections."""
    data = yaml.safe_load(preset_yaml.read_text(encoding="utf-8"))
    missing = REQUIRED_TOP_LEVEL - data.keys()
    assert not missing, f"{preset_yaml.name}: missing required sections {missing}"


def test_preset_embedding_section_intact(preset_yaml):
    """``models.embedding`` must resolve to a valid model + positive dimension.

    Two shapes are accepted (v4.8.0+):
      * Legacy: explicit ``model`` + ``dimensions`` keys (all pre-4.8 presets).
      * Profile-driven: ``profile: "<name>"`` maps to a known preset, in which
        case ``model`` and ``dimensions`` are populated by the resolver at
        Config load time and MAY be omitted from the YAML.
    When both a profile and an explicit dim are declared, they must agree
    with the profile's canonical dim so the index stays portable.
    """
    data = yaml.safe_load(preset_yaml.read_text(encoding="utf-8"))
    embedding = data.get("models", {}).get("embedding", {})
    assert isinstance(embedding, dict), f"{preset_yaml.name}: models.embedding must be a mapping"

    profile = embedding.get("profile", "custom")
    has_model = "model" in embedding
    has_dim = "dimensions" in embedding

    if profile == "custom":
        # Legacy shape — model + dim required
        assert has_model, f"{preset_yaml.name}: models.embedding.model is required for custom profile"
        dim = embedding.get("dimensions", 384)
        assert isinstance(dim, int) and dim > 0, f"{preset_yaml.name}: dimensions must be a positive int, got {dim!r}"
    else:
        # Profile-driven — model + dim may be omitted (resolver fills them).
        # If the profile is unknown, that's a hard error.
        assert profile in _PROFILE_DIM, (
            f"{preset_yaml.name}: unknown embedding profile '{profile}' — "
            f"expected one of {sorted(_PROFILE_DIM)} or 'custom'"
        )
        expected_dim = _PROFILE_DIM[profile]
        if has_dim:
            assert embedding["dimensions"] == expected_dim, (
                f"{preset_yaml.name}: declared dimensions={embedding['dimensions']} "
                f"disagrees with profile '{profile}' canonical dim={expected_dim}"
            )
        # Explicit model coexisting with a non-custom profile is allowed at
        # the YAML layer (resolver logs a WARN and profile wins) — no assert.


def test_preset_reranker_declared_when_section_present(preset_yaml):
    """If a preset opts to declare reranker config, it must include a model name."""
    data = yaml.safe_load(preset_yaml.read_text(encoding="utf-8"))
    reranker = data.get("models", {}).get("reranker")
    if reranker is None:
        pytest.skip(f"{preset_yaml.name} does not declare a reranker section")
    assert isinstance(reranker, dict), f"{preset_yaml.name}: models.reranker must be a mapping"
    assert "model" in reranker, f"{preset_yaml.name}: models.reranker.model is required when section is declared"


def test_preset_chunking_within_sane_bounds(preset_yaml):
    """If chunking is declared, sizes must fit production-realistic ranges."""
    data = yaml.safe_load(preset_yaml.read_text(encoding="utf-8"))
    chunking = data.get("documents", {}).get("chunking")
    if chunking is None:
        pytest.skip(f"{preset_yaml.name} relies on default chunking")
    chunk_size = chunking.get("chunk_size", 1000)
    chunk_overlap = chunking.get("chunk_overlap", 200)
    assert 200 <= chunk_size <= 4000, f"{preset_yaml.name}: chunk_size {chunk_size} outside sane range"
    assert 0 <= chunk_overlap < chunk_size, (
        f"{preset_yaml.name}: chunk_overlap {chunk_overlap} must be < chunk_size {chunk_size}"
    )


def test_at_least_five_presets_shipped():
    """Releasing the project without all five named presets is a regression.

    ``multilingual`` was added in v4.8.0 Fase 1 to back the
    ``intfloat/multilingual-e5-large`` embedding profile.
    """
    expected = {"cybersecurity", "developer", "research", "general", "multilingual"}
    actual = {p.stem for p in PRESETS_DIR.glob("*.yaml")}
    missing = expected - actual
    assert not missing, f"Missing presets: {missing}"
