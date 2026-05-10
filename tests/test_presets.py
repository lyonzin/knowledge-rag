"""Validate every preset YAML in presets/ loads cleanly.

Pillar 4: Versatility — guarantee the 4 shipped configuration profiles
(``cybersecurity``, ``developer``, ``research``, ``general``) remain valid
and structurally complete after every PR. A regression here breaks
freshly installed users who pick a preset out of the box.

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

# Embedding model promise — all presets must use the same dim so the
# index format is portable across presets without rebuild.
EXPECTED_EMBEDDING_DIM = 384


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
    """``models.embedding`` must declare a model name and a 384D dimension."""
    data = yaml.safe_load(preset_yaml.read_text(encoding="utf-8"))
    embedding = data.get("models", {}).get("embedding", {})
    assert isinstance(embedding, dict), f"{preset_yaml.name}: models.embedding must be a mapping"
    assert "model" in embedding, f"{preset_yaml.name}: models.embedding.model is required"
    assert embedding.get("dimensions", EXPECTED_EMBEDDING_DIM) == EXPECTED_EMBEDDING_DIM, (
        f"{preset_yaml.name}: dimensions must remain {EXPECTED_EMBEDDING_DIM} so indices stay portable"
    )


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


def test_at_least_four_presets_shipped():
    """Releasing the project without all four named presets is a regression."""
    expected = {"cybersecurity", "developer", "research", "general"}
    actual = {p.stem for p in PRESETS_DIR.glob("*.yaml")}
    missing = expected - actual
    assert not missing, f"Missing presets: {missing}"
