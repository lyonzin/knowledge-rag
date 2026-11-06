"""Light submodules import without the configuration, which creates data folders as it loads."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The configuration and the server stack; the skills installer needs none of them.
HEAVY = ("mcp_server.config", "mcp_server.ingestion", "mcp_server.server", "chromadb", "fastembed")
PROBE = f"import sys, {{module}}; print(sorted(name for name in {HEAVY!r} if name in sys.modules))"


def _python(arguments: list, cwd: Path, **env: str) -> subprocess.CompletedProcess:
    """Run a fresh interpreter in ``cwd`` with KNOWLEDGE_RAG_DIR unset unless ``env`` sets it."""
    environment = {key: value for key, value in os.environ.items() if key != "KNOWLEDGE_RAG_DIR"}
    environment.update(PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8", **env)
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )


def _project_folder(path: Path) -> Path:
    """A folder with a config.yaml, so a configuration loaded from it would make it the base directory."""
    path.mkdir()
    (path / "config.yaml").write_text("{}\n", encoding="utf-8")
    return path


def test_importing_the_skills_installer_loads_no_configuration_and_creates_nothing(tmp_path):
    control = _project_folder(tmp_path / "control")
    loaded = _python(["-c", PROBE.format(module="mcp_server.config")], control)
    assert loaded.returncode == 0, loaded.stderr
    assert {"data", "documents"} <= {entry.name for entry in control.iterdir()}, "the probe cannot see folders"

    cwd = _project_folder(tmp_path / "cwd")
    result = _python(["-c", PROBE.format(module="mcp_server.skills_cli")], cwd)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
    assert [entry.name for entry in cwd.iterdir()] == ["config.yaml"]


def test_skills_installer_runs_as_a_module_without_creating_folders(tmp_path):
    cwd = _project_folder(tmp_path / "cwd")
    result = _python(["-m", "mcp_server.skills_cli", "status", "--target", str(cwd / "skills")], cwd)
    assert result.returncode == 0, result.stderr
    assert "not installed" in result.stdout
    assert [entry.name for entry in cwd.iterdir()] == ["config.yaml"]


def test_package_reexports_still_resolve(tmp_path):
    script = "\n".join(
        [
            "import sys, mcp_server",
            "assert 'mcp_server.config' not in sys.modules",
            "from mcp_server import Config, Document, DocumentParser",
            "from mcp_server.config import Config as RealConfig",
            "from mcp_server.ingestion import Document as RealDocument, DocumentParser as RealParser",
            "assert (Config, Document, DocumentParser) == (RealConfig, RealDocument, RealParser)",
            "assert mcp_server.Config is RealConfig",
            "namespace = {}",
            "exec('from mcp_server import *', namespace)",
            "assert {name: namespace[name] for name in mcp_server.__all__} == {",
            "    'Config': RealConfig, 'Document': RealDocument, 'DocumentParser': RealParser}",
            "assert not hasattr(mcp_server, 'missing')",
            "print('ok')",
        ]
    )
    result = _python(["-c", script], tmp_path, KNOWLEDGE_RAG_DIR=str(tmp_path / "base"))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
