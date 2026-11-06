"""Argument routing of the npm wrapper, with child_process stubbed: no Python, venv, or download."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "npm" / "bin" / "cli.js"
# Preloaded with `node -r`: Python "exists", venv creation and pip are no-ops, and the
# spawn that would start Python prints its arguments instead.
STUB = """
const childProcess = require("child_process");
childProcess.execFileSync = () => "Python 3.12.1";
childProcess.spawn = (command, args) => {
  require("fs").writeSync(1, JSON.stringify(args) + "\\n");
  process.exit(0);
};
"""


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        pytest.param(
            ["skills", "install", "--target", "dir with spaces", "--yes"],
            ["-m", "mcp_server.skills_cli", "install", "--target", "dir with spaces", "--yes"],
            id="skills-skip-the-server",
        ),
        pytest.param(["init"], ["-m", "mcp_server.server", "init"], id="init"),
        pytest.param(["--transport", "sse"], ["-m", "mcp_server.server", "--transport", "sse"], id="transport"),
        pytest.param([], ["-m", "mcp_server.server"], id="no-arguments"),
    ],
)
def test_npm_wrapper_routes_arguments(tmp_path, arguments, expected):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    stub = tmp_path / "stub.js"
    stub.write_text(STUB, encoding="utf-8")
    # The wrapper keeps its venv under the home folder: point it at tmp_path.
    env = dict(os.environ, HOME=str(tmp_path), USERPROFILE=str(tmp_path))
    result = subprocess.run(
        [node, "-r", str(stub), str(CLI), *arguments],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == expected
