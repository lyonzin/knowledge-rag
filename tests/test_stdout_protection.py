"""Protect the real MCP pipe during startup and background diagnostics."""

import json
import subprocess
import sys

from support.mcp_stdio import ROOT, McpSession, isolated_env, stdio_server

from mcp_server import __version__

PROBE_SERVER = ROOT / "tests" / "support" / "stdio_probe_server.py"


def test_import_preserves_callers_stdout(tmp_path):
    (tmp_path / "config.yaml").write_text("[]", encoding="utf-8")
    # The re-export is lazy, so ask for it: loading the config prints its warning.
    script = (
        "import sys; original = sys.stdout; from mcp_server import Config; "
        "assert sys.stdout is original; print('caller output')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=isolated_env(tmp_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "caller output\n"
    assert "not a valid mapping" in result.stderr


def test_mcp_initialize_and_background_logs_keep_stdout_json_only(tmp_path):
    with stdio_server(PROBE_SERVER, isolated_env(tmp_path)) as (process, stdout, stderr):
        session = McpSession(process, stdout, "stdout-regression")
        assert session.server_info["version"] == __version__
        assert session.call_text("audit_stdout", {}) == "diagnostics complete"
    assert process.returncode == 0
    for line in list(stdout.queue):
        if line is not None:
            assert json.loads(line).get("jsonrpc") == "2.0"
    diagnostics = "".join(line for line in list(stderr.queue) if line)
    assert "migration progress: 10%" in diagnostics
    assert "Skipping empty file" in diagnostics
    assert "GPU STATUS: UNAVAILABLE" in diagnostics


def test_init_command_keeps_human_output_on_stdout(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mcp_server.server", "init"],
        cwd=tmp_path,
        env=isolated_env(tmp_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Done. Quick start:" in result.stdout
    assert (tmp_path / "config.example.yaml").is_file()
