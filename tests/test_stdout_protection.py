"""Protect the real MCP pipe during startup and background diagnostics."""

import json
import os
import queue
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from mcp_server import __version__

ROOT = Path(__file__).resolve().parents[1]


def _isolated_env(tmp_path):
    return {
        **os.environ,
        "KNOWLEDGE_RAG_DIR": str(tmp_path),
        "KNOWLEDGE_RAG_WATCHER_DISABLED": "1",
        "HF_HUB_OFFLINE": "1",
        "PYTHONPATH": str(ROOT),
        "PYTHONIOENCODING": "utf-8",
    }


def test_import_preserves_callers_stdout(tmp_path):
    (tmp_path / "config.yaml").write_text("[]", encoding="utf-8")
    script = (
        "import sys; original = sys.stdout; import mcp_server; assert sys.stdout is original; print('caller output')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=_isolated_env(tmp_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "caller output\n"
    assert "not a valid mapping" in result.stderr


def _copy_lines(stream, destination):
    for line in stream:
        destination.put(line)
    destination.put(None)


@contextmanager
def _stdio_server(tmp_path):
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "tests" / "support" / "stdio_probe_server.py")],
        cwd=ROOT,
        env=_isolated_env(tmp_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    stdout, stderr = queue.Queue(), queue.Queue()
    readers = [
        threading.Thread(target=_copy_lines, args=(stream, target), daemon=True)
        for stream, target in ((process.stdout, stdout), (process.stderr, stderr))
    ]
    for reader in readers:
        reader.start()
    try:
        yield process, stdout, stderr
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        for reader in readers:
            reader.join(timeout=2)
        process.stdout.close()
        process.stderr.close()


def _send(process, method, params, request_id=None):
    payload = {"jsonrpc": "2.0", "method": method, "params": params}
    if request_id is not None:
        payload["id"] = request_id
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()


def _response(stdout, request_id):
    while True:
        line = stdout.get(timeout=30)
        assert line is not None, "MCP server exited without responding"
        payload = json.loads(line)  # Any diagnostic/blank line corrupts JSON-RPC.
        assert payload.get("jsonrpc") == "2.0", payload
        if payload.get("id") == request_id:
            return payload


def test_mcp_initialize_and_background_logs_keep_stdout_json_only(tmp_path):
    with _stdio_server(tmp_path) as (process, stdout, stderr):
        _send(
            process,
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "stdout-regression", "version": "1"},
            },
            1,
        )
        initialized = _response(stdout, 1)
        assert initialized["result"]["serverInfo"]["version"] == __version__
        _send(process, "notifications/initialized", {})
        _send(process, "tools/call", {"name": "audit_stdout", "arguments": {}}, 2)
        reply = _response(stdout, 2)
        assert "result" in reply and not reply["result"].get("isError"), reply
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
        env=_isolated_env(tmp_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Done. Quick start:" in result.stdout
    assert (tmp_path / "config.example.yaml").is_file()
