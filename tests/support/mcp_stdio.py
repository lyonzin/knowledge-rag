"""Drive a real MCP stdio server over JSON-RPC from a test, with no model or user data."""

import json
import os
import queue
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "2025-11-25"
RESPONSE_TIMEOUT_SECONDS = 30
EXIT_TIMEOUT_SECONDS = 15
KILL_TIMEOUT_SECONDS = 5
READER_JOIN_SECONDS = 2


def isolated_env(base_dir, **extra):
    """Environment for a child server: its own base dir, no watcher, no model download."""
    return {
        **os.environ,
        "KNOWLEDGE_RAG_DIR": str(base_dir),
        "KNOWLEDGE_RAG_SINGLE_INSTANCE": "0",
        "KNOWLEDGE_RAG_WATCHER_DISABLED": "1",
        "HF_HUB_OFFLINE": "1",
        "PYTHONPATH": str(ROOT),
        "PYTHONIOENCODING": "utf-8",
        **extra,
    }


def _copy_lines(stream, destination):
    for line in stream:
        destination.put(line)
    destination.put(None)


@contextmanager
def stdio_server(script, env):
    """Run ``script`` as a stdio MCP server; closing stdin on exit lets it shut down."""
    process = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
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
            process.wait(timeout=EXIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=KILL_TIMEOUT_SECONDS)
        for reader in readers:
            reader.join(timeout=READER_JOIN_SECONDS)
        process.stdout.close()
        process.stderr.close()


def send(process, method, params, request_id=None):
    payload = {"jsonrpc": "2.0", "method": method, "params": params}
    if request_id is not None:
        payload["id"] = request_id
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()


def response(stdout, request_id):
    while True:
        line = stdout.get(timeout=RESPONSE_TIMEOUT_SECONDS)
        assert line is not None, "MCP server exited without responding"
        payload = json.loads(line)  # Any diagnostic/blank line corrupts JSON-RPC.
        assert payload.get("jsonrpc") == "2.0", payload
        if payload.get("id") == request_id:
            return payload


class McpSession:
    """One stdio server after the MCP handshake; each request gets the next id."""

    def __init__(self, process, stdout, client_name):
        self.process, self.stdout, self._last_id = process, stdout, 0
        request_id = self._next_id()
        client = {"name": client_name, "version": "1"}
        send(
            process,
            "initialize",
            {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": client},
            request_id,
        )
        self.server_info = response(stdout, request_id)["result"]["serverInfo"]
        send(process, "notifications/initialized", {})

    def _next_id(self):
        self._last_id += 1
        return self._last_id

    def start_call(self, name, arguments):
        """Send a tool call without waiting, so calls to several servers overlap; returns its id."""
        request_id = self._next_id()
        send(self.process, "tools/call", {"name": name, "arguments": arguments}, request_id)
        return request_id

    def finish_call(self, request_id):
        """The text a started call answered."""
        reply = response(self.stdout, request_id)
        assert "result" in reply and not reply["result"].get("isError"), reply
        return reply["result"]["content"][0]["text"]

    def call_text(self, name, arguments):
        """Call one tool and return the text it answered."""
        return self.finish_call(self.start_call(name, arguments))
