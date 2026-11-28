"""Small stand-ins for the guarded probe. They use the parent's explicit target, never config."""

import subprocess

RELEASE_ENV = "KNOWLEDGE_RAG_TEST_PROBE_RELEASE"
INSIDE = "probe-inside"
LEAVING = "probe-leaving"

BODY = r"""
import os
import time


def _note(entry):
    with (target.data_dir / "writes.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{os.getpid()} {entry}\n")


_note("probe-inside")
_release = os.environ.get("KNOWLEDGE_RAG_TEST_PROBE_RELEASE", "")
_deadline = time.monotonic() + 60
while _release and not (target.data_dir / _release).exists():
    if time.monotonic() > _deadline:
        raise SystemExit("the test never released this probe")
    time.sleep(0.05)
_note("probe-leaving")
print("counting\n1", flush=True)
"""


def reply(target, code, *events, stderr=""):
    """Child transcript for unit tests that exercise the decision, not Chroma itself."""
    return subprocess.CompletedProcess([], code, "\n".join((target.wire(), *events)) + "\n", stderr)


def native_crash(target, timeout_seconds=30):
    return reply(target, -11, "counting", stderr="segfault")


def healthy(target, timeout_seconds=30):
    return reply(target, 0, "counting", "1")


def busy(target, timeout_seconds=30):
    return reply(target, 75, "busy")
