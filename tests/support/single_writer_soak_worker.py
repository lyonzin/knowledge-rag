"""One process of the single-writer soak (tests/chaos/test_single_writer_soak.py).

Usage: ``python single_writer_soak_worker.py ROLE [CHURN_SECONDS]``; the data
directory comes from KNOWLEDGE_RAG_DIR. Only the process that owns data_dir, and
holds its index access guard, may open Chroma: every role but ``reopen`` logs
each chromadb client it creates to OPENINGS_LOG and fails when it is not that
process. Events go to stdout as JSON lines after EVENT_PREFIX; GO on stdin
starts a churn, EOF ends the role.

- ``owner SECONDS``: claims data_dir and its index access, seeds, churns after
  GO, reports the count.
- ``follower SECONDS``: ``server.main()`` as a stdio session, calling tools
  without pause; the call that takes over waits for the owner startup, checks
  that it serves, then churns after GO.
- ``session``: ``server.main()`` as a short session that calls each tool once.
- ``reopen``: a fresh process that, once everything stopped, claims data_dir
  and its index access as an owner must, then opens the index.
"""

import json
import os
import queue
import sys
import threading
import time
import traceback
import zlib

import chromadb
import numpy as np
from support.tool_calls import TOOL_CALLS, is_refusal

from mcp_server import _index_access, _writer_role
from mcp_server.config import config

EVENT_PREFIX = "SOAK "
GO = "go"
OPENINGS_LOG = "chroma-openings.log"
OWNER_OPENING = "owner"
OTHER_OPENING = "not-owner"
DIMENSIONS = 384  # the production embedding size
SEED_COUNT = 1500  # past the default HNSW sync_threshold (1000): the segment was persisted
BATCH = 100
QUERY_RESULTS = 5
SAMPLE_IDS = 20
TOPICS = 17
STARTUP_TIMEOUT_SECONDS = 120
ERROR_TAIL_CHARS = 3000
REOPEN_ACCESS_WAIT_SECONDS = 5  # everything has stopped: the guard should be free at once
PROBE_EXIT_CODES = []  # one per real preflight probe of this process
TOOL_NAMES = sorted(TOOL_CALLS)


class FakeEmbeddings:
    """Deterministic vectors per text: the server's paths without a model."""

    def __call__(self, input):
        return [_text_vector(text) for text in input]

    def embed_query(self, input):
        return self(input)

    @staticmethod
    def name():
        return "soak-fake"

    @staticmethod
    def is_legacy():
        return True


def _text_vector(text):
    return np.random.default_rng(zlib.crc32(text.encode("utf-8"))).random(DIMENSIONS, dtype=np.float32).tolist()


def _emit(event, **fields):
    print(EVENT_PREFIX + json.dumps({"event": event, **fields}), flush=True)


class _Stdin:
    """GO and EOF from the test, read by one thread: no timing guesses."""

    def __init__(self):
        self.lines, self.closed = queue.Queue(), threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in sys.stdin:
            self.lines.put(line.strip())
        self.closed.set()
        self.lines.put(None)

    def wait_for_go(self):
        if self.lines.get() != GO:
            raise RuntimeError("stdin closed before the test said go")


def _guard_chroma_openings():
    """Only the owner, inside its access guard, may open Chroma: log every client created here, fail any other."""
    real_client = chromadb.PersistentClient

    def opening(*args, **kwargs):
        owner = _writer_role._claimed is not None and _index_access.is_held()
        with (config.data_dir / OPENINGS_LOG).open("a", encoding="utf-8") as log:
            log.write(f"{os.getpid()} {OWNER_OPENING if owner else OTHER_OPENING}\n")
        if not owner:
            raise RuntimeError(f"process {os.getpid()} opened Chroma without owning data_dir")
        return real_client(*args, **kwargs)

    chromadb.PersistentClient = opening


def _records(ids, rng):
    """Vectors, BM25 text and the metadata the server reads, for upsert/add."""
    return {
        "ids": ids,
        "embeddings": rng.random((len(ids), DIMENSIONS), dtype=np.float32).tolist(),
        "documents": [f"soak chunk {chunk} about topic {zlib.crc32(chunk.encode()) % TOPICS}" for chunk in ids],
        "metadatas": [{"source": f"soak/{chunk}.md", "category": "general", "chunk_index": 0} for chunk in ids],
    }


def _churn(collection, rng, seconds):
    """Delete and re-add, add fresh ids, delete half of them, upsert: until the time is up."""
    deadline = time.monotonic() + seconds
    cycle = 0
    while time.monotonic() < deadline:
        start = (cycle * BATCH) % SEED_COUNT
        recycled = [f"seed-{index}" for index in range(start, start + BATCH)]
        collection.delete(ids=recycled)
        collection.add(**_records(recycled, rng))
        fresh = [f"fresh-{os.getpid()}-{cycle}-{index}" for index in range(BATCH)]
        collection.add(**_records(fresh, rng))
        collection.delete(ids=fresh[: BATCH // 2])
        updated = [f"seed-{(start + SEED_COUNT // 2 + index) % SEED_COUNT}" for index in range(BATCH)]
        collection.upsert(**_records(updated, rng))
        cycle += 1
    return cycle


def run_owner(churn_seconds):
    stdin = _Stdin()
    _guard_chroma_openings()
    owns = _writer_role.claim_writes()
    _emit("lock", owns_writes=owns, pid=os.getpid())
    if not owns:
        return
    _index_access.hold()
    client = chromadb.PersistentClient(path=str(config.chroma_dir))
    collection = client.get_or_create_collection(config.collection_name, embedding_function=FakeEmbeddings())
    rng = np.random.default_rng(os.getpid())
    for start in range(0, SEED_COUNT, BATCH):
        collection.upsert(**_records([f"seed-{index}" for index in range(start, start + BATCH)], rng))
    _emit("seeded", count=collection.count())
    stdin.wait_for_go()
    cycles = _churn(collection, rng, float(churn_seconds))
    _emit("done", count=collection.count(), cycles=cycles)
    stdin.closed.wait()  # the test lets go: the process then exits, which frees data_dir


def _record_probes():
    """Keep the exit code of every real preflight probe this process runs, for the take-over report."""
    from mcp_server import preflight

    real_probe = preflight._probe_chroma

    def probe(target, timeout_seconds=preflight.PROBE_TIMEOUT_SECONDS):
        result = real_probe(target, timeout_seconds=timeout_seconds)
        PROBE_EXIT_CODES.append(result.returncode)
        return result

    preflight._probe_chroma = probe


def _run_server(client):
    """server.main() as a stdio session runs it, with ``client`` in place of the transport."""
    from mcp_server import server

    _guard_chroma_openings()
    _record_probes()
    server.FastEmbedEmbeddings = FakeEmbeddings
    server.config.reranker_enabled = False
    server.config.rate_limit_enabled = False
    server._run_transport = lambda transport: client(server)
    server.main()


def _call(server, name):
    return json.loads(getattr(server, name)(**TOOL_CALLS[name]))


def _follow(server, stdin, churn_seconds):
    """Call every tool without pause; each answer is a refusal until one call takes over."""
    _emit("lock", owns_writes=not _writer_role.STATE.is_follower, pid=os.getpid())
    owners, calls = set(), 0
    while not stdin.closed.is_set():
        answer = _call(server, TOOL_NAMES[calls % len(TOOL_NAMES)])
        calls += 1
        if answer["status"] == "repairing":
            return _take_over(server, stdin, churn_seconds)
        if not is_refusal(answer):
            raise RuntimeError(f"a follower answered {answer}")
        owners.add(answer["owner_pid"])
        if calls == 1:
            _emit("following")
    _emit("done", calls=calls, owner_pids=sorted(owners, key=str))


def _take_over(server, stdin, churn_seconds):
    """This session took over: wait for its owner startup, check it serves, churn after GO."""
    _emit("promoted", pid=os.getpid())
    server._startup_thread.join(timeout=STARTUP_TIMEOUT_SECONDS)
    if server._startup_thread.is_alive() or _writer_role.STATE.failure:
        raise RuntimeError(f"the owner startup did not finish cleanly: {_writer_role.STATE.failure}")
    answer = _call(server, "search_knowledge")
    if answer["status"] != "success":
        raise RuntimeError(f"the new owner answered search_knowledge with {answer}")
    _emit("owner-ready", probe_exit_codes=PROBE_EXIT_CODES)
    stdin.wait_for_go()
    collection = server.get_orchestrator().collection
    cycles = _churn(collection, np.random.default_rng(os.getpid()), churn_seconds)
    _emit("done", count=collection.count(), cycles=cycles)
    stdin.closed.wait()


def run_follower(churn_seconds):
    stdin = _Stdin()
    _run_server(lambda server: _follow(server, stdin, float(churn_seconds)))


def _call_each_tool_once(server):
    """A short session: every tool once, each answer the refusal that names the owner."""
    _emit("lock", owns_writes=not _writer_role.STATE.is_follower, pid=os.getpid())
    owners = set()
    for name in TOOL_NAMES:
        answer = _call(server, name)
        if not is_refusal(answer):
            raise RuntimeError(f"a short session answered {name} with {answer}")
        owners.add(answer["owner_pid"])
    _emit("done", owner_pids=sorted(owners, key=str))


def run_session():
    _run_server(_call_each_tool_once)


def run_reopen():
    if not _writer_role.claim_writes():
        raise RuntimeError("a process of the round still owns data_dir after it stopped")
    _index_access.hold(wait_seconds=REOPEN_ACCESS_WAIT_SECONDS)
    collection = chromadb.PersistentClient(path=str(config.chroma_dir)).get_collection(
        config.collection_name, embedding_function=None
    )
    count = collection.count()
    query = np.random.default_rng().random((1, DIMENSIONS), dtype=np.float32).tolist()
    hits = collection.query(query_embeddings=query, n_results=QUERY_RESULTS)
    sample = collection.get(limit=SAMPLE_IDS, include=["embeddings"])
    _emit("done", count=count, hits=len(hits["ids"][0]), embeddings=len(sample["embeddings"]))


ROLES = {"owner": run_owner, "follower": run_follower, "session": run_session, "reopen": run_reopen}


def main(arguments):
    role, *role_arguments = arguments
    try:
        ROLES[role](*role_arguments)
    except Exception:
        _emit("error", error=traceback.format_exc()[-ERROR_TAIL_CHARS:])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
