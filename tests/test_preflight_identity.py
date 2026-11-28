"""Upgrade, promotion and bootstrap: prove what a fresh process actually opened and counted."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mcp_server import _index_access, _writer_role, instance_lock, preflight

DOCUMENT = "the same indexed document"
CHILD_TIMEOUT = 90
CHROMA = """
import chromadb, json, sys
class FastEmbedLike:
    def __call__(self, input): return [[0.1, 0.2] for _ in input]
    def name(self): return 'fastembed-test'
client = chromadb.PersistentClient(path=sys.argv[1])
name = sys.argv[2]
if sys.argv[3] == 'seed':
    client.create_collection(name, embedding_function=FastEmbedLike()).add(ids=['doc'], documents=[sys.argv[4]])
elif sys.argv[3] == 'empty-default':
    client.delete_collection(name)
    client.create_collection(name)
collection = client.get_collection(name, embedding_function=None)
print(json.dumps([collection.count(), collection.get(ids=['doc'])['documents']]))
"""
COUNT_TRACE = """
import os
from chromadb.api.models.Collection import Collection
_original = Collection.count
def _count(self, *args, **kwargs):
    size = _original(self, *args, **kwargs)
    with (target.data_dir / 'count.trace').open('a', encoding='utf-8') as log:
        print(os.getpid(), self.name, size, file=log)
    return size
Collection.count = _count
assert 'mcp_server.config' not in sys.modules
"""
COUNT_FAILURE = """
import os
from chromadb.api.models.Collection import Collection
from chromadb.errors import InternalError
_original = Collection.count
def _count(self, *args, **kwargs):
    path = target.data_dir / 'attempts'
    again = path.exists()
    with path.open('a', encoding='utf-8') as log:
        print(os.getpid(), self.name, file=log)
    if again and os.environ['R2_COUNT_FAILURE'] == 'once':
        return _original(self, *args, **kwargs)
    raise InternalError('Error loading hnsw index')
Collection.count = _count
"""


@pytest.fixture
def index(tmp_path, monkeypatch):
    base = tmp_path / "rag"
    data = base / "cátalogo A"
    data.mkdir(parents=True)
    monkeypatch.setenv("KNOWLEDGE_RAG_DIR", str(base))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.setattr(_writer_role, "STATE", _writer_role.RoleState())
    monkeypatch.setattr(preflight.config, "data_dir", data)
    monkeypatch.setattr(preflight.config, "chroma_dir", base / "chroma A")
    monkeypatch.setattr(preflight.config, "collection_name", "collection_A")
    monkeypatch.setattr(preflight, "PROBE_RETRY_DELAY_SECONDS", 0)
    return preflight.config


def _chroma(index, mode):
    done = subprocess.run(
        [sys.executable, "-I", "-c", CHROMA, str(index.chroma_dir), index.collection_name, mode, DOCUMENT],
        cwd=index.data_dir.parent,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
        timeout=CHILD_TIMEOUT,
    )
    return json.loads(done.stdout)


def _trace_counts(monkeypatch):
    monkeypatch.setattr(preflight, "_PROBE_BODY", COUNT_TRACE + preflight._PROBE_BODY)


def _probe(index):
    done = preflight._probe_chroma(preflight._capture_target(), timeout_seconds=CHILD_TIMEOUT)
    assert done.returncode == 0, done.stderr
    return done.stdout.splitlines()[1:]


def test_old_checkout_in_base_and_pythonpath_cannot_select_the_child_package(index, monkeypatch):
    base = index.data_dir.parent
    old = base / "mcp_server"
    old.mkdir()
    (old / "__init__.py").write_text("# installed before _index_access existed\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(base))  # no parent checkout or installed source on the child's path
    monkeypatch.chdir(base)
    _trace_counts(monkeypatch)
    assert _chroma(index, "seed") == _chroma(index, "reopen") == [1, [DOCUMENT]]

    assert preflight.run_preflight(CHILD_TIMEOUT) is False

    pid, name, size = (index.data_dir / "count.trace").read_text().split()
    assert (int(pid) != os.getpid(), name, size) == (True, index.collection_name, "1")
    assert _chroma(index, "reopen") == [1, [DOCUMENT]]
    assert not (index.data_dir / "backups").exists()


def test_relative_parent_paths_do_not_resolve_again_in_child_cwd(index, monkeypatch):
    monkeypatch.chdir(index.data_dir.parent)
    monkeypatch.setattr(index, "data_dir", Path(index.data_dir.name))
    monkeypatch.setattr(index, "chroma_dir", Path(index.chroma_dir.name))
    monkeypatch.setenv("KNOWLEDGE_RAG_DIR", "rag")
    _chroma(index, "seed")
    assert _probe(index) == ["counting", "1"]
    assert not Path("rag").exists()


def test_probe_never_creates_collection_or_overwrites_embedding_function(index):
    assert _probe(index) == ["missing"]
    assert not index.chroma_dir.exists()
    assert _chroma(index, "seed") == [1, [DOCUMENT]]  # create_collection fails if the probe created it first
    assert _probe(index) == ["counting", "1"]
    assert _chroma(index, "empty-default") == [0, []]
    assert _probe(index) == ["counting", "0"]
    index.collection_name = "absent_in_existing_store"
    assert _probe(index) == ["missing"]


def test_promoted_follower_never_reads_changed_yaml_or_creates_any_paths_in_b(index, monkeypatch):
    base = index.data_dir.parent
    _chroma(index, "seed")
    _trace_counts(monkeypatch)
    with instance_lock.writer_lock() as owns:
        assert owns and not _writer_role.claim_writes()
        _writer_role.become_follower(lambda: preflight._probe_and_repair(CHILD_TIMEOUT))
        (base / "config.yaml").write_text(
            "paths:\n  data_dir: B/data\n  documents_dir: B/docs\n  models_cache_dir: B/cache\n"
            "search:\n  collection_name: collection_B\n",
            encoding="utf-8",
        )
    _writer_role._take_over_or_refuse(_writer_role.STATE)
    assert _writer_role.STATE.failure is None
    assert (index.data_dir / "count.trace").read_text().split()[1:] == ["collection_A", "1"]
    assert instance_lock._lock_sidecar(index.data_dir / _index_access.ACCESS_FILENAME) is None
    assert not (base / "B").exists()  # importing config even just for the lock creates B and its children
    _writer_role._release_claim()
    assert _chroma(index, "reopen") == [1, [DOCUMENT]]


def test_wrong_origin_is_an_error_before_ready_not_a_repair(index, monkeypatch):
    _chroma(index, "seed")
    # A second copy (editable or installed) absent at runtime must never silently fall back to another copy.
    absent = index.data_dir.parent / "gone" / "mcp_server" / "_preflight_probe.py"
    absent.parent.mkdir(parents=True)
    monkeypatch.setattr(preflight, "_PROBE_MODULE", absent)
    _writer_role.STATE.ready.clear()
    assert not _writer_role.run_startup_check(preflight._probe_and_repair)
    assert _writer_role.index_status() == "error"
    assert not _index_access.is_held()
    assert not (index.data_dir / "backups").exists()
    assert _chroma(index, "reopen") == [1, [DOCUMENT]]


def test_import_failure_repeated_on_healthy_index_never_opens_or_moves_it(index, monkeypatch):
    _chroma(index, "seed")
    blocked = """
class BrokenChroma:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'chromadb':
            raise ImportError('Error loading hnsw index')
sys.meta_path.insert(0, BrokenChroma())
"""
    monkeypatch.setattr(preflight, "_PROBE_BODY", blocked + preflight._PROBE_BODY)
    with pytest.raises(RuntimeError, match="Could not verify"):
        preflight.run_preflight(CHILD_TIMEOUT)
    assert (index.data_dir / "preflight.log").read_text().count("ImportError: Error loading hnsw index") == 3
    assert not (index.data_dir / "backups").exists()
    assert _chroma(index, "reopen") == [1, [DOCUMENT]]


@pytest.mark.parametrize("mode, moved, attempts", [("once", False, 2), ("persistent", True, 3)])
def test_typed_hnsw_failure_requires_persistence_across_independent_counts(index, monkeypatch, mode, moved, attempts):
    _chroma(index, "seed")
    monkeypatch.setenv("R2_COUNT_FAILURE", mode)
    monkeypatch.setattr(preflight, "_PROBE_BODY", COUNT_FAILURE + preflight._PROBE_BODY)
    assert preflight.run_preflight(CHILD_TIMEOUT) is moved
    pids = (index.data_dir / "attempts").read_text().splitlines()
    assert len(set(pids)) == len(pids) == attempts
    if moved:
        assert not index.chroma_dir.exists()
        [backup] = (index.data_dir / "backups").glob("auto-repair-*/chroma_db.hnsw")
        monkeypatch.setattr(index, "chroma_dir", backup)
    else:
        assert not (index.data_dir / "backups").exists()
    assert _chroma(index, "reopen") == [1, [DOCUMENT]]
