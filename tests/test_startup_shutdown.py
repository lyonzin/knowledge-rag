"""Close a writer's transport between the two real renames; a fresh process must see one intact pair."""

import multiprocessing
import os
import sys
from pathlib import Path

from mcp_server import instance_lock

TIMEOUT = 30


def _serve_through_eof(read_end, release, events):
    from mcp_server import _writer_role, preflight, server

    moved, saw_eof, closing = events
    shutdown = _writer_role.close_startup_repair

    def observed_shutdown():
        closing.set()
        shutdown()

    _writer_role.close_startup_repair = observed_shutdown
    metadata = server.config.data_dir / "index_metadata.json"
    rename = os.rename

    def pause_after_metadata(src, dst):
        rename(src, dst)
        if Path(src) == metadata:
            moved.set()
            assert release.poll(TIMEOUT) and release.recv_bytes() == b"finish"

    def transport(name):
        assert name == "stdio"
        with read_end:
            try:
                read_end.recv_bytes()
            except EOFError:
                saw_eof.set()
            else:
                raise AssertionError("expected the client to close its pipe")

    preflight._probe_until_conclusive = lambda *_: preflight._Outcome.HNSW
    os.rename = pause_after_metadata
    server._run_transport = transport
    server._open_index_or_log = lambda: None  # nothing may open a model in this process
    sys.argv = ["knowledge-rag"]
    server.main()


def test_given_eof_between_renames_when_the_writer_exits_then_the_next_owner_sees_a_paired_backup(
    tmp_path, monkeypatch
):
    data = tmp_path / "data"
    chroma = data / "chroma_db"
    chroma.mkdir(parents=True)
    (chroma / "chroma.sqlite3").write_bytes(b"persisted vector")
    (data / "index_metadata.json").write_bytes(b"persisted catalog")
    monkeypatch.setenv("KNOWLEDGE_RAG_DIR", str(tmp_path))
    monkeypatch.setenv(instance_lock.ENV_VAR, "0")
    monkeypatch.setattr(instance_lock.config, "data_dir", data)
    ctx = multiprocessing.get_context("spawn")
    read_end, write_end = ctx.Pipe(duplex=False)
    release, resume = ctx.Pipe(duplex=False)
    moved, saw_eof, closing = ctx.Event(), ctx.Event(), ctx.Event()
    writer = ctx.Process(target=_serve_through_eof, args=(read_end, release, (moved, saw_eof, closing)))
    writer.start()
    read_end.close()
    release.close()
    try:
        assert moved.wait(TIMEOUT), "writer never completed the first rename"
        write_end.close()
        assert saw_eof.wait(TIMEOUT), "transport did not read EOF"
        assert closing.wait(TIMEOUT), "shutdown did not reach the repair barrier"
        with instance_lock.writer_lock() as owns_during_move:
            assert writer.is_alive() and not owns_during_move
    finally:
        write_end.close()
        try:
            resume.send_bytes(b"finish")
        except BrokenPipeError:
            pass  # the regression kills the writer before it can resume
        resume.close()
        writer.join(TIMEOUT)
        if writer.is_alive():
            writer.kill()
            writer.join(TIMEOUT)
    assert writer.exitcode == 0
    with instance_lock.writer_lock() as next_owner:
        (backup,) = (data / "backups").iterdir()
        assert next_owner and not chroma.exists() and not (data / "index_metadata.json").exists()
        assert (backup / "index_metadata.hnsw.json").read_bytes() == b"persisted catalog"
        assert (backup / "chroma_db.hnsw" / "chroma.sqlite3").read_bytes() == b"persisted vector"
