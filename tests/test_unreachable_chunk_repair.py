"""GH #313 — chunks the vector index cannot reach are linked again after indexing.

A query for a chunk's own stored vector does not always return the chunk, and
then no semantic search finds it. After an indexing operation the chunks it
wrote are checked that way, and the unreachable ones are updated with the
vector they already hold, which makes ChromaDB link them again.

A collection of a few rows reaches every one of them, so ``Reachability``
scripts the answers of the check on the real collection: the rows of hidden
ids come back without the chunk until it is updated. Everything else, the
updates included, goes to ChromaDB as it is.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from chromadb.errors import InternalError

from mcp_server import server

UNAVAILABLE = "index unavailable"
WARNING = "[WARN] Repair of unreachable chunks (search.repair_unreachable_chunks) did not finish"
# What this feature writes to stderr. ChromaDB may log lines of its own there.
PRINTED = ("[INDEX] Checking", "[INDEX] Repaired", "[WARN] Repair of unreachable chunks")


class Reachability:
    """Scripts which chunks the reachability check finds, and records what is read and written.

    ``hidden`` ids are not returned by their own vector until they are
    updated; ``hide_first`` hides that many ids of the first check, for ids a
    test cannot know beforehand. ``displaces`` maps an id to the one its
    update hides, and ``stuck`` ids stay hidden whatever is done.

    ``writes`` holds ``(method, ids)`` of every delete, add, upsert and
    update, ``update_fields`` what each update carried, and ``reads`` the
    ``include`` of every get by id. ``failures`` scripts the next calls of
    ``check`` (the query by own vectors) or of a write method: an error to
    raise, or None for a call that goes through.
    """

    def __init__(self, collection, patcher, hidden=(), displaces=None, stuck=(), hide_first=0, failures=None):
        self.hidden = {*hidden, *stuck}
        self.stuck = set(stuck)
        self.displaces = dict(displaces or {})
        self.hide_first = hide_first
        self.failures = {kind: list(errors) for kind, errors in (failures or {}).items()}
        self.checked: list = []
        self.writes: list = []
        self.update_fields: list = []
        self.reads: list = []
        self._asked: list = []
        self._real = {name: getattr(collection, name) for name in ("get", "query", "delete", "add", "upsert", "update")}
        patcher.setattr(collection, "get", self._get)
        patcher.setattr(collection, "query", self._query)
        for name in ("delete", "add", "upsert", "update"):
            patcher.setattr(collection, name, functools.partial(self._write, name))

    def _fail(self, kind):
        error = self.failures[kind].pop(0) if self.failures.get(kind) else None
        if error is not None:
            raise error

    def _get(self, *args, **kwargs):
        result = self._real["get"](*args, **kwargs)
        if "ids" in kwargs:
            self.reads.append(kwargs.get("include"))
            self._asked = list(result["ids"])
        return result

    def _query(self, *args, **kwargs):
        if "query_embeddings" not in kwargs:
            return self._real["query"](*args, **kwargs)
        self._fail("check")
        result = self._real["query"](*args, **kwargs)
        asked, self._asked = self._asked, []
        self.checked.append(sorted(asked))
        self.hidden.update(asked[: self.hide_first])
        self.hide_first = 0
        for position, chunk_id in enumerate(asked):
            if chunk_id in self.hidden:
                result["ids"][position] = ["another-chunk"]
                result["distances"][position] = [0.5]
        return result

    def _write(self, name, **kwargs):
        self.writes.append((name, list(kwargs["ids"])))
        if name == "update":
            self.update_fields.append(sorted(kwargs))
        self._fail(name)
        result = self._real[name](**kwargs)
        if name == "update":
            self._reveal(kwargs["ids"])
        return result

    def _reveal(self, chunk_ids):
        for chunk_id in chunk_ids:
            if chunk_id not in self.stuck:
                self.hidden.discard(chunk_id)
            if chunk_id in self.displaces:
                self.hidden.add(self.displaces.pop(chunk_id))

    def repair_writes(self) -> list:
        """Everything but the upserts of indexing."""
        return [write for write in self.writes if write[0] != "upsert"]


@pytest.fixture(autouse=True)
def switched_on(monkeypatch):
    """The default, whatever the config.yaml of the machine running the tests says."""
    monkeypatch.setattr(server.config, "repair_unreachable_chunks", True)


@pytest.fixture
def orch(isolated_orchestrator, monkeypatch):
    """An orchestrator whose index counts as large enough to be checked."""
    monkeypatch.setattr(isolated_orchestrator, "_REACHABILITY_MIN_CHUNKS", 0)
    return isolated_orchestrator


def _write(name: str, text: str) -> Path:
    path = Path(server.config.documents_dir) / name
    path.write_text(text, encoding="utf-8")
    return path


def _doc_id(orch, name: str) -> str:
    return orch._source_to_docid[str((Path(server.config.documents_dir) / name).resolve())]


def _index(orch, name: str, text: str, vector: float) -> str:
    """Index one single-chunk file under a vector of its own; return its chunk id."""
    orch.embed_fn.value = vector
    _write(name, text)
    assert orch.index_all()["errors"] == 0
    return f"{_doc_id(orch, name)}_0"


def _rows(orch) -> dict:
    result = orch.collection.get(include=["documents", "metadatas", "embeddings"])
    return {
        chunk_id: (text, meta, vector.tolist())
        for chunk_id, text, meta, vector in zip(
            result["ids"], result["documents"], result["metadatas"], result["embeddings"]
        )
    }


def _journal(orch) -> Path:
    return orch._metadata_file.with_name("document_repairs.json")


def _counts(stats: dict) -> tuple:
    """(repaired, remaining), as a pass or a repair that checked reports them."""
    return stats["unreachable_chunks_repaired"], stats["unreachable_chunks_remaining"]


def _printed(capsys) -> list:
    """The lines this feature wrote to stderr since the last read."""
    return [line for line in capsys.readouterr().err.splitlines() if line.startswith(PRINTED)]


def _warning(error: type) -> str:
    return f"{WARNING}, the operation itself is complete: {error.__name__}: {UNAVAILABLE}"


def _scripted_orchestrator(collection):
    """An orchestrator that knows nothing but its collection."""
    orch = object.__new__(server.KnowledgeOrchestrator)
    orch.collection = collection
    return orch


# ------------------------------------------------------------------ config


def test_repair_unreachable_chunks_default_is_on(monkeypatch):
    import mcp_server.config as config_module

    monkeypatch.setattr(config_module, "_yaml", {})
    assert config_module.Config().repair_unreachable_chunks is True


def test_repair_unreachable_chunks_from_yaml_is_respected(monkeypatch):
    import mcp_server.config as config_module

    monkeypatch.setattr(config_module, "_yaml", {"search": {"repair_unreachable_chunks": False}})
    assert config_module.Config().repair_unreachable_chunks is False


@pytest.mark.parametrize("raw", ["false", 0, 1.5, ["no"]])
def test_given_a_switch_that_is_not_a_boolean_when_loading_then_it_stays_on(raw, monkeypatch, capsys):
    import mcp_server.config as config_module

    monkeypatch.setattr(config_module, "_yaml", {"search": {"repair_unreachable_chunks": raw}})

    assert config_module.Config().repair_unreachable_chunks is True
    assert "search.repair_unreachable_chunks has wrong type" in capsys.readouterr().err


def test_given_a_field_that_is_not_a_boolean_when_validating_then_it_falls_back_to_on(monkeypatch, capsys):
    import mcp_server.config as config_module

    monkeypatch.setattr(config_module, "_yaml", {})

    assert config_module.Config(repair_unreachable_chunks="off").repair_unreachable_chunks is True
    assert "[WARN] repair_unreachable_chunks='off' invalid, using True" in capsys.readouterr().err


# --------------------------------------------------------------- the check


def test_given_the_rows_of_a_check_when_finding_unreachable_chunks_then_only_the_missed_one_is_returned():
    collection = MagicMock()
    # "gone" is not returned: its write was rolled back. The rows of the
    # query belong to the ids that came back, not to the ids asked for.
    collection.get.return_value = {"ids": ["own", "deep", "twin", "missed"], "embeddings": [[1.0], [2.0], [3.0], [4.0]]}
    # own: first in its row. deep: further down. twin: another id holds the
    # same vector, at distance zero. missed: neither.
    collection.query.return_value = {
        "ids": [["own", "x"], ["x", "y", "deep"], ["other"], ["x", "y"]],
        "distances": [[0.0, 0.3], [0.1, 0.2, 0.3], [0.0], [0.2, 0.4]],
    }
    orch = _scripted_orchestrator(collection)

    unreachable = orch._find_unreachable_chunks(["own", "gone", "deep", "twin", "missed"])

    assert unreachable == ["missed"]
    collection.get.assert_called_once_with(ids=["own", "gone", "deep", "twin", "missed"], include=["embeddings"])
    collection.query.assert_called_once_with(
        query_embeddings=[[1.0], [2.0], [3.0], [4.0]], n_results=10, include=["distances"]
    )


def test_given_a_batch_the_collection_does_not_hold_when_finding_unreachable_chunks_then_the_next_batch_is_checked(
    monkeypatch,
):
    collection = MagicMock()
    collection.get.side_effect = [{"ids": [], "embeddings": []}, {"ids": ["missed"], "embeddings": [[1.0]]}]
    collection.query.return_value = {"ids": [["other"]], "distances": [[0.4]]}
    orch = _scripted_orchestrator(collection)
    monkeypatch.setattr(orch, "_CHROMA_BATCH_SIZE", 1)

    unreachable = orch._find_unreachable_chunks(["gone", "missed"])

    assert unreachable == ["missed"]
    collection.query.assert_called_once()  # nothing is asked for the batch that holds no chunk


@pytest.mark.parametrize("distance, unreachable", [(5e-7, []), (2e-6, ["chunk"])])
def test_given_another_id_in_first_place_when_finding_unreachable_chunks_then_only_distance_zero_counts_as_found(
    distance, unreachable
):
    collection = MagicMock()
    collection.get.return_value = {"ids": ["chunk"], "embeddings": [[1.0]]}
    collection.query.return_value = {"ids": [["twin"]], "distances": [[distance]]}

    assert _scripted_orchestrator(collection)._find_unreachable_chunks(["chunk"]) == unreachable


# ------------------------------------------------------------- cost guards


@pytest.mark.parametrize("case", ["switched_off", "nothing_written", "small_index"])
def test_given_nothing_to_check_when_repairing_then_the_collection_is_not_called(
    isolated_orchestrator, monkeypatch, capsys, case
):
    orch = isolated_orchestrator
    orch.collection = MagicMock()
    # 1,000 catalogued chunks are enough for a check, 999 are not.
    orch._indexed_docs = {"a" * 16: {"source": "a.md", "chunks": 999 if case == "small_index" else 1000}}
    monkeypatch.setattr(server.config, "repair_unreachable_chunks", case != "switched_off")

    result = orch._repair_unreachable_chunks([] if case == "nothing_written" else ["a_0"])

    assert orch.collection.mock_calls == []
    assert result == {}
    assert _printed(capsys) == []


def test_given_a_catalogue_of_1000_chunks_when_repairing_then_the_written_chunks_are_checked(isolated_orchestrator):
    orch = isolated_orchestrator
    orch.collection = MagicMock()
    orch.collection.get.return_value = {"ids": [], "embeddings": []}
    orch._indexed_docs = {"a" * 16: {"source": "a.md", "chunks": 1000}}

    result = orch._repair_unreachable_chunks(["a_0"])

    orch.collection.get.assert_called_once_with(ids=["a_0"], include=["embeddings"])
    assert _counts(result) == (0, 0)


def test_given_a_small_index_when_documents_are_written_then_no_check_is_made(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    reach = Reachability(orch.collection, monkeypatch)
    _write("a.md", "alpha text")

    stats = orch.index_all()
    orch.add_document_from_content("beta text", "b.md", "general")

    assert reach.checked == [] and reach.reads == []
    assert [method for method, _ in reach.writes] == ["upsert", "upsert"]
    assert _counts(stats) == (0, None)


def test_given_a_catalogue_of_999_chunks_when_a_two_chunk_document_is_added_then_its_chunks_are_checked(
    isolated_orchestrator, monkeypatch
):
    orch = isolated_orchestrator
    orch._indexed_docs = {"f" * 16: {"source": "elsewhere.md", "chunks": 999}}
    reach = Reachability(orch.collection, monkeypatch)
    two_sections = "## One\n\n" + "alpha " * 40 + "\n\n## Two\n\n" + "beta " * 40

    result = orch.add_document_from_content(two_sections, "a.md", "general")

    # 999 + 2: the check counts the catalogue after the document is registered.
    assert result["chunks_added"] == 2
    assert reach.checked == [[f"{_doc_id(orch, 'a.md')}_0", f"{_doc_id(orch, 'a.md')}_1"]]


def test_given_the_switch_is_off_when_a_pass_writes_documents_then_it_collects_no_ids(orch, monkeypatch):
    monkeypatch.setattr(server.config, "repair_unreachable_chunks", False)
    _write("a.md", "alpha text")
    handed_over: list = []
    repair = orch._repair_unreachable_chunks

    def recording(chunk_ids):
        handed_over.append(list(chunk_ids))
        return repair(chunk_ids)

    monkeypatch.setattr(orch, "_repair_unreachable_chunks", recording)

    stats = orch.index_all()

    assert stats["indexed"] == 1
    assert handed_over == [[]]
    assert _counts(stats) == (0, None)


# -------------------------------------------------------------- the repair


def test_given_an_unreachable_chunk_when_repairing_then_one_update_carries_its_id_and_vector_and_no_row_changes(
    orch, monkeypatch, capsys
):
    hidden = _index(orch, "a.md", "alpha text", 1.0)
    other = _index(orch, "b.md", "beta text", 2.0)
    before = _rows(orch)
    reach = Reachability(orch.collection, monkeypatch, hidden=[hidden])
    orch.embed_fn.fail = True  # from here on, computing an embedding raises
    journal_read, journal_write = Mock(return_value={}), Mock()
    monkeypatch.setattr(orch, "_load_document_repairs", journal_read)
    monkeypatch.setattr(orch, "_save_document_repairs", journal_write)
    cached = orch.query_cache.generation

    result = orch._repair_unreachable_chunks([hidden, other])

    assert _rows(orch) == before
    assert reach.writes == [("update", [hidden])], "no delete, add or upsert: the chunk never leaves the collection"
    assert reach.update_fields == [["embeddings", "ids"]]
    assert all(include == ["embeddings"] for include in reach.reads), "no text or metadata is read"
    assert result == {"unreachable_chunks_repaired": 1, "unreachable_chunks_remaining": 0}
    assert _printed(capsys) == ["[INDEX] Repaired 1 unreachable chunks, 0 remaining"]
    journal_read.assert_not_called()
    journal_write.assert_not_called()
    assert not _journal(orch).exists()
    assert orch.query_cache.generation != cached, "results cached before the update are dropped"


def test_given_reachable_chunks_when_repairing_then_nothing_is_written(orch, monkeypatch, capsys):
    chunk_ids = [_index(orch, "a.md", "alpha text", 1.0), _index(orch, "b.md", "beta text", 2.0)]
    reach = Reachability(orch.collection, monkeypatch)

    result = orch._repair_unreachable_chunks(chunk_ids)

    assert reach.checked == [sorted(chunk_ids)]
    assert reach.writes == []
    assert _counts(result) == (0, 0)
    assert _printed(capsys) == []


def test_given_a_repeated_id_when_repairing_then_it_is_checked_and_updated_once(orch, monkeypatch, capsys):
    hidden = _index(orch, "a.md", "alpha text", 1.0)
    other = _index(orch, "b.md", "beta text", 2.0)
    reach = Reachability(orch.collection, monkeypatch, hidden=[hidden])

    # ChromaDB raises DuplicateIDError for a get() or an update() with a repeated id.
    result = orch._repair_unreachable_chunks([hidden, other, hidden])

    assert reach.checked[0] == sorted([hidden, other])
    assert reach.writes == [("update", [hidden])]
    assert _counts(result) == (1, 0)
    assert _printed(capsys) == ["[INDEX] Repaired 1 unreachable chunks, 0 remaining"]


def test_given_an_update_displaces_another_chunk_when_repairing_then_the_next_round_repairs_it(orch, monkeypatch):
    first = _index(orch, "a.md", "alpha text", 1.0)
    second = _index(orch, "b.md", "beta text", 2.0)
    before = _rows(orch)
    reach = Reachability(orch.collection, monkeypatch, hidden=[first], displaces={first: second})

    result = orch._repair_unreachable_chunks([first, second])

    assert reach.writes == [("update", [first]), ("update", [second])]
    assert len(reach.checked) == 3
    assert _counts(result) == (2, 0)
    assert _rows(orch) == before


def test_given_a_chunk_no_update_makes_reachable_when_repairing_then_the_third_check_only_reports(
    orch, monkeypatch, capsys
):
    stuck = _index(orch, "a.md", "alpha text", 1.0)
    other = _index(orch, "b.md", "beta text", 2.0)
    before = _rows(orch)
    reach = Reachability(orch.collection, monkeypatch, stuck=[stuck])

    result = orch._repair_unreachable_chunks([stuck, other])

    assert len(reach.checked) == 3
    assert reach.writes == [("update", [stuck])] * 2
    assert _counts(result) == (0, 1)
    assert _printed(capsys) == ["[INDEX] Repaired 0 unreachable chunks, 1 remaining"]
    assert _rows(orch) == before


def test_given_more_chunks_than_a_batch_when_repairing_then_each_batch_is_updated_on_its_own(orch, monkeypatch):
    first = _index(orch, "a.md", "alpha text", 1.0)
    second = _index(orch, "b.md", "beta text", 2.0)
    before = _rows(orch)
    monkeypatch.setattr(orch, "_CHROMA_BATCH_SIZE", 1)
    reach = Reachability(orch.collection, monkeypatch, hidden=[first, second])

    result = orch._repair_unreachable_chunks([first, second])

    assert reach.writes == [("update", [first]), ("update", [second])]
    assert _counts(result) == (2, 0)
    assert _rows(orch) == before


def test_given_ids_the_collection_no_longer_holds_when_linking_again_then_only_the_ids_it_returned_are_updated(
    monkeypatch,
):
    collection = MagicMock()
    collection.get.side_effect = [{"ids": [], "embeddings": []}, {"ids": ["a", "b"], "embeddings": [[1.0], [2.0]]}]
    orch = _scripted_orchestrator(collection)
    monkeypatch.setattr(orch, "_CHROMA_BATCH_SIZE", 3)

    # The first batch holds nothing any more; the second lost the id between the two it kept.
    orch._relink_chunks(["gone", "gone-too", "gone-as-well", "a", "left", "b"])

    collection.update.assert_called_once_with(ids=["a", "b"], embeddings=[[1.0], [2.0]])


def test_given_published_keyword_indexes_when_a_chunk_is_updated_then_bm25_and_fts5_are_left_alone(orch, monkeypatch):
    hidden = _index(orch, "a.md", "alpha text", 1.0)
    orch._ensure_bm25_index()
    assert orch._bm25_initialized and not orch._bm25_stale.is_set()
    monkeypatch.setattr(server.config, "fts5_enabled", True)
    orch.fts5_index = Mock()
    Reachability(orch.collection, monkeypatch, hidden=[hidden])

    result = orch._repair_unreachable_chunks([hidden])

    # Both hold chunk ids and texts, and the update changes neither.
    assert _counts(result) == (1, 0)
    assert orch._bm25_initialized and not orch._bm25_stale.is_set()
    assert orch.fts5_index.mock_calls == []


@pytest.mark.parametrize("batch_size, announced", [(1, True), (2, False)])
def test_given_a_check_of_more_than_one_batch_when_it_starts_then_it_says_how_many_chunks(
    orch, monkeypatch, capsys, batch_size, announced
):
    chunk_ids = [_index(orch, "a.md", "alpha text", 1.0), _index(orch, "b.md", "beta text", 2.0)]
    monkeypatch.setattr(orch, "_CHROMA_BATCH_SIZE", batch_size)

    orch._repair_unreachable_chunks(chunk_ids)

    assert _printed(capsys) == (["[INDEX] Checking 2 chunks for reachability..."] if announced else [])


# ------------------------------------- a repair that cannot run is a warning


@pytest.mark.parametrize(
    "failing, calls_before, error, completed, repaired",
    [
        ("check", 0, RuntimeError, 0, 0),
        ("update", 0, OSError, 1, 0),
        ("check", 1, InternalError, 1, 0),
        # The second check found one chunk of the three found so far.
        ("check", 2, RuntimeError, 2, 2),
    ],
    ids=["first-check", "first-update", "second-check", "third-check"],
)
def test_given_a_failure_when_repairing_then_it_warns_and_reports_no_count_of_remaining_chunks(
    orch, monkeypatch, capsys, failing, calls_before, error, completed, repaired
):
    first = _index(orch, "a.md", "alpha text", 1.0)
    second = _index(orch, "b.md", "beta text", 2.0)
    third = _index(orch, "c.md", "gamma text", 3.0)
    before = _rows(orch)
    cached = orch.query_cache.generation
    failures = {failing: [None] * calls_before + [error(UNAVAILABLE)]}
    reach = Reachability(
        orch.collection, monkeypatch, hidden=[first, second], displaces={first: third}, failures=failures
    )

    result = orch._repair_unreachable_chunks([first, second, third])

    # Repaired is what a completed check confirmed; no check stands behind a count of remaining chunks.
    assert result == {"unreachable_chunks_repaired": repaired, "unreachable_chunks_remaining": None}
    assert _printed(capsys) == [_warning(error)], "the warning is the report: no [INDEX] Repaired line"
    assert len(reach.checked) == len(reach.writes) == completed
    assert (orch.query_cache.generation != cached) == (completed > 0), "dropped once an update may have run"
    assert _rows(orch) == before


@pytest.mark.parametrize("failing, error", [("check", OSError), ("update", InternalError)])
def test_given_the_repair_cannot_run_when_a_smart_reindex_ends_then_the_pass_ends_as_it_would_without_it(
    orch, monkeypatch, capsys, failing, error
):
    _write("a.md", "alpha text")
    orch._reindex_progress = {"active": True, "operation": "smart_reindex"}
    orch._write_checkpoint(operation="smart_reindex", indexed_doc_ids=[], chunks_processed=0)
    Reachability(orch.collection, monkeypatch, hide_first=1, failures={failing: [error(UNAVAILABLE)]})

    stats = orch.index_all(force=True)

    assert stats["errors"] == 0 and stats["failed_files"] == [], "nothing for a watcher to retry"
    assert stats["indexed"] == 1 and orch.collection.count() == 1
    assert not orch._checkpoint_file.exists(), "a run without errors clears its checkpoint"
    assert list(json.loads(orch._metadata_file.read_text(encoding="utf-8"))) == [_doc_id(orch, "a.md")]
    assert _counts(stats) == (0, None)
    assert _printed(capsys) == [_warning(error)]


def test_given_the_check_raises_when_a_document_is_added_then_the_write_still_succeeds(orch, monkeypatch, capsys):
    Reachability(orch.collection, monkeypatch, failures={"check": [InternalError(UNAVAILABLE)]})

    result = orch.add_document_from_content("alpha text", "a.md", "general")

    assert result["chunks_added"] == 1
    assert _printed(capsys) == [_warning(InternalError)]
    assert orch.collection.count() == 1


# ------------------------------------------------- where the repair runs


@pytest.mark.parametrize("entry", ["index_all", "index_paths"])
def test_given_an_unreachable_chunk_when_a_pass_wrote_it_then_the_stats_count_the_repair(orch, monkeypatch, entry):
    path = _write("a.md", "alpha text")
    reach = Reachability(orch.collection, monkeypatch, hide_first=1)

    stats = orch.index_all() if entry == "index_all" else orch.index_paths([str(path)])

    chunk_id = f"{_doc_id(orch, 'a.md')}_0"
    assert stats["indexed"] == 1 and stats["errors"] == 0
    assert _counts(stats) == (1, 0)
    assert reach.writes == [("upsert", [chunk_id]), ("update", [chunk_id])]
    assert orch.collection.count() == 1


def test_given_a_smart_reindex_when_the_repair_starts_then_the_catalogue_is_saved_and_the_checkpoint_settled(
    orch, monkeypatch
):
    _write("a.md", "alpha text")
    orch._reindex_progress = {"active": True, "operation": "smart_reindex"}
    orch._write_checkpoint(operation="smart_reindex", indexed_doc_ids=[], chunks_processed=0)
    at_start: dict = {}

    def observing(chunk_ids):
        saved = orch._metadata_file
        at_start["catalogue"] = list(json.loads(saved.read_text(encoding="utf-8"))) if saved.exists() else None
        at_start["checkpoint"] = orch._checkpoint_file.exists()
        return {}

    monkeypatch.setattr(orch, "_repair_unreachable_chunks", observing)

    orch.index_all(force=True)

    # A process that dies during the checks leaves no checkpoint of a finished run.
    assert at_start == {"catalogue": [_doc_id(orch, "a.md")], "checkpoint": False}


def test_given_a_pass_that_writes_nothing_when_it_ends_then_no_check_is_made(orch, monkeypatch):
    _index(orch, "a.md", "alpha text", 1.0)
    reach = Reachability(orch.collection, monkeypatch, hide_first=1)

    stats = orch.index_all()

    assert stats["skipped"] == 1
    assert reach.checked == [] and reach.writes == []
    assert _counts(stats) == (0, None)


def test_given_a_document_whose_write_fails_when_a_pass_ends_then_only_written_chunks_are_checked(orch, monkeypatch):
    _write("good.md", "good text")
    _write("bad.md", "bad text")
    replace = orch._replace_document_chunks

    def failing(doc, *args):
        if doc.source.name == "bad.md":
            raise RuntimeError("embedding failure")
        return replace(doc, *args)

    monkeypatch.setattr(orch, "_replace_document_chunks", failing)
    reach = Reachability(orch.collection, monkeypatch)

    stats = orch.index_all()

    assert stats["indexed"] == 1 and stats["errors"] == 1
    assert reach.checked == [[f"{_doc_id(orch, 'good.md')}_0"]]


def test_given_a_forced_reindex_whose_write_fails_when_the_pass_ends_then_the_restored_chunks_are_not_checked(
    orch, monkeypatch
):
    chunk_id = _index(orch, "a.md", "alpha text", 1.0)
    reach = Reachability(orch.collection, monkeypatch)
    orch.embed_fn.fail = True  # the replacement cannot be embedded, so the old chunk is put back

    stats = orch.index_all(force=True)

    # The ids are collected after the write: this pass wrote nothing that stayed.
    assert stats["errors"] == 1
    assert orch.collection.get(include=[])["ids"] == [chunk_id], "restored under the id it had"
    assert reach.checked == []


def _script_staging(orch, monkeypatch, **script) -> list:
    """Script the reachability of the staging collection a rebuild is about to create."""
    scripted: list = []
    create = orch._create_staging_collection

    def create_scripted(ts):
        staging = create(ts)
        scripted.append(Reachability(staging, monkeypatch, **script))
        return staging

    monkeypatch.setattr(orch, "_create_staging_collection", create_scripted)
    return scripted


def test_given_a_full_rebuild_when_a_staged_chunk_is_unreachable_then_staging_is_repaired_not_production(
    orch, monkeypatch
):
    _index(orch, "a.md", "alpha text", 1.0)
    _index(orch, "b.md", "beta text", 1.0)
    before = _rows(orch)
    production = Reachability(orch.collection, monkeypatch)
    staging = _script_staging(orch, monkeypatch, hide_first=1)

    stats = orch.nuclear_rebuild()

    assert _counts(stats) == (1, 0) and stats["errors"] == 0
    assert production.writes == [] and production.checked == []
    assert [method for method, _ in staging[0].repair_writes()] == ["update"]
    assert _rows(orch) == before


def test_given_the_check_raises_when_a_full_rebuild_ends_its_pass_then_the_rebuild_is_still_published(
    orch, monkeypatch, capsys
):
    _index(orch, "a.md", "alpha text", 1.0)
    _index(orch, "b.md", "beta text", 1.0)
    before = _rows(orch)
    production = orch.collection
    _script_staging(orch, monkeypatch, failures={"check": [OSError(UNAVAILABLE)]})

    stats = orch.nuclear_rebuild()

    assert stats["errors"] == 0 and stats["indexed"] == 2
    assert orch.collection.id != production.id, "the staging collection was swapped in"
    assert _rows(orch) == before
    assert _counts(stats) == (0, None)
    assert _printed(capsys) == [_warning(OSError)]


@pytest.mark.parametrize("operation", ["add", "update"])
def test_given_an_unreachable_chunk_when_a_document_is_stored_then_it_is_repaired_and_the_response_keeps_its_shape(
    orch, monkeypatch, capsys, operation
):
    if operation == "update":
        orch.add_document_from_content("alpha text", "a.md", "general")
    reach = Reachability(orch.collection, monkeypatch, hide_first=1)

    if operation == "add":
        result = orch.add_document_from_content("alpha text", "a.md", "general")
        expected = {"chunks_added", "old_chunks_removed", "dedup_skipped", "filepath", "category"}
    else:
        result = orch.update_document_content("a.md", "alpha text, rewritten")
        expected = {"new_chunks_added", "old_chunks_removed", "dedup_skipped", "filepath"}

    chunk_id = f"{_doc_id(orch, 'a.md')}_0"
    assert set(result) == expected
    assert reach.repair_writes()[-1] == ("update", [chunk_id])
    assert _printed(capsys) == ["[INDEX] Repaired 1 unreachable chunks, 0 remaining"]
    assert orch.collection.count() == 1


# ----------------------------------------------- real ChromaDB, real answers


class _VectorPerText:
    """One vector per text length, so no two chunks of the test share one."""

    def __call__(self, input):
        return [[float(len(text)), 1.0] for text in input]

    @staticmethod
    def name():
        return "test-vector-per-text"

    def embed_query(self, input):
        return self(input)

    @staticmethod
    def is_legacy():
        return True


def test_given_a_healthy_index_on_disk_when_a_pass_checks_it_then_no_chunk_is_rewritten(
    isolated_orchestrator, tmp_path, monkeypatch, capsys
):
    import chromadb

    orch = isolated_orchestrator
    name = server.config.collection_name
    orch.chroma_client.delete_collection(name)
    orch.embed_fn = _VectorPerText()
    orch.chroma_client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    orch.collection = orch.chroma_client.create_collection(name, embedding_function=orch.embed_fn)
    documents = 12
    monkeypatch.setattr(orch, "_REACHABILITY_MIN_CHUNKS", documents)
    for index in range(documents):
        _write(f"note-{index}.md", "word " * (index + 1))
    reach = Reachability(orch.collection, monkeypatch)  # nothing hidden: ChromaDB's own answers

    stats = orch.index_all()

    assert stats["indexed"] == documents and stats["errors"] == 0
    assert reach.checked == [sorted(orch.collection.get(include=[])["ids"])]
    assert reach.repair_writes() == []
    assert _counts(stats) == (0, 0)
    assert _printed(capsys) == []
