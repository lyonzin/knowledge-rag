"""Folder scopes must constrain retrieval before ranking, including lexical paths."""

import json
import os
from pathlib import Path

import pytest

from mcp_server import server
from mcp_server.fts5_index import Fts5LexicalIndex
from mcp_server.search_scope import FolderScope


def _seed(orch, monkeypatch):
    """Crowd out the requested folder with higher lexical relevance elsewhere."""
    root = server.config.documents_dir
    monkeypatch.setattr(server.config, "reranker_enabled", False)
    monkeypatch.setattr(server.config, "query_expansions", {})
    for folder in ("wanted", "wanted/sub", "wanted/archive", "wanted-other"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "wanted/answer.md").write_text("scopeprobe " + "context " * 100, encoding="utf8")
    (root / "wanted/sub/child.md").write_text("scopeprobe child answer", encoding="utf8")
    (root / "wanted/archive/old.md").write_text("scopeprobe deprecated", encoding="utf8")
    for number in range(80):
        (root / f"wanted-other/noise-{number}.md").write_text(
            f"scopeprobe scopeprobe scopeprobe decoy {number}", encoding="utf8"
        )
    assert orch.index_all()["errors"] == 0
    index = Fts5LexicalIndex(root.parent / "scope-fts.db", root.parent / "scope-fts.state")
    page = orch.collection.get(include=["documents", "metadatas"])
    for identity, content, metadata in zip(page["ids"], page["documents"], page["metadatas"]):
        index.add_document(identity, content, metadata["filename"], metadata["category"])
    index._ready = True
    orch.fts5_index = index
    monkeypatch.setattr(server.config, "fts5_enabled", True)
    return root


@pytest.mark.parametrize("method,alpha", [("hybrid", 0), ("hybrid", 1), ("hybrid", 0.3), ("fts5", 0.3)])
def test_folder_scope_before_top_k(isolated_orchestrator, monkeypatch, method, alpha):
    orch = isolated_orchestrator
    root = _seed(orch, monkeypatch)
    results = orch.query(
        "scopeprobe",
        max_results=2,
        hybrid_alpha=alpha,
        search_method=method,
        include_folders=["wanted"],
        exclude_folders=["wanted/archive"],
    )
    assert {Path(item["source"]).relative_to(root).as_posix() for item in results} == {
        "wanted/answer.md",
        "wanted/sub/child.md",
    }


def test_scope_cache_and_reindex(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    root = _seed(orch, monkeypatch)
    arguments = {"max_results": 2, "hybrid_alpha": 0, "search_method": "hybrid"}
    first = orch.query("scopeprobe", **arguments, include_folders=["wanted/sub"])
    assert len(first) == 1
    assert not orch.query("scopeprobe", **arguments, include_folders=["missing"])
    second = orch.query("scopeprobe", **arguments, exclude_folders=["wanted"])
    assert second and all("wanted-other" in item["source"] for item in second)
    assert orch.query("scopeprobe", **arguments, include_folders=["wanted/sub/"]) == first
    (root / "wanted/sub/child.md").unlink()
    assert orch.index_all()["errors"] == 0
    assert not orch.query("scopeprobe", **arguments, include_folders=["wanted/sub"])


@pytest.mark.parametrize(
    "invalid",
    ["../outside", "/absolute", "C:/absolute", "C:relative", "\\\\host\\share", "wanted:stream", "wanted\0bad"],
)
def test_unsafe_scope_rejected(isolated_orchestrator, invalid):
    with pytest.raises(ValueError):
        isolated_orchestrator.query("probe", include_folders=[invalid])


def test_mcp_scope_validation_and_forwarding(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    root = _seed(orch, monkeypatch)
    monkeypatch.setattr(server, "get_orchestrator", lambda: orch)
    result = json.loads(
        server.search_knowledge("scopeprobe", include_folders=["wanted/sub"], search_method="fts5", snippet_mode=False)
    )
    assert result["status"] == "success"
    assert result["result_count"] == 1
    assert Path(result["results"][0]["source"]) == root / "wanted/sub/child.md"
    error = json.loads(server.search_knowledge("probe", include_folders=["../escape"]))
    assert error["status"] == "error"


def test_scope_union_exclusions_root_and_separator_normalization(tmp_path):
    root = tmp_path / "documents"
    (root / "team/keep").mkdir(parents=True)
    (root / "team/omit").mkdir()
    (root / "other").mkdir()
    metadata = {
        "keep": {"source": str(root / "team/keep/file.md")},
        "omit": {"source": str(root / "team/omit/file.md")},
        "other": {"source": str(root / "other/file.md")},
        "outside": {"source": str(tmp_path / "file.md")},
    }
    scope = FolderScope.create(root, ["team\\keep", "other/", "./other"], ["team/omit"])
    assert scope.document_ids(metadata) == ["keep", "other"]
    assert scope.key == FolderScope.create(root, ["other", "team/keep"], ["team/omit/"]).key
    assert not FolderScope.create(root, ["."], ["."]).document_ids(metadata)
    assert FolderScope.create(root, None, ["team"]).document_ids(metadata) == ["other"]


def test_folder_scope_rejects_symlink_escape(tmp_path):
    root = tmp_path / "documents"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "escape").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable to this account")
    with pytest.raises(ValueError):
        FolderScope.create(root, ["escape"], None)
    scope = FolderScope.create(root, ["."], None)
    assert scope.document_ids({"unsafe": {"source": str(root / "escape/file.md")}}) == []


def test_folders_are_directories_and_filter_count_is_bounded(tmp_path):
    (tmp_path / "file.md").write_text("text", encoding="utf8")
    for folders in (["file.md"], ["folder"] * 33, "folder", [""]):
        with pytest.raises(ValueError):
            FolderScope.create(tmp_path, folders, None)


@pytest.mark.skipif(os.name != "nt", reason="Windows path identity is case-insensitive")
def test_windows_scope_case_and_ambiguous_components(tmp_path):
    (tmp_path / "CaseFolder").mkdir()
    upper = FolderScope.create(tmp_path, ["CASEFOLDER"], None)
    lower = FolderScope.create(tmp_path, ["casefolder"], None)
    assert upper.key == lower.key
    assert upper.document_ids({"one": {"source": str(tmp_path / "CaseFolder/test.md")}}) == ["one"]
    for path in ("NUL", "COM1.txt", "folder.", "folder "):
        with pytest.raises(ValueError):
            FolderScope.create(tmp_path, [path], None)


def test_fts_scope_more_than_sqlite_legacy_parameter_limit(tmp_path):
    index = Fts5LexicalIndex(tmp_path / "many.db", tmp_path / "many.state")
    try:
        index.add_document("last", "scopeprobe", "file.md", "general")
        allowed = frozenset([f"missing-{number}" for number in range(1200)] + ["last"])
        assert index.search("scopeprobe", top_k=1, allowed_ids=allowed)[0][0] == "last"
        assert index.search("scopeprobe", top_k=1, allowed_ids=frozenset()) == []
    finally:
        index.close()
