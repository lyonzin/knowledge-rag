"""Indexing-pipeline microbenchmarks.

Tracked metrics for the perf gate:
    bench_chunk_markdown_small      single 5KB markdown
    bench_chunk_markdown_large      single 100KB markdown
    bench_parse_json_nested         deeply nested JSON
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def sample_markdown_5kb(tmp_path):
    body = "# Doc\n\n" + ("## Section\n\nContent paragraph.\n\n" * 100)
    path = tmp_path / "small.md"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def sample_markdown_100kb(tmp_path):
    body = "# Doc\n\n" + ("## Section\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit.\n\n" * 1000)
    path = tmp_path / "large.md"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def sample_json_nested(tmp_path):
    data = {f"key_{i}": {"sub": list(range(20))} for i in range(50)}
    path = tmp_path / "nested.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_bench_chunk_markdown_small(benchmark, sample_markdown_5kb):
    from mcp_server.ingestion import DocumentParser

    parser = DocumentParser()

    def parse():
        return parser.parse_file(sample_markdown_5kb)

    doc = benchmark(parse)
    assert doc is not None
    assert len(doc.chunks) > 0


def test_bench_chunk_markdown_large(benchmark, sample_markdown_100kb):
    from mcp_server.ingestion import DocumentParser

    parser = DocumentParser()

    def parse():
        return parser.parse_file(sample_markdown_100kb)

    doc = benchmark(parse)
    assert doc is not None
    assert len(doc.chunks) > 10


def test_bench_parse_json_nested(benchmark, sample_json_nested):
    from mcp_server.ingestion import DocumentParser

    parser = DocumentParser()

    def parse():
        return parser.parse_file(sample_json_nested)

    doc = benchmark(parse)
    assert doc is not None
