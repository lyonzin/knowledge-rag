"""Tests for Jupyter Notebook (.ipynb) parser."""

import json

import pytest

from mcp_server.ingestion import DocumentParser


@pytest.fixture
def parser():
    return DocumentParser()


def _make_notebook(cells, kernel="python3", nbformat=4):
    """Helper to create a valid .ipynb JSON structure."""
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"name": kernel, "display_name": f"Python 3 ({kernel})"}},
        "nbformat": nbformat,
        "nbformat_minor": 2,
    }


@pytest.fixture
def sample_ipynb(tmp_path):
    """Basic notebook with markdown and code cells."""
    nb = _make_notebook(
        [
            {"cell_type": "markdown", "source": ["# Title\n", "\n", "Some intro text."], "metadata": {}},
            {
                "cell_type": "code",
                "source": ["import pandas as pd\n", "df = pd.read_csv('data.csv')"],
                "metadata": {},
                "execution_count": 1,
                "outputs": [{"output_type": "execute_result", "data": {"text/plain": ["   col1  col2"]}}],
            },
            {"cell_type": "markdown", "source": ["## Analysis"], "metadata": {}},
            {
                "cell_type": "code",
                "source": ["print(df.describe())"],
                "metadata": {},
                "execution_count": 2,
                "outputs": [],
            },
        ]
    )
    f = tmp_path / "analysis.ipynb"
    f.write_text(json.dumps(nb), encoding="utf-8")
    return f


@pytest.fixture
def ipynb_with_base64(tmp_path):
    """Notebook with base64-encoded image output (should be ignored)."""
    nb = _make_notebook(
        [
            {
                "cell_type": "code",
                "source": ["import matplotlib.pyplot as plt\n", "plt.plot([1,2,3])"],
                "metadata": {},
                "execution_count": 1,
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {
                            "image/png": "iVBORw0KGgoAAAANSUhEUg" + "A" * 500,  # Fake base64
                            "text/plain": ["<Figure>"],
                        },
                    }
                ],
            },
        ]
    )
    f = tmp_path / "plots.ipynb"
    f.write_text(json.dumps(nb), encoding="utf-8")
    return f


class TestIpynbParser:
    def test_basic_parsing(self, parser, sample_ipynb):
        doc = parser.parse_file(sample_ipynb)
        assert doc is not None
        assert doc.format == ".ipynb"
        assert "Title" in doc.content
        assert "import pandas" in doc.content
        assert "Analysis" in doc.content

    def test_code_cells_wrapped_in_python_blocks(self, parser, sample_ipynb):
        doc = parser.parse_file(sample_ipynb)
        assert "```python" in doc.content
        assert "```" in doc.content

    def test_outputs_not_in_content(self, parser, sample_ipynb):
        """Cell outputs must NOT appear in indexed content."""
        doc = parser.parse_file(sample_ipynb)
        assert "col1" not in doc.content
        assert "execute_result" not in doc.content

    def test_base64_not_in_content(self, parser, ipynb_with_base64):
        """Base64-encoded images must NOT appear in indexed content."""
        doc = parser.parse_file(ipynb_with_base64)
        assert "iVBORw0KGgo" not in doc.content
        assert "image/png" not in doc.content

    def test_metadata_correct(self, parser, sample_ipynb):
        doc = parser.parse_file(sample_ipynb)
        assert doc.metadata["type"] == "jupyter_notebook"
        assert doc.metadata["cells"] == 4
        assert doc.metadata["code_cells"] == 2
        assert doc.metadata["markdown_cells"] == 2
        assert doc.metadata["nbformat"] == 4
        assert "python3" in doc.metadata["kernel"].lower() or "python 3" in doc.metadata["kernel"].lower()

    def test_empty_cells_ignored(self, parser, tmp_path):
        nb = _make_notebook(
            [
                {"cell_type": "markdown", "source": [], "metadata": {}},
                {"cell_type": "code", "source": ["x = 1"], "metadata": {}, "execution_count": None, "outputs": []},
                {"cell_type": "markdown", "source": [""], "metadata": {}},
            ]
        )
        f = tmp_path / "sparse.ipynb"
        f.write_text(json.dumps(nb), encoding="utf-8")
        doc = parser.parse_file(f)
        assert doc is not None
        assert "x = 1" in doc.content
        assert doc.metadata["code_cells"] == 1
        assert doc.metadata["markdown_cells"] == 0  # Both markdown cells were empty

    def test_source_as_string(self, parser, tmp_path):
        """Some notebooks have source as a single string, not a list."""
        nb = _make_notebook(
            [
                {"cell_type": "markdown", "source": "# String Source", "metadata": {}},
                {"cell_type": "code", "source": "print('hello')", "metadata": {}, "execution_count": 1, "outputs": []},
            ]
        )
        f = tmp_path / "string_src.ipynb"
        f.write_text(json.dumps(nb), encoding="utf-8")
        doc = parser.parse_file(f)
        assert "String Source" in doc.content
        assert "hello" in doc.content

    def test_malformed_json(self, parser, tmp_path):
        """Invalid JSON should not crash, falls back to raw content."""
        f = tmp_path / "broken.ipynb"
        f.write_text("{not valid json", encoding="utf-8")
        doc = parser.parse_file(f)
        assert doc is not None
        assert doc.metadata.get("is_valid_json") is False

    def test_malformed_notebook_structure_does_not_crash(self, parser, tmp_path):
        # Valid JSON but not a well-formed notebook: null metadata/cells, a
        # non-dict cell, or a bare list. These must be handled gracefully
        # instead of crashing with AttributeError/TypeError.
        malformed = [
            json.dumps({"nbformat": 4, "metadata": None, "cells": []}),
            json.dumps({"nbformat": 4, "metadata": {}, "cells": None}),
            json.dumps({"nbformat": 4, "metadata": {}, "cells": [42, "x"]}),
            json.dumps([1, 2, 3]),
        ]
        for i, raw in enumerate(malformed):
            f = tmp_path / f"malformed_{i}.ipynb"
            f.write_text(raw, encoding="utf-8")
            # Must not raise AttributeError/TypeError. Returning None (skipped as
            # empty) or a Document are both acceptable outcomes.
            parser.parse_file(f)

    def test_chunks_generated(self, parser, sample_ipynb):
        doc = parser.parse_file(sample_ipynb)
        assert len(doc.chunks) > 0
        for chunk in doc.chunks:
            assert chunk.content
            assert chunk.index >= 0
