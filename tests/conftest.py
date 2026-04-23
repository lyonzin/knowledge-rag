"""Shared fixtures for knowledge-rag tests.

Mocks embeddings and ChromaDB to avoid model downloads in CI.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_embedding():
    """Mock FastEmbed to avoid model download in CI."""
    with patch("mcp_server.server.FastEmbedEmbeddings") as mock:
        instance = MagicMock()
        instance.__call__ = MagicMock(return_value=[[0.1] * 384])
        instance.name.return_value = "mock-embed"
        instance.embed_documents.return_value = [[0.1] * 384]
        instance.embed_query.return_value = [[0.1] * 384]
        instance._dim = 384
        mock.return_value = instance
        yield instance


@pytest.fixture
def sample_markdown(tmp_path):
    """Create a sample markdown file for testing."""
    content = """# Test Document

## Section One

This section covers SQL injection bypass techniques including UNION-based attacks.

## Section Two

Cross-site scripting (XSS) payloads for reflected and DOM-based attacks.

## Section Three

Linux SUID exploitation and kernel privilege escalation methods.
"""
    f = tmp_path / "test.md"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_markdown_with_code(tmp_path):
    """Markdown with code blocks containing # comments."""
    content = """# Main Title

## Real Section

Some content here.

```bash
# This is a comment inside code block
echo "hello"
# Another comment
```

## Another Section

More content after code block.
"""
    f = tmp_path / "test_code.md"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file."""
    content = "Name,Role,Score\nAlice,Admin,95\nBob,User,80\n"
    f = tmp_path / "test.csv"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_json(tmp_path):
    """Create a sample JSON file."""
    content = '{"key": "value", "items": [1, 2, 3]}'
    f = tmp_path / "test.json"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_text(tmp_path):
    """Create a sample text file."""
    content = "Line one.\nLine two.\nLine three.\n"
    f = tmp_path / "test.txt"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_python(tmp_path):
    """Create a sample Python file."""
    content = '''"""Module docstring."""

def hello(name: str) -> str:
    """Say hello."""
    return f"Hello {name}"

class Greeter:
    pass
'''
    f = tmp_path / "test.py"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_c(tmp_path):
    """Create a sample C source file."""
    content = """/**
 * Module documentation.
 */
#include <stdio.h>
#include "helper.h"

struct Config {
    int value;
};

int main(int argc, char *argv[]) {
    return 0;
}

void helper_func(int x) {
    printf("%d", x);
}
"""
    f = tmp_path / "test.c"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_cpp(tmp_path):
    """Create a sample C++ source file."""
    content = """/**
 * C++ module documentation.
 */
#include <iostream>
#include <vector>

class Engine {
public:
    void start();
};

struct Config {
    int value;
};

int main() {
    return 0;
}
"""
    f = tmp_path / "test.cpp"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_js(tmp_path):
    """Create a sample JavaScript file."""
    content = """/**
 * JavaScript module.
 */
import React from 'react';
import { useState } from 'react';

function fetchData(url) {
    return fetch(url);
}

class DataService {
    constructor() {}
}

export function processItems(items) {
    return items.map(i => i.id);
}
"""
    f = tmp_path / "test.js"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_ts(tmp_path):
    """Create a sample TypeScript file."""
    content = """/**
 * TypeScript module.
 */
import { Router } from 'express';
import type { Config } from './types';

interface AppConfig {
    port: number;
}

enum Status {
    Active,
    Inactive,
}

type Result = {
    data: string;
};

export function createRouter(): Router {
    return Router();
}

export class ApiController {
    handle() {}
}
"""
    f = tmp_path / "test.ts"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_xml(tmp_path):
    """Create a sample XML file."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>demo</artifactId>
</project>
"""
    f = tmp_path / "test.xml"
    f.write_text(content, encoding="utf-8")
    return f
