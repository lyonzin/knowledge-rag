"""Tool descriptions that agents rely on (GH #258).

MCP clients show a tool's description as its documentation. Categories come from
``category_mappings`` and ``keyword_routes`` in config.yaml, so a fixed list
offers agents categories that may not exist and hides the configured ones; the
descriptions point to list_categories() instead. A 0.0–1.0 ``min_score`` reads
like an absolute relevance scale, so its description must say that scores are
normalized per query.
"""

import asyncio
import re

# Optional quotes or backticks around each name, so ``security``, "ctf" also counts as a list.
QUOTES = "[`'\"]*"


def _listed_tools():
    from mcp_server import server

    return asyncio.run(server.mcp.list_tools())


def test_given_the_built_in_categories_when_listing_tools_then_no_description_enumerates_them(monkeypatch):
    import mcp_server.config as config_module

    # Listed first: importing the server under the patch below would build its global config from it.
    tools = _listed_tools()
    monkeypatch.setattr(config_module, "_yaml", {})  # built-in defaults, whatever config.yaml says
    defaults = config_module.Config()
    names = "|".join(map(re.escape, {"general", *defaults.keyword_routes, *defaults.category_mappings.values()}))
    # Two category names in a row, such as "security, ctf", is a hard-coded list.
    enumeration = re.compile(rf"\b(?:{names}){QUOTES},\s*{QUOTES}(?:{names})\b", re.IGNORECASE)

    found = {tool.name: enumeration.findall(tool.description) for tool in tools}

    assert {name: lists for name, lists in found.items() if lists} == {}


def test_given_a_category_parameter_when_listing_tools_then_its_description_points_to_list_categories():
    with_category = [tool for tool in _listed_tools() if "category" in tool.input_schema.get("properties", {})]

    missing = [tool.name for tool in with_category if "list_categories()" not in tool.description]

    assert with_category, "no tool has a category parameter, so this test checks nothing"
    assert missing == []


def test_given_search_knowledge_when_listing_tools_then_min_score_says_scores_are_normalized_per_query():
    description = next(tool.description for tool in _listed_tools() if tool.name == "search_knowledge")

    min_score_text = description.split("min_score:", 1)[1]

    assert "min-max normalized per query" in min_score_text
