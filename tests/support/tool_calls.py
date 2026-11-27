"""One valid call for each of the 13 MCP tools, to check what a session answers to all of them."""

SOAK_FILE = "soak/notes.md"

TOOL_CALLS = {
    "search_knowledge": {"query": "soak notes"},
    "get_document": {"filepath": SOAK_FILE},
    "reindex_documents": {},
    "get_reindex_status": {},
    "list_categories": {},
    "list_documents": {},
    "get_index_stats": {},
    "add_document": {"content": "soak notes", "filepath": SOAK_FILE},
    "update_document": {"filepath": SOAK_FILE, "content": "changed soak notes"},
    "remove_document": {"filepath": SOAK_FILE},
    "add_from_url": {"url": "https://example.com/soak"},
    "search_similar": {"filepath": SOAK_FILE},
    "evaluate_retrieval": {"test_cases": '[{"query": "soak", "expected_filepath": "soak/notes.md"}]'},
}

REFUSAL_FIELDS = ("status", "message", "owner_pid", "owner_running", "hint")


def is_refusal(answer):
    """A follower's answer: an error that names the owner and points to the shared server."""
    return answer.get("status") == "error" and all(field in answer for field in REFUSAL_FIELDS)
