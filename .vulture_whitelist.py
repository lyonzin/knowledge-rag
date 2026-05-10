"""Whitelist for false-positive findings from vulture.

Each "unused" item below is referenced externally (signal handler
signature, ChromaDB callback, watchdog event, MCP framework dispatch,
etc.) so vulture cannot see the consumer. Vulture executes this file
to exercise the names — we deliberately reference them so the dead-code
report stays clean.

Run with:
    vulture mcp_server/ scripts/ .vulture_whitelist.py --min-confidence 80
"""

# Signal handler signature requires (signum, frame); vulture sees `frame` as unused
# but Python's signal module passes both arguments unconditionally.
_dummy_frame = object()


def _signal_handler_stub(signum, frame):  # noqa: ARG001
    return signum, frame
