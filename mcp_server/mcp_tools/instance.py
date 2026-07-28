"""
╭─╴ FASTMCP SINGLETON ╶──────────────────────────────────────────╮
│                                                                │
│   Owns the single ``FastMCP`` instance shared by ``server.py`` │
│   and the ``@mcp.tool()`` handlers in ``tools.py``.            │
│                                                                │
│   Kept in its own module so both server.py and tools.py can    │
│   import it without a load-order cycle.                        │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``
"""

from mcp.server.fastmcp import FastMCP

from ..config import config

mcp = FastMCP(
    "knowledge-rag",
    host=config.server_host,
    port=config.server_port,
)
