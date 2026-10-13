"""Optional MCP tool arguments that default to None must accept an explicit null.

Some MCP clients send ``"arg": null`` for an optional argument they leave unset.
A parameter annotated ``str = None`` advertises ``"default": null`` but its type
excludes null, so the SDK's argument model rejects the call before the tool runs.
"""

import asyncio


def test_null_default_tool_args_accept_explicit_null():
    from mcp_server import server

    rejected = []
    checked = 0
    for tool in asyncio.run(server.mcp.list_tools()):
        schema = tool.input_schema
        props = schema.get("properties", {})
        required = schema.get("required", [])
        arg_model = server.mcp._tool_manager.get_tool(tool.name).fn_metadata.arg_model
        for name, prop in props.items():
            if "default" not in prop or prop["default"] is not None:
                continue
            checked += 1
            args = {r: "x" if props[r].get("type") == "string" else 1 for r in required}
            args[name] = None
            try:
                arg_model.model_validate(args)
            except Exception:
                rejected.append(f"{tool.name}.{name}")

    assert checked >= 5
    assert rejected == []
