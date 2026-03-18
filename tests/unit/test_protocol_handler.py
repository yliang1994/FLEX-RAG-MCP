from __future__ import annotations

from mcp_server.protocol_handler import ProtocolHandler, ToolDefinition
from mcp_server.tools import get_registered_tools


def _echo_tool(arguments: dict[str, object]) -> dict[str, object]:
    text = arguments.get("text", "")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return {
        "content": [
            {
                "type": "text",
                "text": text.upper(),
            }
        ],
        "structuredContent": {
            "echo": text.upper(),
        },
    }


def build_handler() -> ProtocolHandler:
    return ProtocolHandler(
        tools=[
            ToolDefinition(
                name="echo",
                description="Uppercase the provided text.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                },
                handler=_echo_tool,
            )
        ]
    )


def test_initialize_returns_server_info_and_capabilities() -> None:
    handler = build_handler()

    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0.0.0"},
            },
        }
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "modular-rag-mcp-server",
                "version": "0.1.0",
            },
            "capabilities": {
                "tools": {},
            },
        },
    }


def test_tools_list_returns_registered_schema() -> None:
    handler = build_handler()

    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
    )

    assert response is not None
    tools = response["result"]["tools"]
    assert tools == [
        {
            "name": "echo",
            "description": "Uppercase the provided text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            },
        }
    ]


def test_tools_call_routes_to_tool_handler() -> None:
    handler = build_handler()

    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "echo",
                "arguments": {"text": "hello"},
            },
        }
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 3,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": "HELLO",
                }
            ],
            "structuredContent": {
                "echo": "HELLO",
            },
        },
    }


def test_unknown_method_returns_method_not_found() -> None:
    handler = build_handler()

    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "not/a/real/method",
        }
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 4,
        "error": {
            "code": -32601,
            "message": "Method not found: not/a/real/method",
        },
    }


def test_invalid_params_return_invalid_params_error() -> None:
    handler = build_handler()

    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "echo",
                "arguments": "bad-shape",
            },
        }
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 5,
        "error": {
            "code": -32602,
            "message": "tools/call arguments must be an object",
        },
    }


def test_internal_tool_error_is_hidden_behind_internal_error() -> None:
    handler = build_handler()

    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "echo",
                "arguments": {"text": 42},
            },
        }
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 6,
        "error": {
            "code": -32603,
            "message": "Internal error",
        },
    }


def test_tools_list_includes_registered_mcp_tools() -> None:
    handler = ProtocolHandler(tools=get_registered_tools())

    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/list",
            "params": {},
        }
    )

    assert response is not None
    tool_names = [tool["name"] for tool in response["result"]["tools"]]
    assert "query_knowledge_hub" in tool_names
    assert "list_collections" in tool_names
    assert "get_document_summary" in tool_names
