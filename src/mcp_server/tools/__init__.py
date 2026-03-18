"""MCP tool definitions."""

from __future__ import annotations

from mcp_server.protocol_handler import ToolDefinition
from mcp_server.tools.query_knowledge_hub import query_knowledge_hub


def get_registered_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="query_knowledge_hub",
            description="Run hybrid retrieval and return cited markdown snippets.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1},
                    "collection": {"type": "string"},
                },
                "required": ["query"],
            },
            handler=lambda arguments: query_knowledge_hub(
                query=str(arguments["query"]),
                top_k=int(arguments["top_k"]) if "top_k" in arguments else None,
                collection=str(arguments["collection"]) if "collection" in arguments else None,
            ),
        )
    ]
