"""MCP tool definitions."""

from __future__ import annotations

from mcp_server.protocol_handler import InvalidParamsError, ToolDefinition
from mcp_server.tools.get_document_summary import get_document_summary
from mcp_server.tools.list_collections import list_collections
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
        ),
        ToolDefinition(
            name="list_collections",
            description="List available local document collections.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=lambda arguments: list_collections(),
        ),
        ToolDefinition(
            name="get_document_summary",
            description="Return title, summary, and tags for a stored document.",
            input_schema={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                },
                "required": ["doc_id"],
            },
            handler=lambda arguments: _handle_get_document_summary(arguments),
        ),
    ]


def _handle_get_document_summary(arguments: dict[str, object]) -> dict[str, object]:
    try:
        return get_document_summary(doc_id=str(arguments["doc_id"]))
    except KeyError as exc:
        raise InvalidParamsError(f"missing required argument: {exc.args[0]}") from exc
    except ValueError as exc:
        raise InvalidParamsError(str(exc)) from exc
