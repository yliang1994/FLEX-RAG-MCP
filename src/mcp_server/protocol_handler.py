"""JSON-RPC protocol handling for the MCP server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"


class ProtocolError(Exception):
    """Base protocol error with JSON-RPC code mapping."""

    code = -32603
    message = "Internal error"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)
        self.rpc_message = message or self.message


class InvalidRequestError(ProtocolError):
    code = -32600
    message = "Invalid Request"


class MethodNotFoundError(ProtocolError):
    code = -32601
    message = "Method not found"


class InvalidParamsError(ProtocolError):
    code = -32602
    message = "Invalid params"


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class ProtocolHandler:
    """Handle the JSON-RPC methods required by the MCP server."""

    def __init__(
        self,
        *,
        tools: list[ToolDefinition] | None = None,
        server_name: str = SERVER_NAME,
        server_version: str = SERVER_VERSION,
        protocol_version: str = MCP_PROTOCOL_VERSION,
    ) -> None:
        self._tools = {tool.name: tool for tool in (tools or [])}
        self._server_name = server_name
        self._server_version = server_version
        self._protocol_version = protocol_version

    def handle_request(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        try:
            request_id = request.get("id")
            self._validate_envelope(request)
            method = request["method"]
            params = request.get("params", {})

            if method == "initialize":
                if not isinstance(params, Mapping):
                    raise InvalidParamsError("initialize params must be an object")
                result = self.handle_initialize(dict(params))
            elif method == "tools/list":
                self._ensure_params_object(params, method)
                result = self.handle_tools_list()
            elif method == "tools/call":
                if not isinstance(params, Mapping):
                    raise InvalidParamsError("tools/call params must be an object")
                result = self.handle_tools_call(dict(params))
            elif method == "notifications/initialized":
                return None
            elif method == "exit":
                return None
            else:
                raise MethodNotFoundError(f"Method not found: {method}")

            if request_id is None:
                return None
            return {
                "jsonrpc": JSONRPC_VERSION,
                "id": request_id,
                "result": result,
            }
        except ProtocolError as exc:
            if request.get("id") is None:
                return None
            return self._build_error_response(request.get("id"), exc.code, exc.rpc_message)
        except Exception:
            if request.get("id") is None:
                return None
            return self._build_error_response(request.get("id"), -32603, "Internal error")

    def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        client_version = params.get("protocolVersion")
        if client_version is not None and not isinstance(client_version, str):
            raise InvalidParamsError("protocolVersion must be a string")

        return {
            "protocolVersion": self._protocol_version,
            "serverInfo": {
                "name": self._server_name,
                "version": self._server_version,
            },
            "capabilities": {
                "tools": {},
            },
        }

    def handle_tools_list(self) -> dict[str, Any]:
        return {
            "tools": [tool.to_schema() for tool in self._tools.values()],
        }

    def handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise InvalidParamsError("tools/call requires a non-empty string name")

        arguments = params.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise InvalidParamsError("tools/call arguments must be an object")

        tool = self._tools.get(name)
        if tool is None:
            raise MethodNotFoundError(f"Tool not found: {name}")

        return tool.handler(dict(arguments))

    def _validate_envelope(self, request: Mapping[str, Any]) -> None:
        if request.get("jsonrpc") != JSONRPC_VERSION:
            raise InvalidRequestError("jsonrpc must be '2.0'")

        method = request.get("method")
        if not isinstance(method, str) or not method:
            raise InvalidRequestError("method must be a non-empty string")

    def _ensure_params_object(self, params: Any, method: str) -> None:
        if params in ({}, None):
            return
        if not isinstance(params, Mapping):
            raise InvalidParamsError(f"{method} params must be an object")

    def _build_error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }
