"""Minimal MCP server over stdio."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import IO, Any

from core.settings import Settings, load_settings
from mcp_server.protocol_handler import ProtocolHandler
from observability.logger import get_logger


@dataclass(slots=True)
class ServerContext:
    settings: Settings
    logger_name: str = "mcp_server.server"


class MCPServer:
    """Serve JSON-RPC messages over stdio using MCP framing."""

    def __init__(
        self,
        context: ServerContext,
        *,
        stdin: IO[bytes] | None = None,
        stdout: IO[bytes] | None = None,
        protocol_handler: ProtocolHandler | None = None,
    ) -> None:
        self._context = context
        self._stdin = stdin or sys.stdin.buffer
        self._stdout = stdout or sys.stdout.buffer
        self._logger = get_logger(context.logger_name)
        self._protocol_handler = protocol_handler or ProtocolHandler()
        self._running = True

    def serve_forever(self) -> int:
        self._logger.info("MCP server starting on stdio transport")
        while self._running:
            message = self._read_message()
            if message is None:
                self._logger.info("MCP server stopping after EOF")
                return 0

            response = self._handle_message(message)
            if response is None:
                continue

            self._write_message(response)
        return 0

    def _read_message(self) -> dict[str, Any] | None:
        content_length: int | None = None

        while True:
            line = self._stdin.readline()
            if line == b"":
                return None
            if line in {b"\r\n", b"\n"}:
                break

            header = line.decode("ascii").strip()
            if not header:
                continue
            name, _, value = header.partition(":")
            if name.lower() == "content-length":
                try:
                    content_length = int(value.strip())
                except ValueError as exc:
                    raise ValueError("Invalid Content-Length header") from exc

        if content_length is None:
            raise ValueError("Missing Content-Length header")

        body = self._stdin.read(content_length)
        if len(body) != content_length:
            raise ValueError("Unexpected EOF while reading message body")

        try:
            decoded = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON payload") from exc

        if not isinstance(decoded, dict):
            raise ValueError("JSON-RPC payload must be an object")
        return decoded

    def _handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        self._logger.info("Received method=%s", method)

        if method == "notifications/initialized":
            self._logger.info("Client initialization acknowledged")
        elif method == "exit":
            self._logger.info("Received exit notification")
            self._running = False

        response = self._protocol_handler.handle_request(message)
        if response is not None and "error" in response:
            self._logger.warning(
                "Responding with error code=%s for method=%s",
                response["error"]["code"],
                method,
            )
        return response

    def _write_message(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._stdout.write(header)
        self._stdout.write(body)
        self._stdout.flush()
        self._logger.info("Sent response for id=%s", payload.get("id"))


def main() -> int:
    settings = load_settings("config/settings.yaml")
    server = MCPServer(ServerContext(settings=settings))
    return server.serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
