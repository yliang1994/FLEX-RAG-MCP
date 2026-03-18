from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _encode_message(payload: dict[str, object]) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def _read_message(stream) -> dict[str, object]:
    content_length: int | None = None
    while True:
        line = stream.readline()
        assert line != b"", "server closed stdout before responding"
        if line in {b"\r\n", b"\n"}:
            break

        header = line.decode("ascii").strip()
        name, _, value = header.partition(":")
        if name.lower() == "content-length":
            content_length = int(value.strip())

    assert content_length is not None, "missing Content-Length header"
    body = stream.read(content_length)
    return json.loads(body.decode("utf-8"))


@pytest.mark.integration
def test_server_initialize_over_stdio_keeps_stdout_clean() -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        process.stdin.write(
            _encode_message(
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
        )
        process.stdin.flush()

        response = _read_message(process.stdout)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1

        result = response["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "modular-rag-mcp-server"
        assert result["capabilities"]["tools"] == {}

        process.stdin.write(
            _encode_message(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                }
            )
        )
        process.stdin.flush()
        process.stdin.close()

        return_code = process.wait(timeout=5)
        assert return_code == 0

        stderr_output = process.stderr.read().decode("utf-8")
        assert "MCP server starting on stdio transport" in stderr_output
        assert "Received method=initialize" in stderr_output
        assert "Client initialization acknowledged" in stderr_output
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


@pytest.mark.integration
def test_server_query_knowledge_hub_tool_returns_friendly_empty_response() -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        assert process.stdin is not None
        assert process.stdout is not None

        process.stdin.write(
            _encode_message(
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
        )
        process.stdin.flush()
        _read_message(process.stdout)

        process.stdin.write(
            _encode_message(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "query_knowledge_hub",
                        "arguments": {
                            "query": "hybrid retrieval",
                            "top_k": 3,
                        },
                    },
                }
            )
        )
        process.stdin.flush()

        response = _read_message(process.stdout)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2

        result = response["result"]
        assert result["content"][0]["type"] == "text"
        assert "未找到相关文档" in result["content"][0]["text"]
        assert result["structuredContent"]["citations"] == []
        assert result["structuredContent"]["result_count"] == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
