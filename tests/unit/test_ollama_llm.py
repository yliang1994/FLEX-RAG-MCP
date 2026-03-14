from __future__ import annotations

import pytest

from libs.llm.base_llm import ChatMessage
from libs.llm.ollama_llm import OllamaLLM


def test_ollama_llm_uses_local_endpoint_transport() -> None:
    captured: dict[str, object] = {}

    def fake_transport(payload: dict[str, object]) -> dict[str, object]:
        captured.update(payload)
        return {"message": {"content": "mocked ollama reply"}}

    llm = OllamaLLM(
        model="llama3.2",
        base_url="http://localhost:11434",
        transport=fake_transport,
    )

    response = llm.chat([ChatMessage(role="user", content="hello")])

    assert captured["model"] == "llama3.2"
    assert captured["stream"] is False
    assert response.content == "mocked ollama reply"
    assert response.raw["base_url"] == "http://localhost:11434"


def test_ollama_llm_reports_connection_errors_without_leaking_secrets() -> None:
    def broken_transport(payload: dict[str, object]) -> dict[str, object]:
        raise OSError("connection refused")

    llm = OllamaLLM(
        model="llama3.2",
        api_key="super-secret",
        transport=broken_transport,
    )

    with pytest.raises(RuntimeError, match="ollama: connection_error: failed to reach local ollama endpoint") as exc:
        llm.chat([ChatMessage(role="user", content="hello")])

    assert "super-secret" not in str(exc.value)


def test_ollama_llm_reports_timeout_errors() -> None:
    def slow_transport(payload: dict[str, object]) -> dict[str, object]:
        raise TimeoutError("timed out")

    llm = OllamaLLM(model="llama3.2", transport=slow_transport)

    with pytest.raises(RuntimeError, match="ollama: timeout_error: request to local ollama endpoint timed out"):
        llm.chat([ChatMessage(role="user", content="hello")])


def test_ollama_llm_validates_response_payload() -> None:
    llm = OllamaLLM(model="llama3.2", transport=lambda payload: {"unexpected": True})

    with pytest.raises(RuntimeError, match="ollama: response_error: invalid response payload"):
        llm.chat([ChatMessage(role="user", content="hello")])
