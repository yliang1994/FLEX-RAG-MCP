from __future__ import annotations

import re

import pytest

from libs.embedding.ollama_embedding import OllamaEmbedding


def test_ollama_embedding_uses_local_endpoint_transport() -> None:
    captured: dict[str, object] = {}

    def fake_transport(payload: dict[str, object]) -> dict[str, object]:
        captured.update(payload)
        return {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

    embedding = OllamaEmbedding(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
        transport=fake_transport,
    )

    vectors = embedding.embed(["hello", "world"])

    assert captured["model"] == "nomic-embed-text"
    assert captured["input"] == ["hello", "world"]
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_ollama_embedding_reports_connection_errors() -> None:
    def broken_transport(payload: dict[str, object]) -> dict[str, object]:
        raise OSError("connection refused")

    embedding = OllamaEmbedding(model="nomic-embed-text", transport=broken_transport)

    with pytest.raises(
        RuntimeError,
        match=re.escape("ollama: connection_error: failed to reach local ollama endpoint"),
    ):
        embedding.embed(["hello"])


def test_ollama_embedding_reports_timeout_errors() -> None:
    def slow_transport(payload: dict[str, object]) -> dict[str, object]:
        raise TimeoutError("timed out")

    embedding = OllamaEmbedding(model="nomic-embed-text", transport=slow_transport)

    with pytest.raises(
        RuntimeError,
        match=re.escape("ollama: timeout_error: request to local ollama endpoint timed out"),
    ):
        embedding.embed(["hello"])


def test_ollama_embedding_validates_input_and_response_shape() -> None:
    embedding = OllamaEmbedding(model="nomic-embed-text", transport=lambda payload: {"embeddings": [[0.1, 0.2]]})

    with pytest.raises(
        ValueError,
        match=re.escape("ollama: validation_error: texts must not be empty"),
    ):
        embedding.embed([])

    with pytest.raises(
        RuntimeError,
        match=re.escape("ollama: response_error: embeddings count does not match input count"),
    ):
        embedding.embed(["hello", "world"])
