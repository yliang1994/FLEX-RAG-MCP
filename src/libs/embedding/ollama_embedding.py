"""Minimal Ollama embedding provider."""

from __future__ import annotations

import json
from typing import Any, Callable
from urllib import error, request

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.openai_embedding import _stable_vector, _validate_texts


Transport = Callable[[dict[str, Any]], dict[str, Any]]


class OllamaEmbedding(BaseEmbedding):
    """Placeholder implementation for local Ollama embedding models."""

    provider_name = "ollama"

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.base_url = str(kwargs.get("base_url", "http://localhost:11434"))
        self.timeout = float(kwargs.get("timeout", 10.0))
        self.max_input_length = int(kwargs.get("max_input_length", 8192))
        self._transport: Transport = kwargs.get("transport", self._default_transport)

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        _validate_texts(self.provider_name, texts, self.max_input_length)
        payload = {"model": self.model, "input": texts}

        try:
            response_payload = self._transport(payload)
        except TimeoutError as exc:
            raise RuntimeError("ollama: timeout_error: request to local ollama endpoint timed out") from exc
        except OSError as exc:
            raise RuntimeError("ollama: connection_error: failed to reach local ollama endpoint") from exc

        try:
            embeddings = response_payload["embeddings"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("ollama: response_error: invalid response payload") from exc

        if len(embeddings) != len(texts):
            raise RuntimeError("ollama: response_error: embeddings count does not match input count")
        return embeddings

    def _default_transport(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"embeddings": [_stable_vector(f"ollama:{text}") for text in payload["input"]]}

    def _http_transport(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        endpoint = f"{self.base_url.rstrip('/')}/api/embed"
        req = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, TimeoutError):
                raise TimeoutError from exc
            raise OSError from exc
