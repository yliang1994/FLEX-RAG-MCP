"""Minimal Azure OpenAI embedding provider."""

from __future__ import annotations

from typing import Any

from libs.embedding.openai_embedding import OpenAIEmbedding, _stable_vector, _validate_texts


class AzureEmbedding(OpenAIEmbedding):
    """Placeholder implementation for Azure-hosted embedding models."""

    provider_name = "azure"

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.endpoint = str(kwargs.get("endpoint", "https://example-resource.openai.azure.com"))
        self.api_version = str(kwargs.get("api_version", "2024-02-01"))

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        _validate_texts(self.provider_name, texts, self.max_input_length)
        return [_stable_vector(f"azure:{text}") for text in texts]
