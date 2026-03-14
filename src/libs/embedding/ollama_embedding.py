"""Minimal Ollama embedding provider."""

from __future__ import annotations

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.openai_embedding import _stable_vector


class OllamaEmbedding(BaseEmbedding):
    """Placeholder implementation for local Ollama embedding models."""

    provider_name = "ollama"

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        if not texts:
            return []
        return [_stable_vector(f"ollama:{text}") for text in texts]
