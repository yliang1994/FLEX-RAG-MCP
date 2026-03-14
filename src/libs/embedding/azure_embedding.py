"""Minimal Azure OpenAI embedding provider."""

from __future__ import annotations

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.openai_embedding import _stable_vector


class AzureEmbedding(BaseEmbedding):
    """Placeholder implementation for Azure-hosted embedding models."""

    provider_name = "azure"

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        if not texts:
            return []
        return [_stable_vector(f"azure:{text}") for text in texts]
