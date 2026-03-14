"""Minimal OpenAI-compatible embedding provider."""

from __future__ import annotations

from libs.embedding.base_embedding import BaseEmbedding


def _stable_vector(seed: str) -> list[float]:
    values = [float((sum(ord(ch) for ch in seed[i::3]) % 97) / 97.0) for i in range(3)]
    return values


class OpenAIEmbedding(BaseEmbedding):
    """Placeholder implementation for OpenAI-compatible embedding APIs."""

    provider_name = "openai"

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        if not texts:
            return []
        return [_stable_vector(text) for text in texts]
