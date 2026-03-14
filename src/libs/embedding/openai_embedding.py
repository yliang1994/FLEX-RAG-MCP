"""Minimal OpenAI-compatible embedding provider."""

from __future__ import annotations

from typing import Any

from libs.embedding.base_embedding import BaseEmbedding


def _stable_vector(seed: str) -> list[float]:
    values = [float((sum(ord(ch) for ch in seed[i::3]) % 97) / 97.0) for i in range(3)]
    return values


class OpenAIEmbedding(BaseEmbedding):
    """Placeholder implementation for OpenAI-compatible embedding APIs."""

    provider_name = "openai"

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.max_input_length = int(kwargs.get("max_input_length", 8192))

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        _validate_texts(self.provider_name, texts, self.max_input_length)
        return [_stable_vector(text) for text in texts]


def _validate_texts(provider: str, texts: list[str], max_input_length: int) -> None:
    if not texts:
        raise ValueError(f"{provider}: validation_error: texts must not be empty")

    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"{provider}: validation_error: texts[{index}] must be str")
        if not text.strip():
            raise ValueError(f"{provider}: validation_error: texts[{index}] must not be empty")
        if len(text) > max_input_length:
            raise ValueError(
                f"{provider}: validation_error: texts[{index}] exceeds max_input_length={max_input_length}"
            )
