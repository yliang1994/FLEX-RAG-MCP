from __future__ import annotations

import pytest

from core.settings import EmbeddingSettings, Settings, load_settings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import (
    EmbeddingFactory,
    UnsupportedEmbeddingProviderError,
)


class FakeEmbedding(BaseEmbedding):
    provider_name = "fake"

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


def _settings_with_embedding(provider: str, model: str) -> Settings:
    base = load_settings("config/settings.yaml")
    return Settings(
        llm=base.llm,
        embedding=EmbeddingSettings(provider=provider, model=model),
        splitter=base.splitter,
        vector_store=base.vector_store,
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=base.evaluation,
        observability=base.observability,
    )


def test_embedding_factory_routes_from_settings_object() -> None:
    embedding = EmbeddingFactory.create(_settings_with_embedding("openai", "text-embedding-3-small"))

    assert embedding.provider_name == "openai"
    assert embedding.model == "text-embedding-3-small"
    assert embedding.embed(["abc"]) == [[0.0, 0.010309278350515464, 0.020618556701030927]]


def test_embedding_factory_normalizes_provider_name() -> None:
    embedding = EmbeddingFactory.create(
        EmbeddingSettings(provider="  OLLAMA  ", model="nomic-embed-text")
    )

    assert embedding.provider_name == "ollama"
    first = embedding.embed(["ping"])
    second = embedding.embed(["ping"])
    assert first == second
    assert len(first) == 1
    assert len(first[0]) == 3


def test_embedding_factory_supports_custom_registration() -> None:
    EmbeddingFactory.register("fake", FakeEmbedding)

    embedding = EmbeddingFactory.create(EmbeddingSettings(provider="fake", model="unit-test"))

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.embed(["hello", "world"]) == [[0.0, 5.0], [1.0, 5.0]]


def test_embedding_factory_rejects_unknown_provider() -> None:
    with pytest.raises(UnsupportedEmbeddingProviderError, match="unsupported-provider"):
        EmbeddingFactory.create(EmbeddingSettings(provider="unsupported-provider", model="x"))
