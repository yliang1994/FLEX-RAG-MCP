from __future__ import annotations

import re

import pytest

from core.settings import EmbeddingSettings
from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


@pytest.mark.parametrize(
    ("provider", "model", "texts", "expected_dimension"),
    [
        ("openai", "text-embedding-3-small", ["hello", "world"], 3),
        ("azure", "text-embedding-ada-002", ["azure text"], 3),
    ],
)
def test_openai_and_azure_embeddings_route_and_embed(
    provider: str,
    model: str,
    texts: list[str],
    expected_dimension: int,
) -> None:
    embedding = EmbeddingFactory.create(EmbeddingSettings(provider=provider, model=model))

    vectors = embedding.embed(texts)

    assert embedding.provider_name == provider
    assert len(vectors) == len(texts)
    assert all(len(vector) == expected_dimension for vector in vectors)


def test_azure_embedding_exposes_azure_specific_configuration() -> None:
    embedding = AzureEmbedding(
        model="text-embedding-ada-002",
        endpoint="https://rag-test.openai.azure.com",
        api_version="2024-10-21",
    )

    vectors = embedding.embed(["hello azure"])

    assert embedding.endpoint == "https://rag-test.openai.azure.com"
    assert embedding.api_version == "2024-10-21"
    assert len(vectors[0]) == 3


@pytest.mark.parametrize(
    ("provider", "texts", "expected_error"),
    [
        ("openai", [], "openai: validation_error: texts must not be empty"),
        ("azure", ["   "], "azure: validation_error: texts[0] must not be empty"),
    ],
)
def test_openai_and_azure_embeddings_validate_text_inputs(
    provider: str,
    texts: list[str],
    expected_error: str,
) -> None:
    embedding = EmbeddingFactory.create(EmbeddingSettings(provider=provider, model="test-model"))

    with pytest.raises(ValueError, match=re.escape(expected_error)):
        embedding.embed(texts)


def test_openai_embedding_rejects_overlong_text() -> None:
    embedding = EmbeddingFactory.create(EmbeddingSettings(provider="openai", model="text-embedding-3-small"))

    with pytest.raises(
        ValueError,
        match=re.escape("openai: validation_error: texts[0] exceeds max_input_length=8192"),
    ):
        embedding.embed(["a" * 8193])


def test_openai_embedding_rejects_non_string_inputs() -> None:
    embedding = EmbeddingFactory.create(EmbeddingSettings(provider="openai", model="text-embedding-3-small"))

    with pytest.raises(
        TypeError,
        match=re.escape("openai: validation_error: texts[0] must be str"),
    ):
        embedding.embed([123])  # type: ignore[list-item]
