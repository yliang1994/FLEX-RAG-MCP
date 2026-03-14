"""Factory for creating embedding providers from settings."""

from __future__ import annotations

from typing import Type

from core.settings import EmbeddingSettings, Settings
from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.ollama_embedding import OllamaEmbedding
from libs.embedding.openai_embedding import OpenAIEmbedding


class UnsupportedEmbeddingProviderError(ValueError):
    """Raised when the configured embedding provider has no implementation."""


class EmbeddingFactory:
    """Resolve provider names to concrete embedding implementations."""

    _registry: dict[str, Type[BaseEmbedding]] = {
        "azure": AzureEmbedding,
        "ollama": OllamaEmbedding,
        "openai": OpenAIEmbedding,
        "stub": OpenAIEmbedding,
    }

    @classmethod
    def create(cls, settings: Settings | EmbeddingSettings) -> BaseEmbedding:
        embedding_settings = settings.embedding if isinstance(settings, Settings) else settings
        provider = embedding_settings.provider.strip().lower()
        provider_cls = cls._registry.get(provider)
        if provider_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise UnsupportedEmbeddingProviderError(
                f"Unsupported embedding provider: {embedding_settings.provider}. "
                f"Supported providers: {supported}"
            )
        return provider_cls(model=embedding_settings.model)

    @classmethod
    def register(cls, provider: str, provider_cls: Type[BaseEmbedding]) -> None:
        """Allow tests or extensions to register additional providers."""

        cls._registry[provider.strip().lower()] = provider_cls
