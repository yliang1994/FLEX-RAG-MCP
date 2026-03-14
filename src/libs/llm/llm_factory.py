"""Factory for creating LLM providers from settings."""

from __future__ import annotations

from typing import Type

from core.settings import LLMSettings, Settings
from libs.llm.azure_llm import AzureLLM
from libs.llm.base_llm import BaseLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.ollama_llm import OllamaLLM
from libs.llm.openai_llm import OpenAILLM


class UnsupportedLLMProviderError(ValueError):
    """Raised when the configured LLM provider has no implementation."""


class LLMFactory:
    """Resolve provider names to concrete LLM implementations."""

    _registry: dict[str, Type[BaseLLM]] = {
        "azure": AzureLLM,
        "deepseek": DeepSeekLLM,
        "ollama": OllamaLLM,
        "openai": OpenAILLM,
        "stub": OpenAILLM,
    }

    @classmethod
    def create(cls, settings: Settings | LLMSettings) -> BaseLLM:
        llm_settings = settings.llm if isinstance(settings, Settings) else settings
        provider = llm_settings.provider.strip().lower()
        provider_cls = cls._registry.get(provider)
        if provider_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise UnsupportedLLMProviderError(
                f"Unsupported llm provider: {llm_settings.provider}. "
                f"Supported providers: {supported}"
            )
        return provider_cls(model=llm_settings.model, api_key=llm_settings.api_key)

    @classmethod
    def register(cls, provider: str, provider_cls: Type[BaseLLM]) -> None:
        """Allow tests or extensions to register additional providers."""

        cls._registry[provider.strip().lower()] = provider_cls
