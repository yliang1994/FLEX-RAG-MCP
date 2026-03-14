from __future__ import annotations

import pytest

from core.settings import LLMSettings, Settings, load_settings
from libs.llm.base_llm import BaseLLM, ChatMessage, LLMResponse
from libs.llm.llm_factory import LLMFactory, UnsupportedLLMProviderError


class FakeLLM(BaseLLM):
    provider_name = "fake"

    def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        return LLMResponse(content=messages[-1].content, raw={"provider": self.provider_name})


def test_llm_factory_routes_from_settings_object() -> None:
    settings = Settings(
        llm=LLMSettings(provider="openai", model="gpt-test", api_key="secret"),
        embedding=load_settings("config/settings.yaml").embedding,
        splitter=load_settings("config/settings.yaml").splitter,
        vector_store=load_settings("config/settings.yaml").vector_store,
        retrieval=load_settings("config/settings.yaml").retrieval,
        rerank=load_settings("config/settings.yaml").rerank,
        evaluation=load_settings("config/settings.yaml").evaluation,
        observability=load_settings("config/settings.yaml").observability,
    )

    llm = LLMFactory.create(settings)

    assert llm.provider_name == "openai"
    assert llm.model == "gpt-test"
    assert llm.api_key == "secret"


def test_llm_factory_normalizes_provider_name() -> None:
    llm = LLMFactory.create(LLMSettings(provider="  OLLAMA  ", model="llama3"))

    assert llm.provider_name == "ollama"
    response = llm.chat([ChatMessage(role="user", content="ping")])
    assert response.content == "[ollama:llama3] ping"


def test_llm_factory_supports_custom_registration() -> None:
    LLMFactory.register("fake", FakeLLM)

    llm = LLMFactory.create(LLMSettings(provider="fake", model="unit-test"))

    assert isinstance(llm, FakeLLM)
    assert llm.chat([ChatMessage(role="user", content="hello")]).raw["provider"] == "fake"


def test_llm_factory_rejects_unknown_provider() -> None:
    with pytest.raises(UnsupportedLLMProviderError, match="unsupported-provider"):
        LLMFactory.create(LLMSettings(provider="unsupported-provider", model="x"))
