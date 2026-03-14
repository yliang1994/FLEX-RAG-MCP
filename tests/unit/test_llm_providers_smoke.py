from __future__ import annotations

import re

import pytest

from core.settings import LLMSettings
from libs.llm.base_llm import ChatMessage
from libs.llm.llm_factory import LLMFactory


@pytest.mark.parametrize(
    ("provider", "model", "expected_prefix"),
    [
        ("openai", "gpt-4o-mini", "[openai:gpt-4o-mini]"),
        ("azure", "gpt-4o", "[azure:gpt-4o]"),
        ("deepseek", "deepseek-chat", "[deepseek:deepseek-chat]"),
    ],
)
def test_openai_compatible_llm_providers_route_and_chat(
    provider: str,
    model: str,
    expected_prefix: str,
) -> None:
    llm = LLMFactory.create(LLMSettings(provider=provider, model=model, api_key="secret"))

    response = llm.chat([ChatMessage(role="user", content="hello world")])

    assert response.content == f"{expected_prefix} hello world"
    assert response.raw["provider"] == provider
    assert response.raw["message_count"] == 1


@pytest.mark.parametrize(
    ("provider", "messages", "expected_error"),
    [
        ("openai", [], "openai: validation_error: messages must not be empty"),
        (
            "azure",
            [ChatMessage(role="", content="hello")],
            "azure: validation_error: message[0].role must not be empty",
        ),
        (
            "deepseek",
            [ChatMessage(role="user", content="   ")],
            "deepseek: validation_error: message[0].content must not be empty",
        ),
    ],
)
def test_openai_compatible_llm_providers_validate_message_shape(
    provider: str,
    messages: list[ChatMessage],
    expected_error: str,
) -> None:
    llm = LLMFactory.create(LLMSettings(provider=provider, model="test-model", api_key="secret"))

    with pytest.raises(ValueError, match=re.escape(expected_error)):
        llm.chat(messages)


def test_openai_compatible_llm_provider_rejects_non_chat_message() -> None:
    llm = LLMFactory.create(LLMSettings(provider="openai", model="gpt-4o-mini", api_key="secret"))

    with pytest.raises(
        TypeError,
        match=re.escape("openai: validation_error: message[0] must be ChatMessage"),
    ):
        llm.chat([object()])
