"""Minimal OpenAI-compatible LLM provider."""

from __future__ import annotations

from libs.llm.base_llm import BaseLLM, ChatMessage, LLMResponse


class OpenAILLM(BaseLLM):
    """Placeholder implementation for OpenAI-compatible backends."""

    provider_name = "openai"

    def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        _validate_messages(self.provider_name, messages)
        return LLMResponse(
            content=f"[openai:{self.model}] {messages[-1].content}",
            raw={"provider": self.provider_name, "message_count": len(messages)},
        )


def _validate_messages(provider: str, messages: list[ChatMessage]) -> None:
    if not messages:
        raise ValueError(f"{provider}: validation_error: messages must not be empty")

    for index, message in enumerate(messages):
        if not isinstance(message, ChatMessage):
            raise TypeError(f"{provider}: validation_error: message[{index}] must be ChatMessage")
        if not message.role.strip():
            raise ValueError(f"{provider}: validation_error: message[{index}].role must not be empty")
        if not message.content.strip():
            raise ValueError(f"{provider}: validation_error: message[{index}].content must not be empty")
