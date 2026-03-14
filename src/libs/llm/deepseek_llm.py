"""Minimal DeepSeek LLM provider."""

from __future__ import annotations

from libs.llm.base_llm import BaseLLM, ChatMessage, LLMResponse
from libs.llm.openai_llm import _validate_messages


class DeepSeekLLM(BaseLLM):
    """Placeholder implementation for DeepSeek-hosted models."""

    provider_name = "deepseek"

    def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        _validate_messages(self.provider_name, messages)
        return LLMResponse(
            content=f"[deepseek:{self.model}] {messages[-1].content}",
            raw={"provider": self.provider_name, "message_count": len(messages)},
        )
