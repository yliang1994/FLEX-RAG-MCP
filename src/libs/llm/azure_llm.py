"""Minimal Azure OpenAI LLM provider."""

from __future__ import annotations

from libs.llm.base_llm import BaseLLM, ChatMessage, LLMResponse


class AzureLLM(BaseLLM):
    """Placeholder implementation for Azure-hosted chat models."""

    provider_name = "azure"

    def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        if not messages:
            raise ValueError("messages must not be empty")
        return LLMResponse(
            content=f"[azure:{self.model}] {messages[-1].content}",
            raw={"provider": self.provider_name, "message_count": len(messages)},
        )
