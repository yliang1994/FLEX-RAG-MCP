"""Minimal Ollama LLM provider."""

from __future__ import annotations

from libs.llm.base_llm import BaseLLM, ChatMessage, LLMResponse


class OllamaLLM(BaseLLM):
    """Placeholder implementation for local Ollama models."""

    provider_name = "ollama"

    def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        if not messages:
            raise ValueError("messages must not be empty")
        return LLMResponse(
            content=f"[ollama:{self.model}] {messages[-1].content}",
            raw={"provider": self.provider_name, "message_count": len(messages)},
        )
