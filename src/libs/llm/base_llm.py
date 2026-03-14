"""Base abstractions for pluggable LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChatMessage:
    """Normalized chat message payload used across providers."""

    role: str
    content: str


@dataclass(slots=True)
class LLMResponse:
    """Minimal provider-agnostic response shape."""

    content: str
    raw: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """Common interface all LLM providers must implement."""

    provider_name = "base"

    def __init__(self, model: str, api_key: str = "", **kwargs: Any) -> None:
        self.model = model
        self.api_key = api_key
        self.options = kwargs

    @abstractmethod
    def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Generate a completion from normalized chat messages."""
