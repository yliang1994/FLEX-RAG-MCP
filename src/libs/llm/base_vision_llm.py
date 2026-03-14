"""Base abstractions for pluggable vision-capable LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VisionInput:
    """Normalized multimodal input contract for vision-capable LLMs."""

    text: str
    image: str | bytes
    mime_type: str = "image/png"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VisionResponse:
    """Provider-agnostic vision response shape."""

    content: str
    raw: dict[str, Any] = field(default_factory=dict)


class BaseVisionLLM(ABC):
    """Common interface all vision-capable LLM providers must implement."""

    provider_name = "vision-base"

    def __init__(self, model: str, api_key: str = "", **kwargs: Any) -> None:
        self.model = model
        self.api_key = api_key
        self.options = kwargs

    def preprocess_image(self, image: str | bytes) -> str | bytes:
        """Hook for future compression or format conversion before inference."""

        return image

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> VisionResponse:
        """Generate a multimodal response from text plus image input."""
