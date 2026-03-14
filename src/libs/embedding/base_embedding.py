"""Base abstractions for pluggable embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EmbeddingResult:
    """Provider-agnostic embedding payload."""

    vectors: list[list[float]]
    raw: dict[str, Any] = field(default_factory=dict)


class BaseEmbedding(ABC):
    """Common interface all embedding providers must implement."""

    provider_name = "base"

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.options = kwargs

    @abstractmethod
    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        """Encode text batches into deterministic float vectors."""
