"""Base abstractions for pluggable rerankers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.types import QueryMatch


class BaseReranker(ABC):
    """Common interface all rerankers must implement."""

    backend_name = "base"

    def __init__(self, model: str = "", **kwargs: Any) -> None:
        self.model = model
        self.options = kwargs

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[QueryMatch],
        trace: Any | None = None,
    ) -> list[QueryMatch]:
        """Rerank retrieval candidates for a given query."""


class NoneReranker(BaseReranker):
    """Fallback reranker that preserves the original order."""

    backend_name = "none"

    def rerank(
        self,
        query: str,
        candidates: list[QueryMatch],
        trace: Any | None = None,
    ) -> list[QueryMatch]:
        return list(candidates)
