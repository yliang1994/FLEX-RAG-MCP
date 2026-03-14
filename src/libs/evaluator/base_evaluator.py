"""Base abstractions for pluggable evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """Common interface all evaluators must implement."""

    backend_name = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.options = kwargs

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
    ) -> dict[str, float]:
        """Return normalized evaluation metrics for one query."""
