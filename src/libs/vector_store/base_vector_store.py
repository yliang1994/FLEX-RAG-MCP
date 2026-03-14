"""Base abstractions for pluggable vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.types import QueryMatch, VectorRecord


class BaseVectorStore(ABC):
    """Common interface all vector stores must implement."""

    backend_name = "base"

    def __init__(self, persist_path: str, **kwargs: Any) -> None:
        self.persist_path = persist_path
        self.options = kwargs

    @abstractmethod
    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> int:
        """Insert or update vector records and return the affected count."""

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[QueryMatch]:
        """Return the top-k matches for the given vector."""
