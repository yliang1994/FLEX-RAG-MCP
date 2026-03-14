"""Base abstractions for pluggable text splitters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSplitter(ABC):
    """Common interface all splitter strategies must implement."""

    method_name = "base"

    def __init__(self, chunk_size: int, chunk_overlap: int = 0, **kwargs: Any) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.options = kwargs

    @abstractmethod
    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        """Split the input text into ordered chunks."""
