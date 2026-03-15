"""Base abstractions for ingestion-time chunk transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.trace.trace_context import TraceContext
from core.types import Chunk


class BaseTransform(ABC):
    """Common contract for chunk-level transforms in the ingestion pipeline."""

    @abstractmethod
    def transform(self, chunks: list[Chunk], trace: TraceContext | None = None) -> list[Chunk]:
        """Return a transformed list while preserving pipeline continuity."""
