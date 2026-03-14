"""Core shared types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class VectorRecord:
    """Minimal vector-store record contract used across ingestion and retrieval."""

    id: str
    text: str
    vector: list[float]
    metadata: dict[str, str | int | float | bool | list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class QueryMatch:
    """Normalized vector-store query result."""

    id: str
    score: float
    text: str
    metadata: dict[str, str | int | float | bool | list[str]] = field(default_factory=dict)
