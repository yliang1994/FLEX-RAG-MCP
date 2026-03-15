"""Core shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MetadataValue = str | int | float | bool | None | list[Any] | dict[str, Any]
MetadataDict = dict[str, MetadataValue]


@dataclass(slots=True)
class ImageRef:
    """Structured reference to an extracted image inside a source document."""

    id: str
    path: str
    page: int | None = None
    text_offset: int = 0
    text_length: int = 0
    position: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    """Normalized source document shared across ingestion and retrieval flows."""

    id: str
    text: str
    metadata: MetadataDict = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """Chunked document payload with stable source offsets."""

    id: str
    text: str
    metadata: MetadataDict = field(default_factory=dict)
    start_offset: int = 0
    end_offset: int = 0
    source_ref: str | None = None


@dataclass(slots=True)
class ChunkRecord:
    """Storage-oriented chunk payload used by embedding and upsert stages."""

    id: str
    text: str
    metadata: MetadataDict = field(default_factory=dict)
    dense_vector: list[float] | None = None
    sparse_vector: dict[str, float] | None = None


@dataclass(slots=True)
class VectorRecord:
    """Minimal vector-store record contract used across ingestion and retrieval."""

    id: str
    text: str
    vector: list[float]
    metadata: MetadataDict = field(default_factory=dict)


@dataclass(slots=True)
class QueryMatch:
    """Normalized vector-store query result."""

    id: str
    score: float
    text: str
    metadata: MetadataDict = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """Canonical retrieval output shared by dense/sparse/hybrid search."""

    chunk_id: str
    score: float
    text: str
    metadata: MetadataDict = field(default_factory=dict)
