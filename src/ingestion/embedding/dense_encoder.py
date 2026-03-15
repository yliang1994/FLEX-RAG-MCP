"""Dense chunk encoder built on top of libs.embedding."""

from __future__ import annotations

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class DenseEncoder:
    """Encode chunk texts into dense vectors while preserving chunk metadata."""

    def __init__(self, settings: Settings, embedding: BaseEmbedding | None = None) -> None:
        self.settings = settings
        self.embedding = embedding if embedding is not None else EmbeddingFactory.create(settings)

    def encode(self, chunks: list[Chunk], trace: TraceContext | None = None) -> list[ChunkRecord]:
        if not chunks:
            if trace is not None:
                trace.record_stage("dense_encoder.encode", chunk_count=0, vector_count=0)
            return []

        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding.embed(texts, trace=trace)
        if len(vectors) != len(chunks):
            raise ValueError(
                "dense_encoder: response_error: vector count does not match chunk count"
            )

        records = [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                dense_vector=vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if trace is not None:
            trace.record_stage(
                "dense_encoder.encode",
                chunk_count=len(chunks),
                vector_count=len(vectors),
                dimensions=len(vectors[0]) if vectors else 0,
            )
        return records
