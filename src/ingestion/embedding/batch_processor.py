"""Batch-oriented orchestration for dense and sparse chunk encoding."""

from __future__ import annotations

from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder


class BatchProcessor:
    """Split chunks into stable batches and merge encoder outputs."""

    def __init__(
        self,
        batch_size: int = 16,
        dense_encoder: DenseEncoder | None = None,
        sparse_encoder: SparseEncoder | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.batch_size = batch_size
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder

    def process(self, chunks: list[Chunk], trace: TraceContext | None = None) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []

        for batch_index, batch in enumerate(self._iter_batches(chunks)):
            dense_records = self.dense_encoder.encode(batch, trace=trace) if self.dense_encoder else []
            sparse_records = self.sparse_encoder.encode(batch, trace=trace) if self.sparse_encoder else []
            records.extend(self._merge_records(batch, dense_records, sparse_records))
            if trace is not None:
                trace.record_stage(
                    "batch_processor.batch",
                    batch_index=batch_index,
                    batch_size=len(batch),
                )

        if trace is not None:
            trace.record_stage(
                "batch_processor.process",
                chunk_count=len(chunks),
                batch_count=len(records) // self.batch_size + (1 if len(records) % self.batch_size else 0) if records else 0,
            )
        return records

    def _iter_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        return [chunks[index : index + self.batch_size] for index in range(0, len(chunks), self.batch_size)]

    def _merge_records(
        self,
        chunks: list[Chunk],
        dense_records: list[ChunkRecord],
        sparse_records: list[ChunkRecord],
    ) -> list[ChunkRecord]:
        if dense_records and len(dense_records) != len(chunks):
            raise ValueError("batch_processor: dense record count does not match batch size")
        if sparse_records and len(sparse_records) != len(chunks):
            raise ValueError("batch_processor: sparse record count does not match batch size")

        dense_by_id = {record.id: record for record in dense_records}
        sparse_by_id = {record.id: record for record in sparse_records}
        merged: list[ChunkRecord] = []
        for chunk in chunks:
            dense = dense_by_id.get(chunk.id)
            sparse = sparse_by_id.get(chunk.id)
            merged.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    dense_vector=dense.dense_vector if dense else None,
                    sparse_vector=sparse.sparse_vector if sparse else None,
                )
            )
        return merged
