from __future__ import annotations

import pytest

from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding.batch_processor import BatchProcessor


class FakeDenseEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, chunks, trace=None):
        self.calls.append([chunk.id for chunk in chunks])
        return [
            ChunkRecord(id=chunk.id, text=chunk.text, metadata=dict(chunk.metadata), dense_vector=[float(index)])
            for index, chunk in enumerate(chunks, start=1)
        ]


class FakeSparseEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, chunks, trace=None):
        self.calls.append([chunk.id for chunk in chunks])
        return [
            ChunkRecord(id=chunk.id, text=chunk.text, metadata=dict(chunk.metadata), sparse_vector={chunk.id: 1.0})
            for chunk in chunks
        ]


def make_chunks(count: int) -> list[Chunk]:
    return [Chunk(id=f"chunk-{index}", text=f"text-{index}", metadata={"order": index}) for index in range(count)]


def test_batch_processor_splits_five_chunks_into_three_batches() -> None:
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()
    processor = BatchProcessor(batch_size=2, dense_encoder=dense, sparse_encoder=sparse)

    records = processor.process(make_chunks(5))

    assert dense.calls == [["chunk-0", "chunk-1"], ["chunk-2", "chunk-3"], ["chunk-4"]]
    assert sparse.calls == dense.calls
    assert [record.id for record in records] == [f"chunk-{index}" for index in range(5)]


def test_batch_processor_merges_dense_and_sparse_outputs() -> None:
    processor = BatchProcessor(batch_size=2, dense_encoder=FakeDenseEncoder(), sparse_encoder=FakeSparseEncoder())

    records = processor.process(make_chunks(2))

    assert records[0].dense_vector == [1.0]
    assert records[0].sparse_vector == {"chunk-0": 1.0}
    assert records[1].dense_vector == [2.0]


def test_batch_processor_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        BatchProcessor(batch_size=0)


def test_batch_processor_records_trace_summary() -> None:
    trace = TraceContext()
    processor = BatchProcessor(batch_size=2, dense_encoder=FakeDenseEncoder(), sparse_encoder=FakeSparseEncoder())

    processor.process(make_chunks(3), trace=trace)

    assert [stage.name for stage in trace.stages if stage.name == "batch_processor.batch"] == [
        "batch_processor.batch",
        "batch_processor.batch",
    ]
    assert trace.stages[-1].name == "batch_processor.process"
    assert trace.stages[-1].details["batch_count"] == 2
