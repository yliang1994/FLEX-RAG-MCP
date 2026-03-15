from __future__ import annotations

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.embedding.sparse_encoder import SparseEncoder


def test_sparse_encoder_outputs_term_weights() -> None:
    encoder = SparseEncoder()
    chunks = [Chunk(id="chunk-1", text="Apple apple banana", metadata={"page": 1})]

    records = encoder.encode(chunks)

    assert records[0].sparse_vector == {"apple": 1.0, "banana": 0.5}
    assert records[0].metadata["page"] == 1


def test_sparse_encoder_returns_empty_vector_for_empty_text() -> None:
    encoder = SparseEncoder()
    chunks = [Chunk(id="chunk-1", text="   ", metadata={})]

    records = encoder.encode(chunks)

    assert records[0].sparse_vector == {}


def test_sparse_encoder_keeps_order_for_multiple_chunks() -> None:
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="chunk-1", text="alpha beta", metadata={}),
        Chunk(id="chunk-2", text="gamma gamma delta", metadata={}),
    ]

    records = encoder.encode(chunks)

    assert [record.id for record in records] == ["chunk-1", "chunk-2"]
    assert records[1].sparse_vector == {"delta": 0.5, "gamma": 1.0}


def test_sparse_encoder_records_trace_summary() -> None:
    encoder = SparseEncoder()
    trace = TraceContext()

    encoder.encode([Chunk(id="chunk-1", text="alpha", metadata={})], trace=trace)

    assert trace.stages[-1].name == "sparse_encoder.encode"
    assert trace.stages[-1].details["non_empty_vectors"] == 1
