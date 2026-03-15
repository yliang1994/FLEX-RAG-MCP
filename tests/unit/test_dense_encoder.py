from __future__ import annotations

import pytest

from core.settings import (
    ChunkRefinerSettings,
    EmbeddingSettings,
    EvaluationSettings,
    ImageCaptionerSettings,
    IngestionSettings,
    LLMSettings,
    MetadataEnricherSettings,
    ObservabilitySettings,
    RetrievalSettings,
    RerankSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.embedding.dense_encoder import DenseEncoder
from libs.embedding.base_embedding import BaseEmbedding


class FakeEmbedding(BaseEmbedding):
    provider_name = "fake"

    def __init__(self, vectors: list[list[float]]) -> None:
        super().__init__(model="fake-model")
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], trace=None):  # type: ignore[override]
        self.calls.append(list(texts))
        return self.vectors


def make_settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="stub", model="stub-llm"),
        embedding=EmbeddingSettings(provider="stub", model="stub-embedding"),
        splitter=SplitterSettings(method="recursive", chunk_size=500, chunk_overlap=50),
        vector_store=VectorStoreSettings(backend="chroma", persist_path="./data/db/chroma"),
        retrieval=RetrievalSettings(
            sparse_backend="bm25",
            fusion_algorithm="rrf",
            top_k_dense=20,
            top_k_sparse=20,
            top_k_final=10,
        ),
        rerank=RerankSettings(backend="none", model="", top_m=30),
        evaluation=EvaluationSettings(
            backends=["custom"], golden_test_set="./tests/fixtures/golden_test_set.json"
        ),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
        ingestion=IngestionSettings(
            chunk_refiner=ChunkRefinerSettings(use_llm=False),
            metadata_enricher=MetadataEnricherSettings(use_llm=False),
            image_captioner=ImageCaptionerSettings(enabled=False),
        ),
    )


def test_dense_encoder_returns_chunk_records_with_vectors() -> None:
    embedding = FakeEmbedding(vectors=[[0.1, 0.2], [0.3, 0.4]])
    encoder = DenseEncoder(make_settings(), embedding=embedding)
    chunks = [
        Chunk(id="chunk-1", text="alpha", metadata={"source": "a"}),
        Chunk(id="chunk-2", text="beta", metadata={"source": "b"}),
    ]

    records = encoder.encode(chunks)

    assert [record.id for record in records] == ["chunk-1", "chunk-2"]
    assert [record.dense_vector for record in records] == [[0.1, 0.2], [0.3, 0.4]]
    assert records[0].metadata["source"] == "a"
    assert embedding.calls == [["alpha", "beta"]]


def test_dense_encoder_returns_empty_list_for_empty_input() -> None:
    encoder = DenseEncoder(make_settings(), embedding=FakeEmbedding(vectors=[]))

    assert encoder.encode([]) == []


def test_dense_encoder_rejects_mismatched_vector_count() -> None:
    encoder = DenseEncoder(make_settings(), embedding=FakeEmbedding(vectors=[[0.1, 0.2]]))
    chunks = [Chunk(id="chunk-1", text="alpha", metadata={}), Chunk(id="chunk-2", text="beta", metadata={})]

    with pytest.raises(ValueError, match="vector count"):
        encoder.encode(chunks)


def test_dense_encoder_records_trace_summary() -> None:
    trace = TraceContext()
    encoder = DenseEncoder(make_settings(), embedding=FakeEmbedding(vectors=[[0.1, 0.2, 0.3]]))
    chunks = [Chunk(id="chunk-1", text="alpha", metadata={})]

    encoder.encode(chunks, trace=trace)

    assert trace.stages[-1].name == "dense_encoder.encode"
    assert trace.stages[-1].details["dimensions"] == 3
