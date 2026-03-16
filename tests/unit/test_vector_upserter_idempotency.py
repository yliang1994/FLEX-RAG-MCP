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
from core.types import ChunkRecord, VectorRecord
from ingestion.storage.vector_upserter import VectorUpserter
from libs.vector_store.base_vector_store import BaseVectorStore


class FakeVectorStore(BaseVectorStore):
    backend_name = "fake"

    def __init__(self) -> None:
        super().__init__(persist_path="/tmp/fake")
        self.calls: list[list[VectorRecord]] = []

    def upsert(self, records, trace=None):  # type: ignore[override]
        self.calls.append(records)
        return len(records)

    def query(self, vector, top_k, filters=None, trace=None):  # type: ignore[override]
        raise NotImplementedError

    def get_by_ids(self, ids, trace=None):  # type: ignore[override]
        raise NotImplementedError


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


def make_record(text: str = "alpha", chunk_index: int = 0) -> ChunkRecord:
    return ChunkRecord(
        id="chunk-1",
        text=text,
        metadata={"source_path": "/docs/a.pdf", "chunk_index": chunk_index},
        dense_vector=[0.1, 0.2, 0.3],
    )


def test_vector_upserter_generates_same_id_for_same_content() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(make_settings(), store=store)

    first_ids = upserter.upsert([make_record()])
    second_ids = upserter.upsert([make_record()])

    assert first_ids == second_ids


def test_vector_upserter_changes_id_when_content_changes() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(make_settings(), store=store)

    first_ids = upserter.upsert([make_record(text="alpha")])
    second_ids = upserter.upsert([make_record(text="beta")])

    assert first_ids != second_ids


def test_vector_upserter_preserves_batch_order() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(make_settings(), store=store)

    ids = upserter.upsert([make_record(text="alpha", chunk_index=0), make_record(text="beta", chunk_index=1)])

    assert len(ids) == 2
    assert [record.metadata["original_chunk_id"] for record in store.calls[0]] == ["chunk-1", "chunk-1"]
    assert [record.text for record in store.calls[0]] == ["alpha", "beta"]


def test_vector_upserter_rejects_missing_dense_vector() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(make_settings(), store=store)
    record = ChunkRecord(id="chunk-1", text="alpha", metadata={}, dense_vector=None)

    with pytest.raises(ValueError, match="missing dense_vector"):
        upserter.upsert([record])


def test_vector_upserter_records_trace_summary() -> None:
    trace = TraceContext()
    store = FakeVectorStore()
    upserter = VectorUpserter(make_settings(), store=store)

    ids = upserter.upsert([make_record()], trace=trace)

    assert trace.stages[-1].name == "vector_upserter.upsert"
    assert trace.stages[-1].details["ids"] == ids
