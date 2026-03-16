from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import Settings, load_settings
from core.trace.trace_context import TraceContext
from core.types import QueryMatch, RetrievalResult
from core.query_engine.dense_retriever import DenseRetriever
from libs.embedding.base_embedding import BaseEmbedding
from libs.vector_store.base_vector_store import BaseVectorStore


class FakeEmbedding(BaseEmbedding):
    provider_name = "fake"

    def __init__(self) -> None:
        super().__init__(model="fake-embedding")
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.5, 0.25]]


class FakeVectorStore(BaseVectorStore):
    backend_name = "fake"

    def __init__(self) -> None:
        super().__init__(persist_path="./tmp/fake")
        self.last_query: tuple[list[float], int, dict[str, object] | None] | None = None

    def upsert(self, records, trace: object | None = None) -> int:  # type: ignore[override]
        return len(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
        trace: object | None = None,
    ) -> list[QueryMatch]:
        self.last_query = (vector, top_k, filters)
        return [
            QueryMatch(
                id="chunk-1",
                score=0.9,
                text="dense result",
                metadata={"collection": "kb", "page": 1},
            )
        ]


class InvalidEmbedding(BaseEmbedding):
    provider_name = "invalid"

    def __init__(self) -> None:
        super().__init__(model="bad")

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        return [[0.1], [0.2]]


def _load_base_settings(tmp_path: Path) -> Settings:
    settings = load_settings("config/settings.yaml")
    settings.vector_store.persist_path = str(tmp_path / "chroma")
    return settings


def test_dense_retriever_embeds_query_and_shapes_results(tmp_path: Path) -> None:
    embedding = FakeEmbedding()
    vector_store = FakeVectorStore()
    retriever = DenseRetriever(
        _load_base_settings(tmp_path),
        embedding_client=embedding,
        vector_store=vector_store,
    )
    trace = TraceContext()

    results = retriever.retrieve(" hybrid retrieval ", top_k=3, filters={"collection": "kb"}, trace=trace)

    assert embedding.calls == [["hybrid retrieval"]]
    assert vector_store.last_query == ([0.5, 0.25], 3, {"collection": "kb"})
    assert results == [
        RetrievalResult(
            chunk_id="chunk-1",
            score=0.9,
            text="dense result",
            metadata={"collection": "kb", "page": 1},
        )
    ]
    assert trace.stages[-1].name == "dense_retriever.retrieve"
    assert trace.stages[-1].details["result_count"] == 1


def test_dense_retriever_rejects_empty_query(tmp_path: Path) -> None:
    retriever = DenseRetriever(
        _load_base_settings(tmp_path),
        embedding_client=FakeEmbedding(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve("   ", top_k=1)


def test_dense_retriever_rejects_invalid_embedding_response(tmp_path: Path) -> None:
    retriever = DenseRetriever(
        _load_base_settings(tmp_path),
        embedding_client=InvalidEmbedding(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError, match="expected exactly one query vector"):
        retriever.retrieve("dense", top_k=2)
