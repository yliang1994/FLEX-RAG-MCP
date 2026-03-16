from __future__ import annotations

from pathlib import Path

from core.query_engine.fusion import RRFusion
from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.query_processor import QueryProcessor
from core.settings import Settings, load_settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult


class StubDenseRetriever:
    def __init__(self, results: list[RetrievalResult] | None = None, *, raises: bool = False) -> None:
        self.results = results or []
        self.raises = raises
        self.calls: list[tuple[str, int, dict[str, object] | None]] = []

    def retrieve(self, query: str, top_k: int, filters=None, trace=None):  # type: ignore[override]
        self.calls.append((query, top_k, filters))
        if self.raises:
            raise RuntimeError("dense unavailable")
        return self.results[:top_k]


class StubSparseRetriever:
    def __init__(self, results: list[RetrievalResult] | None = None, *, raises: bool = False) -> None:
        self.results = results or []
        self.raises = raises
        self.calls: list[tuple[list[str], int]] = []

    def retrieve(self, keywords: list[str], top_k: int, trace=None):  # type: ignore[override]
        self.calls.append((keywords, top_k))
        if self.raises:
            raise RuntimeError("sparse unavailable")
        return self.results[:top_k]


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings("config/settings.yaml")
    settings.vector_store.persist_path = str(tmp_path / "chroma")
    return settings


def _result(chunk_id: str, score: float, collection: str, page: int) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"text:{chunk_id}",
        metadata={"collection": collection, "page": page},
    )


def test_hybrid_search_fuses_dense_and_sparse_results_with_filters(tmp_path: Path) -> None:
    dense = StubDenseRetriever(
        [
            _result("chunk-1", 0.9, "kb", 1),
            _result("chunk-2", 0.8, "other", 2),
        ]
    )
    sparse = StubSparseRetriever(
        [
            _result("chunk-2", 2.0, "other", 2),
            _result("chunk-1", 1.0, "kb", 1),
        ]
    )
    search = HybridSearch(
        _settings(tmp_path),
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=RRFusion(k=60),
    )
    trace = TraceContext()

    results = search.search("collection:kb hybrid retrieval", top_k=3, trace=trace)

    assert [item.chunk_id for item in results] == ["chunk-1"]
    assert dense.calls == [("hybrid retrieval", 3, {"collection": "kb"})]
    assert sparse.calls == [(["hybrid", "retrieval"], 3)]
    assert trace.stages[-1].name == "hybrid_search.search"
    assert trace.stages[-1].details["result_count"] == 1


def test_hybrid_search_falls_back_to_sparse_when_dense_fails(tmp_path: Path) -> None:
    search = HybridSearch(
        _settings(tmp_path),
        query_processor=QueryProcessor(),
        dense_retriever=StubDenseRetriever(raises=True),
        sparse_retriever=StubSparseRetriever([_result("chunk-3", 0.5, "kb", 3)]),
        fusion=RRFusion(),
    )

    results = search.search("hybrid fallback", top_k=2)

    assert [item.chunk_id for item in results] == ["chunk-3"]


def test_hybrid_search_falls_back_to_dense_when_sparse_fails_and_explicit_filters_override(tmp_path: Path) -> None:
    dense = StubDenseRetriever([_result("chunk-4", 0.7, "docs", 4)])
    search = HybridSearch(
        _settings(tmp_path),
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=StubSparseRetriever(raises=True),
        fusion=RRFusion(),
    )

    results = search.search("collection:kb api docs", top_k=2, filters={"collection": "docs"})

    assert [item.chunk_id for item in results] == ["chunk-4"]
    assert dense.calls == [("api docs", 2, {"collection": "docs"})]
