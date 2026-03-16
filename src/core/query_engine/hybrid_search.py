"""Hybrid retrieval orchestrator."""

from __future__ import annotations

from typing import Any

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import RRFusion
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult


class HybridSearch:
    """Compose query processing, dense retrieval, sparse retrieval, and fusion."""

    def __init__(
        self,
        settings: Settings,
        query_processor: QueryProcessor | None = None,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        fusion: RRFusion | None = None,
    ) -> None:
        self.settings = settings
        self.query_processor = query_processor if query_processor is not None else QueryProcessor()
        self.dense_retriever = (
            dense_retriever if dense_retriever is not None else DenseRetriever(settings)
        )
        self.sparse_retriever = (
            sparse_retriever if sparse_retriever is not None else SparseRetriever(settings)
        )
        self.fusion = fusion if fusion is not None else RRFusion()

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        processed = self.query_processor.process(query)
        combined_filters = dict(processed.filters)
        combined_filters.update(filters or {})

        dense_results: list[RetrievalResult] = []
        sparse_results: list[RetrievalResult] = []
        dense_error = False
        sparse_error = False

        try:
            dense_results = self.dense_retriever.retrieve(
                processed.normalized_query or processed.query,
                top_k=top_k,
                filters=combined_filters or None,
                trace=trace,
            )
        except Exception:
            dense_error = True

        try:
            sparse_results = self.sparse_retriever.retrieve(
                processed.keywords,
                top_k=top_k,
                trace=trace,
            )
        except Exception:
            sparse_error = True

        if dense_results and sparse_results:
            candidates = self.fusion.fuse(dense_results, sparse_results, top_k=top_k)
        elif dense_results:
            candidates = dense_results
        else:
            candidates = sparse_results

        filtered = self._apply_metadata_filters(candidates, combined_filters)
        results = filtered[:top_k]

        if trace is not None:
            trace.record_stage(
                "hybrid_search.search",
                query=processed.query,
                normalized_query=processed.normalized_query,
                keywords=processed.keywords,
                filters=combined_filters,
                dense_count=len(dense_results),
                sparse_count=len(sparse_results),
                result_count=len(results),
                dense_fallback=dense_error,
                sparse_fallback=sparse_error,
            )

        return results

    def _apply_metadata_filters(
        self,
        candidates: list[RetrievalResult],
        filters: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        if not filters:
            return candidates
        return [
            candidate
            for candidate in candidates
            if all(candidate.metadata.get(key) == expected for key, expected in filters.items())
        ]
