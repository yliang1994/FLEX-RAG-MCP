"""Core reranker orchestration with graceful fallback semantics."""

from __future__ import annotations

from time import perf_counter

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import QueryMatch, RetrievalResult
from libs.reranker.base_reranker import BaseReranker
from libs.reranker.reranker_factory import RerankerFactory


class Reranker:
    """Adapt libs-level rerankers to the core retrieval result contract."""

    def __init__(self, settings: Settings, backend: BaseReranker | None = None) -> None:
        self.settings = settings
        self.backend = backend if backend is not None else RerankerFactory.create(settings)
        self.last_fallback = False
        self.last_fallback_reason: str | None = None

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        self.last_fallback = False
        self.last_fallback_reason = None

        if not candidates:
            return []

        query_matches = [
            QueryMatch(
                id=candidate.chunk_id,
                score=candidate.score,
                text=candidate.text,
                metadata=dict(candidate.metadata),
            )
            for candidate in candidates
        ]

        started_at = perf_counter()
        try:
            reranked = self.backend.rerank(query, query_matches, trace=trace)
        except Exception as exc:
            self.last_fallback = True
            self.last_fallback_reason = f"backend_error:{exc.__class__.__name__}"
            reranked = query_matches

        result_by_id = {candidate.chunk_id: candidate for candidate in candidates}
        ordered_ids = [match.id for match in reranked]
        if set(ordered_ids) != set(result_by_id):
            self.last_fallback = True
            self.last_fallback_reason = "backend_error:invalid_candidate_ids"
            reranked_results = list(candidates)
        else:
            reranked_results = [result_by_id[chunk_id] for chunk_id in ordered_ids]

        if trace is not None:
            trace.record_stage(
                "rerank",
                elapsed_ms=round((perf_counter() - started_at) * 1000, 3),
                method=getattr(self.backend, "backend_name", "unknown"),
                query=query,
                candidate_count=len(candidates),
                result_count=len(reranked_results),
                backend=getattr(self.backend, "backend_name", "unknown"),
                fallback=self.last_fallback,
                fallback_reason=self.last_fallback_reason,
            )

        return reranked_results
