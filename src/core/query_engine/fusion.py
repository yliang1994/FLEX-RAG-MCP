"""Fusion algorithms for combining dense and sparse retrieval rankings."""

from __future__ import annotations

from dataclasses import replace

from core.types import RetrievalResult


class RRFusion:
    """Reciprocal Rank Fusion for deterministic hybrid retrieval merging."""

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        scored: dict[str, RetrievalResult] = {}
        rrf_scores: dict[str, float] = {}

        for results in (dense_results, sparse_results):
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk_id
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.k + rank))
                if chunk_id not in scored:
                    scored[chunk_id] = result

        ranked = sorted(
            scored.values(),
            key=lambda item: (-rrf_scores[item.chunk_id], item.chunk_id),
        )
        fused = [replace(result, score=rrf_scores[result.chunk_id]) for result in ranked]
        return fused if top_k is None else fused[:top_k]
