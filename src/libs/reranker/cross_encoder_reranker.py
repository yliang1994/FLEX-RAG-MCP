"""Minimal cross-encoder reranker placeholder implementation."""

from __future__ import annotations

from typing import Callable

from core.types import QueryMatch
from libs.reranker.base_reranker import BaseReranker


Scorer = Callable[[str, list[QueryMatch]], list[float]]


class CrossEncoderReranker(BaseReranker):
    """Scorer-driven reranker with deterministic fallback behavior."""

    backend_name = "cross-encoder"

    def __init__(self, model: str = "", **kwargs: object) -> None:
        super().__init__(model=model, **kwargs)
        self.scorer: Scorer | None = kwargs.get("scorer")
        self.last_fallback_reason: str | None = None

    def rerank(
        self,
        query: str,
        candidates: list[QueryMatch],
        trace: object | None = None,
    ) -> list[QueryMatch]:
        self.last_fallback_reason = None
        if not candidates:
            return []

        if self.scorer is None:
            return self._heuristic_rank(query, candidates)

        try:
            scores = self.scorer(query, candidates)
        except Exception as exc:  # pragma: no cover - defensive fallback signal
            self.last_fallback_reason = f"scorer_error:{exc.__class__.__name__}"
            return list(candidates)

        if len(scores) != len(candidates):
            raise ValueError("cross_encoder_reranker: schema_error: scores count must match candidates")

        reranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return [candidate for candidate, _score in reranked]

    def _heuristic_rank(self, query: str, candidates: list[QueryMatch]) -> list[QueryMatch]:
        query_length = len(query)
        return sorted(
            candidates,
            key=lambda candidate: (abs(len(candidate.text) - query_length), -candidate.score),
        )
