"""Minimal cross-encoder reranker placeholder implementation."""

from __future__ import annotations

from core.types import QueryMatch
from libs.reranker.base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """Sort candidates by text-length proximity to the query length."""

    backend_name = "cross-encoder"

    def rerank(
        self,
        query: str,
        candidates: list[QueryMatch],
        trace: object | None = None,
    ) -> list[QueryMatch]:
        query_length = len(query)
        return sorted(
            candidates,
            key=lambda candidate: (abs(len(candidate.text) - query_length), -candidate.score),
        )
