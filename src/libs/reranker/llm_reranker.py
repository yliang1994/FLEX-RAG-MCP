"""Minimal LLM reranker placeholder implementation."""

from __future__ import annotations

from core.types import QueryMatch
from libs.reranker.base_reranker import BaseReranker


class LLMReranker(BaseReranker):
    """Score candidates with a simple lexical overlap heuristic."""

    backend_name = "llm"

    def rerank(
        self,
        query: str,
        candidates: list[QueryMatch],
        trace: object | None = None,
    ) -> list[QueryMatch]:
        query_terms = {term.lower() for term in query.split() if term.strip()}
        return sorted(
            candidates,
            key=lambda candidate: (
                len(query_terms.intersection(candidate.text.lower().split())),
                candidate.score,
            ),
            reverse=True,
        )
