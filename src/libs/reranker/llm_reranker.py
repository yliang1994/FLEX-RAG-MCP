"""Minimal LLM reranker placeholder implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from core.types import QueryMatch
from libs.reranker.base_reranker import BaseReranker


LLMRunner = Callable[[str], str | dict[str, Any]]


class LLMReranker(BaseReranker):
    """Prompt-driven reranker with structured ranked-id output."""

    backend_name = "llm"

    def __init__(self, model: str = "", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.prompt_path = Path(kwargs.get("prompt_path", "config/prompts/rerank.txt"))
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8").strip()
        self.runner: LLMRunner | None = kwargs.get("runner")
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

        prompt = self._build_prompt(query, candidates)
        if self.runner is None:
            return self._heuristic_rank(query, candidates)

        try:
            response = self.runner(prompt)
        except Exception as exc:  # pragma: no cover - defensive fallback signal
            return self._fallback_rank(candidates, reason=f"runner_error:{exc.__class__.__name__}")

        ranked_ids = self._parse_ranked_ids(response)
        return self._apply_ranked_ids(candidates, ranked_ids)

    def _build_prompt(self, query: str, candidates: list[QueryMatch]) -> str:
        serialized_candidates = [
            {"id": candidate.id, "text": candidate.text, "score": candidate.score}
            for candidate in candidates
        ]
        return (
            f"{self.prompt_template}\n\n"
            f"Query: {query}\n"
            f"Candidates: {json.dumps(serialized_candidates, ensure_ascii=True)}\n"
            'Return JSON only, in the form {"ranked_ids": ["id1", "id2"]}.'
        )

    def _parse_ranked_ids(self, response: str | dict[str, Any]) -> list[str]:
        payload = json.loads(response) if isinstance(response, str) else response
        ranked_ids = payload.get("ranked_ids")
        if not isinstance(ranked_ids, list) or not all(isinstance(item, str) for item in ranked_ids):
            raise ValueError("llm_reranker: schema_error: ranked_ids must be a list[str]")
        return ranked_ids

    def _apply_ranked_ids(self, candidates: list[QueryMatch], ranked_ids: list[str]) -> list[QueryMatch]:
        candidate_map = {candidate.id: candidate for candidate in candidates}
        if set(ranked_ids) != set(candidate_map):
            raise ValueError("llm_reranker: schema_error: ranked_ids must match candidate ids exactly")
        return [candidate_map[candidate_id] for candidate_id in ranked_ids]

    def _fallback_rank(self, candidates: list[QueryMatch], reason: str) -> list[QueryMatch]:
        self.last_fallback_reason = reason
        return list(candidates)

    def _heuristic_rank(self, query: str, candidates: list[QueryMatch]) -> list[QueryMatch]:
        query_terms = {term.lower() for term in query.split() if term.strip()}
        return sorted(
            candidates,
            key=lambda candidate: (
                len(query_terms.intersection(candidate.text.lower().split())),
                candidate.score,
            ),
            reverse=True,
        )
