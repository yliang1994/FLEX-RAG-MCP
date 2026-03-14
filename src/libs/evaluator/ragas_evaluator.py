"""Minimal Ragas evaluator placeholder implementation."""

from __future__ import annotations

from libs.evaluator.base_evaluator import BaseEvaluator


class RagasEvaluator(BaseEvaluator):
    """Placeholder evaluator that keeps the future backend contract stable."""

    backend_name = "ragas"

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: object | None = None,
    ) -> dict[str, float]:
        overlap = 1.0 if set(retrieved_ids).intersection(golden_ids) else 0.0
        return {
            "answer_relevancy": overlap,
            "faithfulness": overlap,
        }
