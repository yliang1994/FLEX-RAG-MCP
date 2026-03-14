"""Minimal custom evaluator with lightweight ranking metrics."""

from __future__ import annotations

from libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Compute lightweight retrieval metrics without external dependencies."""

    backend_name = "custom"

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: object | None = None,
    ) -> dict[str, float]:
        golden_set = set(golden_ids)
        hits = [index for index, candidate_id in enumerate(retrieved_ids, start=1) if candidate_id in golden_set]
        hit_rate = 1.0 if hits else 0.0
        mrr = 1.0 / hits[0] if hits else 0.0
        precision_at_k = (sum(1 for candidate_id in retrieved_ids if candidate_id in golden_set) / len(retrieved_ids)) if retrieved_ids else 0.0
        recall = (sum(1 for golden_id in golden_set if golden_id in retrieved_ids) / len(golden_set)) if golden_set else 0.0

        return {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "precision_at_k": precision_at_k,
            "recall": recall,
        }
