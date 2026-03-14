from __future__ import annotations

import pytest

from core.settings import EvaluationSettings, Settings, load_settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.evaluator_factory import (
    EvaluatorFactory,
    InlineEvaluatorSettings,
    UnsupportedEvaluatorError,
)


class FakeEvaluator(BaseEvaluator):
    backend_name = "fake"

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: object | None = None,
    ) -> dict[str, float]:
        return {"fake_score": float(len(retrieved_ids) + len(golden_ids))}


def _settings_with_evaluator(backends: list[str]) -> Settings:
    base = load_settings("config/settings.yaml")
    return Settings(
        llm=base.llm,
        embedding=base.embedding,
        splitter=base.splitter,
        vector_store=base.vector_store,
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=EvaluationSettings(backends=backends, golden_test_set=base.evaluation.golden_test_set),
        observability=base.observability,
    )


def test_evaluator_factory_routes_custom_backend_from_settings() -> None:
    evaluator = EvaluatorFactory.create(_settings_with_evaluator(["custom"]))

    metrics = evaluator.evaluate(
        query="what is rag",
        retrieved_ids=["doc-2", "doc-1", "doc-3"],
        golden_ids=["doc-1", "doc-4"],
    )

    assert evaluator.backend_name == "custom"
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["precision_at_k"] == pytest.approx(1.0 / 3.0)
    assert metrics["recall"] == 0.5


def test_evaluator_factory_normalizes_backend_name() -> None:
    evaluator = EvaluatorFactory.create(
        InlineEvaluatorSettings(backends=["  RAGAS  "], golden_test_set="./tests/fixtures/golden.json")
    )

    metrics = evaluator.evaluate("q", ["a"], ["a"])

    assert evaluator.backend_name == "ragas"
    assert metrics["answer_relevancy"] == 1.0
    assert metrics["faithfulness"] == 1.0


def test_evaluator_factory_supports_custom_registration() -> None:
    EvaluatorFactory.register("fake", FakeEvaluator)

    evaluator = EvaluatorFactory.create(InlineEvaluatorSettings(backends=["fake"]))

    assert isinstance(evaluator, FakeEvaluator)
    assert evaluator.evaluate("q", ["a", "b"], ["b"])["fake_score"] == 3.0


def test_evaluator_factory_rejects_unknown_backend() -> None:
    with pytest.raises(UnsupportedEvaluatorError, match="unsupported-backend"):
        EvaluatorFactory.create(InlineEvaluatorSettings(backends=["unsupported-backend"]))
