"""Factory for creating evaluators from settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from core.settings import EvaluationSettings, Settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.ragas_evaluator import RagasEvaluator


class UnsupportedEvaluatorError(ValueError):
    """Raised when the configured evaluator backend has no implementation."""


@dataclass(slots=True)
class InlineEvaluatorSettings:
    """Fallback settings shape for tests or local construction."""

    backends: list[str] = field(default_factory=lambda: ["custom"])
    golden_test_set: str = ""


class EvaluatorFactory:
    """Resolve evaluator backends to concrete implementations."""

    _registry: dict[str, Type[BaseEvaluator]] = {
        "custom": CustomEvaluator,
        "ragas": RagasEvaluator,
        "stub": CustomEvaluator,
    }

    @classmethod
    def create(
        cls, settings: Settings | EvaluationSettings | InlineEvaluatorSettings
    ) -> BaseEvaluator:
        evaluation_settings = settings.evaluation if isinstance(settings, Settings) else settings
        if not evaluation_settings.backends:
            raise UnsupportedEvaluatorError("No evaluator backend configured")
        backend = evaluation_settings.backends[0].strip().lower()
        evaluator_cls = cls._registry.get(backend)
        if evaluator_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise UnsupportedEvaluatorError(
                f"Unsupported evaluator backend: {evaluation_settings.backends[0]}. "
                f"Supported backends: {supported}"
            )
        return evaluator_cls(golden_test_set=evaluation_settings.golden_test_set)

    @classmethod
    def register(cls, backend: str, evaluator_cls: Type[BaseEvaluator]) -> None:
        """Allow tests or extensions to register additional evaluators."""

        cls._registry[backend.strip().lower()] = evaluator_cls
