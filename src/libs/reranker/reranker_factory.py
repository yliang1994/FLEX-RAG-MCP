"""Factory for creating rerankers from settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from core.settings import RerankSettings, Settings
from libs.reranker.base_reranker import BaseReranker, NoneReranker
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.llm_reranker import LLMReranker


class UnsupportedRerankerError(ValueError):
    """Raised when the configured reranker backend has no implementation."""


@dataclass(slots=True)
class InlineRerankSettings:
    """Fallback settings shape for tests or local construction."""

    backend: str
    model: str = ""
    top_m: int = 0


class RerankerFactory:
    """Resolve reranker backends to concrete implementations."""

    _registry: dict[str, Type[BaseReranker]] = {
        "cross-encoder": CrossEncoderReranker,
        "llm": LLMReranker,
        "none": NoneReranker,
        "stub": NoneReranker,
    }

    @classmethod
    def create(cls, settings: Settings | RerankSettings | InlineRerankSettings) -> BaseReranker:
        rerank_settings = settings.rerank if isinstance(settings, Settings) else settings
        backend = rerank_settings.backend.strip().lower()
        reranker_cls = cls._registry.get(backend)
        if reranker_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise UnsupportedRerankerError(
                f"Unsupported reranker backend: {rerank_settings.backend}. "
                f"Supported backends: {supported}"
            )
        return reranker_cls(model=rerank_settings.model, top_m=rerank_settings.top_m)

    @classmethod
    def register(cls, backend: str, reranker_cls: Type[BaseReranker]) -> None:
        """Allow tests or extensions to register additional rerankers."""

        cls._registry[backend.strip().lower()] = reranker_cls
