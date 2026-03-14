from __future__ import annotations

import pytest

from core.settings import RerankSettings, Settings, load_settings
from core.types import QueryMatch
from libs.reranker.base_reranker import BaseReranker
from libs.reranker.reranker_factory import (
    InlineRerankSettings,
    RerankerFactory,
    UnsupportedRerankerError,
)


class FakeReranker(BaseReranker):
    backend_name = "fake"

    def rerank(
        self,
        query: str,
        candidates: list[QueryMatch],
        trace: object | None = None,
    ) -> list[QueryMatch]:
        return list(reversed(candidates))


def _settings_with_reranker(backend: str, model: str = "", top_m: int = 10) -> Settings:
    base = load_settings("config/settings.yaml")
    return Settings(
        llm=base.llm,
        embedding=base.embedding,
        splitter=base.splitter,
        vector_store=base.vector_store,
        retrieval=base.retrieval,
        rerank=RerankSettings(backend=backend, model=model, top_m=top_m),
        evaluation=base.evaluation,
        observability=base.observability,
    )


def _sample_candidates() -> list[QueryMatch]:
    return [
        QueryMatch(id="c1", score=0.9, text="short answer", metadata={"source": "a"}),
        QueryMatch(id="c2", score=0.8, text="a much longer candidate answer", metadata={"source": "b"}),
    ]


def test_reranker_factory_routes_none_backend_from_settings() -> None:
    reranker = RerankerFactory.create(_settings_with_reranker("none"))

    candidates = _sample_candidates()
    reranked = reranker.rerank("query text", candidates)

    assert reranker.backend_name == "none"
    assert [item.id for item in reranked] == [item.id for item in candidates]


def test_reranker_factory_normalizes_backend_name() -> None:
    reranker = RerankerFactory.create(
        InlineRerankSettings(backend="  LLM  ", model="judge-model", top_m=5)
    )

    reranked = reranker.rerank("longer candidate", _sample_candidates())

    assert reranker.backend_name == "llm"
    assert reranked[0].id == "c2"


def test_reranker_factory_supports_custom_registration() -> None:
    RerankerFactory.register("fake", FakeReranker)

    reranker = RerankerFactory.create(InlineRerankSettings(backend="fake"))

    assert isinstance(reranker, FakeReranker)
    assert [item.id for item in reranker.rerank("q", _sample_candidates())] == ["c2", "c1"]


def test_reranker_factory_rejects_unknown_backend() -> None:
    with pytest.raises(UnsupportedRerankerError, match="unsupported-backend"):
        RerankerFactory.create(InlineRerankSettings(backend="unsupported-backend"))
