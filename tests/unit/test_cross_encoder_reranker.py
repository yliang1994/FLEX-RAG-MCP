from __future__ import annotations

import re

import pytest

from core.types import QueryMatch
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.reranker_factory import InlineRerankSettings, RerankerFactory


def _candidates() -> list[QueryMatch]:
    return [
        QueryMatch(id="c1", score=0.7, text="short answer", metadata={}),
        QueryMatch(id="c2", score=0.6, text="longer candidate answer", metadata={}),
    ]


def test_cross_encoder_factory_routes_backend_alias() -> None:
    reranker = RerankerFactory.create(
        InlineRerankSettings(backend="cross_encoder", model="bge-reranker", top_m=5)
    )

    assert reranker.backend_name == "cross-encoder"


def test_cross_encoder_reranker_uses_injected_scorer() -> None:
    def scorer(query: str, candidates: list[QueryMatch]) -> list[float]:
        return [0.2, 0.9]

    reranker = CrossEncoderReranker(model="bge-reranker", scorer=scorer)
    reranked = reranker.rerank("candidate", _candidates())

    assert [candidate.id for candidate in reranked] == ["c2", "c1"]


def test_cross_encoder_reranker_exposes_fallback_signal_on_timeout() -> None:
    def failing_scorer(query: str, candidates: list[QueryMatch]) -> list[float]:
        raise TimeoutError("timed out")

    reranker = CrossEncoderReranker(model="bge-reranker", scorer=failing_scorer)
    reranked = reranker.rerank("candidate", _candidates())

    assert [candidate.id for candidate in reranked] == ["c1", "c2"]
    assert reranker.last_fallback_reason == "scorer_error:TimeoutError"


def test_cross_encoder_reranker_rejects_invalid_score_shape() -> None:
    reranker = CrossEncoderReranker(model="bge-reranker", scorer=lambda q, c: [0.1])

    with pytest.raises(
        ValueError,
        match=re.escape("cross_encoder_reranker: schema_error: scores count must match candidates"),
    ):
        reranker.rerank("candidate", _candidates())
