from __future__ import annotations

import json
import re

import pytest

from core.types import QueryMatch
from libs.reranker.llm_reranker import LLMReranker
from libs.reranker.reranker_factory import InlineRerankSettings, RerankerFactory


def _candidates() -> list[QueryMatch]:
    return [
        QueryMatch(id="c1", score=0.7, text="alpha answer", metadata={}),
        QueryMatch(id="c2", score=0.6, text="beta answer", metadata={}),
    ]


def test_llm_reranker_factory_routes_llm_backend() -> None:
    reranker = RerankerFactory.create(InlineRerankSettings(backend="llm", model="judge", top_m=5))

    assert reranker.backend_name == "llm"


def test_llm_reranker_uses_prompt_and_returns_structured_order(tmp_path) -> None:
    prompt_path = tmp_path / "rerank.txt"
    prompt_path.write_text("Rank these candidates carefully.", encoding="utf-8")

    captured: dict[str, str] = {}

    def runner(prompt: str) -> str:
        captured["prompt"] = prompt
        return json.dumps({"ranked_ids": ["c2", "c1"]})

    reranker = LLMReranker(model="judge", prompt_path=str(prompt_path), runner=runner)
    reranked = reranker.rerank("beta", _candidates())

    assert "Rank these candidates carefully." in captured["prompt"]
    assert "Query: beta" in captured["prompt"]
    assert [candidate.id for candidate in reranked] == ["c2", "c1"]


def test_llm_reranker_rejects_invalid_schema(tmp_path) -> None:
    prompt_path = tmp_path / "rerank.txt"
    prompt_path.write_text("Return ranked ids.", encoding="utf-8")

    reranker = LLMReranker(
        model="judge",
        prompt_path=str(prompt_path),
        runner=lambda prompt: json.dumps({"bad": []}),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("llm_reranker: schema_error: ranked_ids must be a list[str]"),
    ):
        reranker.rerank("beta", _candidates())


def test_llm_reranker_exposes_fallback_signal_on_runner_failure(tmp_path) -> None:
    prompt_path = tmp_path / "rerank.txt"
    prompt_path.write_text("Return ranked ids.", encoding="utf-8")

    def failing_runner(prompt: str) -> str:
        raise TimeoutError("timed out")

    reranker = LLMReranker(model="judge", prompt_path=str(prompt_path), runner=failing_runner)
    reranked = reranker.rerank("beta", _candidates())

    assert [candidate.id for candidate in reranked] == ["c1", "c2"]
    assert reranker.last_fallback_reason == "runner_error:TimeoutError"
