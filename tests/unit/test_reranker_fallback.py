from __future__ import annotations

from pathlib import Path

from core.query_engine.reranker import Reranker
from core.settings import Settings, load_settings
from core.trace.trace_context import TraceContext
from core.types import QueryMatch, RetrievalResult
from libs.reranker.base_reranker import BaseReranker


class ReversingBackend(BaseReranker):
    backend_name = "reverse"

    def rerank(self, query: str, candidates: list[QueryMatch], trace=None):  # type: ignore[override]
        return list(reversed(candidates))


class FailingBackend(BaseReranker):
    backend_name = "failing"

    def rerank(self, query: str, candidates: list[QueryMatch], trace=None):  # type: ignore[override]
        raise RuntimeError("backend down")


class InvalidBackend(BaseReranker):
    backend_name = "invalid"

    def rerank(self, query: str, candidates: list[QueryMatch], trace=None):  # type: ignore[override]
        return [QueryMatch(id="missing", score=1.0, text="missing", metadata={})]


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings("config/settings.yaml")
    settings.vector_store.persist_path = str(tmp_path / "chroma")
    return settings


def _candidates() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chunk-1",
            score=0.7,
            text="first",
            metadata={"source": "docs/a.pdf"},
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            score=0.6,
            text="second",
            metadata={"source": "docs/b.pdf"},
        ),
    ]


def test_reranker_reorders_candidates_when_backend_succeeds(tmp_path: Path) -> None:
    reranker = Reranker(_settings(tmp_path), backend=ReversingBackend())

    results = reranker.rerank("query", _candidates())

    assert [item.chunk_id for item in results] == ["chunk-2", "chunk-1"]
    assert reranker.last_fallback is False
    assert reranker.last_fallback_reason is None


def test_reranker_returns_original_order_when_backend_raises(tmp_path: Path) -> None:
    trace = TraceContext()
    reranker = Reranker(_settings(tmp_path), backend=FailingBackend())

    results = reranker.rerank("query", _candidates(), trace=trace)

    assert [item.chunk_id for item in results] == ["chunk-1", "chunk-2"]
    assert reranker.last_fallback is True
    assert reranker.last_fallback_reason == "backend_error:RuntimeError"
    assert trace.stages[-1].name == "reranker.rerank"
    assert trace.stages[-1].details["fallback"] is True


def test_reranker_returns_original_order_when_backend_returns_unknown_ids(tmp_path: Path) -> None:
    reranker = Reranker(_settings(tmp_path), backend=InvalidBackend())

    results = reranker.rerank("query", _candidates())

    assert [item.chunk_id for item in results] == ["chunk-1", "chunk-2"]
    assert reranker.last_fallback is True
    assert reranker.last_fallback_reason == "backend_error:invalid_candidate_ids"
