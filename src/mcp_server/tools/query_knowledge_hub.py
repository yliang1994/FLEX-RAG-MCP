"""MCP tool for querying the local knowledge hub."""

from __future__ import annotations

from pathlib import Path

from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.reranker import Reranker
from core.response.response_builder import ResponseBuilder
from core.settings import Settings, load_settings


def query_knowledge_hub(
    query: str,
    top_k: int | None = None,
    collection: str | None = None,
    *,
    settings: Settings | None = None,
    hybrid_search: HybridSearch | None = None,
    reranker: Reranker | None = None,
    response_builder: ResponseBuilder | None = None,
) -> dict[str, object]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")

    resolved_settings = settings or load_settings(Path("config/settings.yaml"))
    resolved_top_k = top_k or resolved_settings.retrieval.top_k_final
    filters = {"collection": collection} if collection else None

    search = hybrid_search or HybridSearch(resolved_settings)
    ranker = reranker or Reranker(resolved_settings)
    builder = response_builder or ResponseBuilder()

    candidates = search.search(query, top_k=resolved_top_k, filters=filters)
    results = ranker.rerank(query, candidates)
    return builder.build(results, query=query)
