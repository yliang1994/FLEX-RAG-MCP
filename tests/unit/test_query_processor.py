from __future__ import annotations

from dataclasses import asdict

from core import ProcessedQuery
from core.query_engine.query_processor import QueryProcessor


def test_processed_query_is_serializable() -> None:
    payload = asdict(
        ProcessedQuery(
            query="find docs",
            normalized_query="find docs",
            keywords=["find", "docs"],
            filters={"collection": "demo"},
        )
    )

    assert payload == {
        "query": "find docs",
        "normalized_query": "find docs",
        "keywords": ["find", "docs"],
        "filters": {"collection": "demo"},
    }


def test_query_processor_extracts_keywords_and_deduplicates_tokens() -> None:
    processed = QueryProcessor().process("How to use hybrid retrieval with retrieval metrics?")

    assert processed.normalized_query == "How to use hybrid retrieval with retrieval metrics?"
    assert processed.filters == {}
    assert processed.keywords == ["use", "hybrid", "retrieval", "metrics"]


def test_query_processor_extracts_supported_filters_from_inline_query() -> None:
    processed = QueryProcessor().process(
        "collection:kb doc_type=pdf language:zh 检索增强生成 方案"
    )

    assert processed.filters == {
        "collection": "kb",
        "doc_type": "pdf",
        "language": "zh",
    }
    assert processed.normalized_query == "检索增强生成 方案"
    assert processed.keywords == ["检索增强生成", "方案"]


def test_query_processor_falls_back_to_normalized_query_when_no_keyword_match() -> None:
    processed = QueryProcessor().process("C")

    assert processed.keywords == ["c"]
    assert processed.filters == {}
