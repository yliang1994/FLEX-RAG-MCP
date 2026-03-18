from __future__ import annotations

from core.response.response_builder import ResponseBuilder
from core.types import RetrievalResult


def test_response_builder_formats_markdown_and_structured_citations() -> None:
    builder = ResponseBuilder()

    response = builder.build(
        [
            RetrievalResult(
                chunk_id="chunk-001",
                score=0.98,
                text="Hybrid retrieval combines dense and sparse signals.",
                metadata={"source_path": "/tmp/doc.pdf", "page": 3},
            ),
            RetrievalResult(
                chunk_id="chunk-002",
                score=0.76,
                text="Rerank improves top-k precision after coarse recall.",
                metadata={"source": "notes.md"},
            ),
        ],
        query="how does hybrid retrieval work",
    )

    assert response["content"][0]["type"] == "text"
    assert "[1] Hybrid retrieval combines dense and sparse signals." in response["content"][0]["text"]
    assert "[2] Rerank improves top-k precision after coarse recall." in response["content"][0]["text"]

    citations = response["structuredContent"]["citations"]
    assert citations == [
        {
            "index": 1,
            "source": "/tmp/doc.pdf",
            "page": 3,
            "chunk_id": "chunk-001",
            "score": 0.98,
        },
        {
            "index": 2,
            "source": "notes.md",
            "page": None,
            "chunk_id": "chunk-002",
            "score": 0.76,
        },
    ]


def test_response_builder_returns_friendly_message_when_no_results() -> None:
    builder = ResponseBuilder()

    response = builder.build([], query="missing")

    assert response == {
        "content": [
            {
                "type": "text",
                "text": "未找到相关文档，请先运行 ingest.py 摄取数据，或检查查询条件。",
            }
        ],
        "structuredContent": {
            "query": "missing",
            "citations": [],
            "result_count": 0,
        },
    }
