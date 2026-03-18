"""Build MCP-compatible tool responses from retrieval results."""

from __future__ import annotations

from core.response.citation_generator import CitationGenerator
from core.types import RetrievalResult


class ResponseBuilder:
    """Format retrieval results into MCP text and structured content."""

    def __init__(self, citation_generator: CitationGenerator | None = None) -> None:
        self.citation_generator = citation_generator or CitationGenerator()

    def build(self, retrieval_results: list[RetrievalResult], query: str) -> dict[str, object]:
        if not retrieval_results:
            message = "未找到相关文档，请先运行 ingest.py 摄取数据，或检查查询条件。"
            return {
                "content": [
                    {
                        "type": "text",
                        "text": message,
                    }
                ],
                "structuredContent": {
                    "query": query,
                    "citations": [],
                    "result_count": 0,
                },
            }

        citations = self.citation_generator.generate(retrieval_results)
        markdown = self._build_markdown(retrieval_results, citations)
        return {
            "content": [
                {
                    "type": "text",
                    "text": markdown,
                }
            ],
            "structuredContent": {
                "query": query,
                "citations": citations,
                "result_count": len(retrieval_results),
            },
        }

    def _build_markdown(
        self,
        retrieval_results: list[RetrievalResult],
        citations: list[dict[str, object]],
    ) -> str:
        lines = ["检索结果：", ""]
        for citation, result in zip(citations, retrieval_results, strict=True):
            snippet = " ".join(result.text.split())
            lines.append(f"[{citation['index']}] {snippet}")
            lines.append(
                f"来源: {citation['source']} | page={citation['page'] or '-'} | score={citation['score']}"
            )
            lines.append("")
        return "\n".join(lines).strip()
