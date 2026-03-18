"""Helpers for generating structured citations from retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, asdict

from core.types import RetrievalResult


@dataclass(slots=True)
class Citation:
    index: int
    source: str
    page: int | None
    chunk_id: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CitationGenerator:
    """Build stable citation payloads for MCP structured content."""

    def generate(self, retrieval_results: list[RetrievalResult]) -> list[dict[str, object]]:
        citations: list[dict[str, object]] = []
        for index, result in enumerate(retrieval_results, start=1):
            citations.append(
                Citation(
                    index=index,
                    source=self._resolve_source(result),
                    page=self._resolve_page(result),
                    chunk_id=result.chunk_id,
                    score=round(float(result.score), 6),
                ).to_dict()
            )
        return citations

    def _resolve_source(self, result: RetrievalResult) -> str:
        metadata = result.metadata
        source = metadata.get("source_path") or metadata.get("source") or metadata.get("document_id")
        if source is None:
            return "unknown"
        return str(source)

    def _resolve_page(self, result: RetrievalResult) -> int | None:
        page = result.metadata.get("page")
        return page if isinstance(page, int) else None
