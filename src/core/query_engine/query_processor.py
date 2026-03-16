"""Query pre-processing for retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.types import MetadataDict, ProcessedQuery

_FILTER_PATTERN = re.compile(
    r"(?P<prefix>(?:^|\s))(?P<key>collection|doc_type|language|access_level)"
    r"(?::|=)(?P<value>[^\s]+)",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*|[\u4e00-\u9fff]{2,}")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
}


@dataclass(slots=True)
class QueryProcessor:
    """Rule-based query normalizer for dense and sparse retrieval."""

    stopwords: set[str] = field(default_factory=lambda: set(_STOPWORDS))

    def process(self, query: str) -> ProcessedQuery:
        """Extract a normalized query, sparse keywords, and optional filters."""
        raw_query = query.strip()
        filters, normalized_query = self._extract_filters(raw_query)
        keywords = self._extract_keywords(normalized_query)
        if not keywords and normalized_query:
            keywords = [normalized_query]

        return ProcessedQuery(
            query=raw_query,
            normalized_query=normalized_query,
            keywords=keywords,
            filters=filters,
        )

    def _extract_filters(self, query: str) -> tuple[MetadataDict, str]:
        filters: MetadataDict = {}

        def _replace(match: re.Match[str]) -> str:
            filters[match.group("key").lower()] = match.group("value")
            return match.group("prefix")

        normalized = _FILTER_PATTERN.sub(_replace, query)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return filters, normalized

    def _extract_keywords(self, query: str) -> list[str]:
        keywords: list[str] = []
        seen: set[str] = set()

        for token in _TOKEN_PATTERN.findall(query.lower()):
            if token in self.stopwords:
                continue
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
        return keywords
