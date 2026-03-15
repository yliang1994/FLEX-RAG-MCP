"""Metadata enrichment transform with rule-based and optional LLM enrichment."""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.llm_factory import LLMFactory


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}

WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}")
DEFAULT_SUMMARY_LIMIT = 180
DEFAULT_TAG_COUNT = 5
DEFAULT_PROMPT = """Extract metadata for the following chunk.
Return strict JSON with keys "title", "summary", and "tags".
`tags` must be a JSON array of short strings.

Chunk:
{text}
"""


class MetadataEnricher(BaseTransform):
    """Attach title, summary, and tags to chunks."""

    def __init__(self, settings: Settings, llm: BaseLLM | None = None) -> None:
        self.settings = settings
        self.use_llm = settings.ingestion.metadata_enricher.use_llm
        self.llm = llm if llm is not None else self._build_llm()

    def transform(self, chunks: list[Chunk], trace: TraceContext | None = None) -> list[Chunk]:
        enriched_chunks: list[Chunk] = []

        for chunk in chunks:
            rule_metadata = self._rule_metadata(chunk.text)
            final_metadata = deepcopy(chunk.metadata)
            final_metadata.update(rule_metadata)
            final_metadata["metadata_enriched_by"] = "rule"

            llm_metadata = self._llm_metadata(chunk.text, trace)
            if llm_metadata is not None:
                final_metadata.update(llm_metadata)
                final_metadata["metadata_enriched_by"] = "llm"
            elif self.use_llm and self.llm is not None:
                final_metadata["metadata_enricher_fallback"] = "llm_failed"

            enriched_chunks.append(
                Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=final_metadata,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    source_ref=chunk.source_ref,
                )
            )

        if trace is not None:
            trace.record_stage(
                "metadata_enricher.transform",
                chunk_count=len(chunks),
                llm_enabled=self.use_llm and self.llm is not None,
            )
        return enriched_chunks

    def _build_llm(self) -> BaseLLM | None:
        if not self.use_llm:
            return None
        return LLMFactory.create(self.settings)

    def _rule_metadata(self, text: str) -> dict[str, object]:
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            return {
                "title": "Untitled Chunk",
                "summary": "Empty chunk.",
                "tags": ["empty"],
            }

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title = self._derive_title(lines, cleaned)
        summary = cleaned[:DEFAULT_SUMMARY_LIMIT].rstrip()
        if len(cleaned) > DEFAULT_SUMMARY_LIMIT:
            summary = f"{summary}..."
        tags = self._derive_tags(cleaned)
        return {
            "title": title,
            "summary": summary or title,
            "tags": tags or ["general"],
        }

    def _derive_title(self, lines: list[str], fallback: str) -> str:
        for line in lines:
            compact = " ".join(line.split())
            if 4 <= len(compact) <= 80:
                return compact
        return " ".join(fallback.split()[:8]).strip(". ") or "Untitled Chunk"

    def _derive_tags(self, text: str) -> list[str]:
        counts = Counter(
            word.lower()
            for word in WORD_RE.findall(text)
            if len(word) > 2 and word.lower() not in STOPWORDS
        )
        return [word for word, _count in counts.most_common(DEFAULT_TAG_COUNT)]

    def _llm_metadata(self, text: str, trace: TraceContext | None = None) -> dict[str, object] | None:
        if not self.use_llm or self.llm is None:
            return None

        prompt = DEFAULT_PROMPT.format(text=text)
        try:
            response = self.llm.chat([ChatMessage(role="user", content=prompt)])
            parsed = json.loads(response.content)
        except Exception as exc:
            if trace is not None:
                trace.record_stage("metadata_enricher.llm_fallback", reason=str(exc))
            return None

        title = str(parsed.get("title", "")).strip()
        summary = str(parsed.get("summary", "")).strip()
        raw_tags = parsed.get("tags", [])
        if not isinstance(raw_tags, list):
            raw_tags = []
        tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
        if not title or not summary or not tags:
            if trace is not None:
                trace.record_stage("metadata_enricher.llm_fallback", reason="invalid_payload")
            return None
        return {"title": title, "summary": summary, "tags": tags}
