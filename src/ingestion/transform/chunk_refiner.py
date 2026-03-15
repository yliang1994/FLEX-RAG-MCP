"""Chunk refinement transform with rule-based cleanup and optional LLM rewrite."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.llm_factory import LLMFactory


DEFAULT_PROMPT = """Rewrite the input chunk into a cleaner, self-contained passage.

Input:
{text}
"""

CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
PAGE_NUMBER_RE = re.compile(r"^\s*(page|p\.)\s*\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE)
SEPARATOR_RE = re.compile(r"^\s*[-_=*]{3,}\s*$")
WHITESPACE_RE = re.compile(r"[ \t]+")
MULTI_BLANK_RE = re.compile(r"\n{3,}")


class ChunkRefiner(BaseTransform):
    """Clean noisy chunks and optionally ask an LLM for a rewrite."""

    def __init__(
        self,
        settings: Settings,
        llm: BaseLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.use_llm = settings.ingestion.chunk_refiner.use_llm
        self.llm = llm if llm is not None else self._build_llm()
        self.prompt_template = self._load_prompt(prompt_path)

    def transform(self, chunks: list[Chunk], trace: TraceContext | None = None) -> list[Chunk]:
        refined_chunks: list[Chunk] = []

        for chunk in chunks:
            metadata = deepcopy(chunk.metadata)
            try:
                rule_text = self._rule_based_refine(chunk.text)
                refined_text = rule_text
                refined_by = "rule"

                llm_text = self._llm_refine(rule_text, trace)
                if llm_text is not None:
                    refined_text = llm_text
                    refined_by = "llm"

                metadata["refined_by"] = refined_by
                refined_chunks.append(
                    Chunk(
                        id=chunk.id,
                        text=refined_text,
                        metadata=metadata,
                        start_offset=chunk.start_offset,
                        end_offset=chunk.end_offset,
                        source_ref=chunk.source_ref,
                    )
                )
            except Exception as exc:
                metadata["refined_by"] = "original"
                metadata["refine_error"] = str(exc)
                if trace is not None:
                    trace.record_stage("chunk_refiner.error", chunk_id=chunk.id, error=str(exc))
                refined_chunks.append(
                    Chunk(
                        id=chunk.id,
                        text=chunk.text,
                        metadata=metadata,
                        start_offset=chunk.start_offset,
                        end_offset=chunk.end_offset,
                        source_ref=chunk.source_ref,
                    )
                )

        if trace is not None:
            trace.record_stage(
                "chunk_refiner.transform",
                chunk_count=len(chunks),
                llm_enabled=self.use_llm and self.llm is not None,
            )
        return refined_chunks

    def _build_llm(self) -> BaseLLM | None:
        if not self.use_llm:
            return None
        return LLMFactory.create(self.settings)

    def _rule_based_refine(self, text: str) -> str:
        code_blocks: list[str] = []

        def _preserve_code(match: re.Match[str]) -> str:
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        working = text.replace("\r\n", "\n").replace("\r", "\n")
        working = CODE_BLOCK_RE.sub(_preserve_code, working)
        working = HTML_COMMENT_RE.sub("", working)
        working = MARKDOWN_IMAGE_RE.sub("", working)

        cleaned_lines: list[str] = []
        previous_line = ""
        for raw_line in working.split("\n"):
            normalized = WHITESPACE_RE.sub(" ", raw_line).strip()
            if PAGE_NUMBER_RE.match(normalized):
                continue
            if SEPARATOR_RE.match(normalized):
                continue
            if normalized and normalized == previous_line and len(normalized) <= 80:
                continue
            cleaned_lines.append(normalized)
            if normalized:
                previous_line = normalized

        refined = "\n".join(cleaned_lines)
        refined = MULTI_BLANK_RE.sub("\n\n", refined).strip()

        for index, code_block in enumerate(code_blocks):
            refined = refined.replace(f"__CODE_BLOCK_{index}__", code_block)

        return refined

    def _llm_refine(self, text: str, trace: TraceContext | None = None) -> str | None:
        if not self.use_llm or self.llm is None:
            return None

        prompt = self.prompt_template.format(text=text)
        try:
            response = self.llm.chat([ChatMessage(role="user", content=prompt)])
        except Exception as exc:
            if trace is not None:
                trace.record_stage("chunk_refiner.llm_fallback", reason=str(exc))
            return None

        refined = response.content.strip()
        if not refined:
            if trace is not None:
                trace.record_stage("chunk_refiner.llm_fallback", reason="empty_response")
            return None
        return refined

    def _load_prompt(self, prompt_path: str | Path | None) -> str:
        resolved_path = Path(prompt_path) if prompt_path is not None else Path("config/prompts/chunk_refinement.txt")
        if resolved_path.exists():
            template = resolved_path.read_text(encoding="utf-8").strip()
            if "{text}" not in template:
                template = f"{template}\n\n{{text}}"
            return template
        return DEFAULT_PROMPT

