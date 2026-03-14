"""Minimal recursive splitter implementation."""

from __future__ import annotations

from dataclasses import dataclass

from libs.splitter.base_splitter import BaseSplitter


@dataclass(slots=True)
class MarkdownBlock:
    """Semantic markdown block used to preserve structure while chunking."""

    kind: str
    text: str


class RecursiveSplitter(BaseSplitter):
    """Split markdown text while preserving headings and fenced code blocks."""

    method_name = "recursive"

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []

        blocks = self._parse_markdown_blocks(normalized)
        chunks: list[str] = []
        current_blocks: list[str] = []

        for block in blocks:
            candidate_parts = [*current_blocks, block.text]
            candidate = "\n\n".join(candidate_parts)
            if len(candidate) <= self.chunk_size:
                current_blocks.append(block.text)
                continue

            if current_blocks:
                chunks.append("\n\n".join(current_blocks))
                current_blocks = []

            if block.kind == "code":
                chunks.append(block.text)
                continue

            chunks.extend(self._split_large_block(block.text))

        if current_blocks:
            chunks.append("\n\n".join(current_blocks))
        return chunks

    def _parse_markdown_blocks(self, text: str) -> list[MarkdownBlock]:
        lines = text.splitlines()
        blocks: list[MarkdownBlock] = []
        paragraph_lines: list[str] = []
        in_code_block = False
        code_lines: list[str] = []

        def flush_paragraph() -> None:
            if paragraph_lines:
                blocks.append(MarkdownBlock(kind="paragraph", text="\n".join(paragraph_lines).strip()))
                paragraph_lines.clear()

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                flush_paragraph()
                code_lines.append(line)
                if in_code_block:
                    blocks.append(MarkdownBlock(kind="code", text="\n".join(code_lines).strip()))
                    code_lines.clear()
                    in_code_block = False
                else:
                    in_code_block = True
                continue

            if in_code_block:
                code_lines.append(line)
                continue

            if stripped.startswith("#"):
                flush_paragraph()
                blocks.append(MarkdownBlock(kind="heading", text=stripped))
                continue

            if not stripped:
                flush_paragraph()
                continue

            paragraph_lines.append(line)

        flush_paragraph()
        if code_lines:
            blocks.append(MarkdownBlock(kind="code", text="\n".join(code_lines).strip()))
        return blocks

    def _split_large_block(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        words = text.split()
        chunks: list[str] = []
        current_words: list[str] = []

        for word in words:
            candidate = " ".join([*current_words, word])
            if current_words and len(candidate) > self.chunk_size:
                chunks.append(" ".join(current_words))
                overlap_words = self._overlap_tail(current_words)
                current_words = [*overlap_words, word]
            else:
                current_words.append(word)

        if current_words:
            chunks.append(" ".join(current_words))
        return chunks

    def _overlap_tail(self, words: list[str]) -> list[str]:
        if self.chunk_overlap == 0 or not words:
            return []

        selected: list[str] = []
        total = 0
        for word in reversed(words):
            projected = total + len(word) + (1 if selected else 0)
            if projected > self.chunk_overlap:
                break
            selected.insert(0, word)
            total = projected
        return selected
