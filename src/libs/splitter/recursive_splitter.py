"""Minimal recursive splitter implementation."""

from __future__ import annotations

from libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """Split on paragraph boundaries first, then fall back to fixed windows."""

    method_name = "recursive"

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []

        chunks: list[str] = []
        current = ""
        for paragraph in normalized.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            candidate = paragraph if not current else f"{current}\n\n{paragraph}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            chunks.extend(self._split_large_block(paragraph))
            current = ""

        if current:
            chunks.append(current)
        return chunks

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
