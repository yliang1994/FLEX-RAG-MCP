"""Minimal fixed-length splitter implementation."""

from __future__ import annotations

from libs.splitter.base_splitter import BaseSplitter


class FixedLengthSplitter(BaseSplitter):
    """Split text into fixed-size windows with optional overlap."""

    method_name = "fixed"

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = start + self.chunk_size
            chunks.append(normalized[start:end])
            start += step
        return chunks
