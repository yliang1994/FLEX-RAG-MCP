"""Minimal semantic splitter implementation."""

from __future__ import annotations

import re

from libs.splitter.base_splitter import BaseSplitter


class SemanticSplitter(BaseSplitter):
    """Split text on sentence boundaries while respecting chunk limits."""

    method_name = "semantic"

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", normalized)
            if sentence.strip()
        ]

        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = sentence if not current else f"{current} {sentence}"
            if current and len(candidate) > self.chunk_size:
                chunks.append(current)
                current = sentence
            else:
                current = candidate

        if current:
            chunks.append(current)
        return chunks
