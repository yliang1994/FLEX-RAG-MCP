"""Sparse chunk encoder that emits BM25-friendly term weights."""

from __future__ import annotations

import re
from collections import Counter

from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class SparseEncoder:
    """Convert chunk text into sparse term-weight dictionaries."""

    def encode(self, chunks: list[Chunk], trace: TraceContext | None = None) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        for chunk in chunks:
            sparse_vector = self._encode_text(chunk.text)
            records.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    sparse_vector=sparse_vector,
                )
            )

        if trace is not None:
            trace.record_stage(
                "sparse_encoder.encode",
                chunk_count=len(chunks),
                non_empty_vectors=sum(1 for record in records if record.sparse_vector),
            )
        return records

    def _encode_text(self, text: str) -> dict[str, float]:
        terms = [token.lower() for token in TOKEN_RE.findall(text)]
        if not terms:
            return {}

        counts = Counter(terms)
        max_tf = max(counts.values())
        return {term: count / max_tf for term, count in sorted(counts.items())}
