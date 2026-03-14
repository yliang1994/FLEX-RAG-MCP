"""Minimal in-memory Chroma-like vector store."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.types import QueryMatch, VectorRecord
from libs.vector_store.base_vector_store import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """JSON-backed store that preserves the future Chroma contract."""

    backend_name = "chroma"

    def __init__(self, persist_path: str, **kwargs: Any) -> None:
        super().__init__(persist_path, **kwargs)
        self.persist_dir = Path(persist_path)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = self.persist_dir / "records.json"
        self._records: dict[str, VectorRecord] = self._load_records()

    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> int:
        for record in records:
            if not record.id:
                raise ValueError("record.id must not be empty")
            if not record.vector:
                raise ValueError(f"record.vector must not be empty: {record.id}")
            self._records[record.id] = record
        self._save_records()
        return len(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[QueryMatch]:
        if not vector:
            raise ValueError("vector must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        matches: list[QueryMatch] = []
        for record in self._records.values():
            if filters and not _match_filters(record.metadata, filters):
                continue
            matches.append(
                QueryMatch(
                    id=record.id,
                    score=_cosine_similarity(vector, record.vector),
                    text=record.text,
                    metadata=record.metadata,
                )
            )

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:top_k]

    def _load_records(self) -> dict[str, VectorRecord]:
        if not self.data_file.exists():
            return {}

        payload = json.loads(self.data_file.read_text(encoding="utf-8"))
        return {item["id"]: VectorRecord(**item) for item in payload}

    def _save_records(self) -> None:
        payload = [asdict(record) for record in self._records.values()]
        self.data_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _match_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if metadata.get(key) != expected:
            return False
    return True


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("query vector and record vector must have the same dimension")

    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
