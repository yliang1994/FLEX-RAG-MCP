"""BM25 index builder with JSON persistence and query support."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from core.types import ChunkRecord


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class BM25Indexer:
    """Build, persist, load, and query a small BM25 inverted index."""

    def __init__(self, index_dir: str | Path = "data/db/bm25") -> None:
        self.index_dir = Path(index_dir)
        self.index_file = self.index_dir / "index.json"
        self.docs_file = self.index_dir / "docs.json"
        self.index: dict[str, dict] = {}
        self.documents: dict[str, dict] = {}
        self.avg_doc_length = 0.0

    def build(self, records: list[ChunkRecord]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.documents = {
            record.id: {
                "text": record.text,
                "metadata": record.metadata,
                "sparse_vector": record.sparse_vector or self._term_frequencies(record.text),
                "doc_length": self._doc_length(record),
            }
            for record in records
        }
        self._rebuild_index()
        self._persist()

    def upsert(self, records: list[ChunkRecord]) -> None:
        if self.docs_file.exists() and not self.documents:
            self.load()
        for record in records:
            self.documents[record.id] = {
                "text": record.text,
                "metadata": record.metadata,
                "sparse_vector": record.sparse_vector or self._term_frequencies(record.text),
                "doc_length": self._doc_length(record),
            }
        self._rebuild_index()
        self._persist()

    def load(self) -> None:
        if not self.index_file.exists() or not self.docs_file.exists():
            raise FileNotFoundError(f"BM25 index not found under {self.index_dir}")
        payload = json.loads(self.index_file.read_text(encoding="utf-8"))
        self.index = payload["index"]
        self.avg_doc_length = float(payload["avg_doc_length"])
        self.documents = json.loads(self.docs_file.read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, float | str]]:
        query_terms = [token.lower() for token in TOKEN_RE.findall(query)]
        if not query_terms:
            return []

        scores: dict[str, float] = {}
        for term in query_terms:
            entry = self.index.get(term)
            if entry is None:
                continue
            idf = float(entry["idf"])
            for posting in entry["postings"]:
                chunk_id = str(posting["chunk_id"])
                tf = float(posting["tf"])
                doc_length = float(posting["doc_length"])
                scores[chunk_id] = scores.get(chunk_id, 0.0) + self._bm25_score(tf, idf, doc_length)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            {"chunk_id": chunk_id, "score": score, "text": self.documents[chunk_id]["text"]}
            for chunk_id, score in ranked[:top_k]
        ]

    def _rebuild_index(self) -> None:
        total_length = sum(float(document["doc_length"]) for document in self.documents.values())
        self.avg_doc_length = total_length / len(self.documents) if self.documents else 0.0

        terms: dict[str, list[dict[str, float | str]]] = {}
        for chunk_id, document in self.documents.items():
            sparse_vector = document["sparse_vector"]
            doc_length = float(document["doc_length"])
            for term, tf in sparse_vector.items():
                terms.setdefault(term, []).append(
                    {"chunk_id": chunk_id, "tf": float(tf), "doc_length": doc_length}
                )

        total_docs = len(self.documents)
        self.index = {}
        for term, postings in terms.items():
            df = len(postings)
            idf = math.log((total_docs - df + 0.5) / (df + 0.5)) if total_docs else 0.0
            self.index[term] = {
                "idf": idf,
                "postings": sorted(postings, key=lambda posting: str(posting["chunk_id"])),
            }

    def _persist(self) -> None:
        self.index_file.write_text(
            json.dumps(
                {"avg_doc_length": self.avg_doc_length, "index": self.index},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.docs_file.write_text(
            json.dumps(self.documents, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _term_frequencies(self, text: str) -> dict[str, float]:
        tokens = [token.lower() for token in TOKEN_RE.findall(text)]
        if not tokens:
            return {}
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        max_tf = max(counts.values())
        return {term: count / max_tf for term, count in sorted(counts.items())}

    def _doc_length(self, record: ChunkRecord) -> int:
        tokens = TOKEN_RE.findall(record.text)
        return len(tokens)

    def _bm25_score(self, tf: float, idf: float, doc_length: float, k1: float = 1.5, b: float = 0.75) -> float:
        avg_length = self.avg_doc_length or 1.0
        denominator = tf + k1 * (1.0 - b + b * (doc_length / avg_length))
        if denominator == 0:
            return 0.0
        return idf * ((tf * (k1 + 1.0)) / denominator)
