from __future__ import annotations

from pathlib import Path

from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import Settings, load_settings
from core.trace.trace_context import TraceContext
from core.types import ChunkRecord
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.vector_store.base_vector_store import BaseVectorStore


class FakeVectorStore(BaseVectorStore):
    backend_name = "fake"

    def __init__(self) -> None:
        super().__init__(persist_path="./tmp/fake")
        self.requested_ids: list[str] = []

    def upsert(self, records, trace=None):  # type: ignore[override]
        return len(records)

    def query(self, vector, top_k, filters=None, trace=None):  # type: ignore[override]
        raise NotImplementedError

    def get_by_ids(self, ids, trace=None):  # type: ignore[override]
        self.requested_ids = list(ids)
        payloads = {
            "chunk-1": {
                "id": "chunk-1",
                "text": "alpha beta notes",
                "metadata": {"source": "docs/a.pdf", "page": 1},
            },
            "chunk-2": {
                "id": "chunk-2",
                "text": "beta gamma guide",
                "metadata": {"source": "docs/b.pdf", "page": 2},
            },
        }
        return [payloads[record_id] for record_id in ids if record_id in payloads]


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings("config/settings.yaml")
    settings.vector_store.persist_path = str(tmp_path / "chroma")
    return settings


def _build_index(tmp_path: Path) -> BM25Indexer:
    indexer = BM25Indexer(tmp_path / "bm25")
    indexer.build(
        [
            ChunkRecord(
                id="chunk-1",
                text="alpha beta notes",
                metadata={"source": "docs/a.pdf", "page": 1},
                sparse_vector={"alpha": 1.0, "beta": 0.5},
            ),
            ChunkRecord(
                id="chunk-2",
                text="beta gamma guide",
                metadata={"source": "docs/b.pdf", "page": 2},
                sparse_vector={"beta": 1.0, "gamma": 1.0},
            ),
        ]
    )
    return indexer


def test_sparse_retriever_joins_bm25_scores_with_vector_store_payloads(tmp_path: Path) -> None:
    indexer = _build_index(tmp_path)
    vector_store = FakeVectorStore()
    retriever = SparseRetriever(_settings(tmp_path), bm25_indexer=indexer, vector_store=vector_store)
    trace = TraceContext()

    results = retriever.retrieve(["alpha", "beta"], top_k=2, trace=trace)

    assert [result.chunk_id for result in results] == ["chunk-1", "chunk-2"]
    assert vector_store.requested_ids == ["chunk-1", "chunk-2"]
    assert results[0].text == "alpha beta notes"
    assert results[0].metadata["page"] == 1
    assert trace.stages[-1].name == "sparse_retriever.retrieve"
    assert trace.stages[-1].details["result_count"] == 2


def test_sparse_retriever_returns_empty_for_blank_keywords(tmp_path: Path) -> None:
    retriever = SparseRetriever(
        _settings(tmp_path),
        bm25_indexer=_build_index(tmp_path),
        vector_store=FakeVectorStore(),
    )

    assert retriever.retrieve(["", "   "], top_k=3) == []


def test_sparse_retriever_skips_missing_vector_payloads(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path / "bm25")
    indexer.build(
        [
            ChunkRecord(
                id="missing",
                text="orphan term",
                metadata={"source": "docs/missing.pdf"},
                sparse_vector={"orphan": 1.0},
            )
        ]
    )
    retriever = SparseRetriever(_settings(tmp_path), bm25_indexer=indexer, vector_store=FakeVectorStore())

    assert retriever.retrieve(["orphan"], top_k=1) == []
