"""Sparse BM25 retriever backed by the persisted vector-store payloads."""

from __future__ import annotations

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class SparseRetriever:
    """Keyword retriever that joins BM25 scores with stored chunk payloads."""

    def __init__(
        self,
        settings: Settings,
        bm25_indexer: BM25Indexer | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.bm25_indexer = bm25_indexer if bm25_indexer is not None else BM25Indexer()
        self.vector_store = (
            vector_store if vector_store is not None else VectorStoreFactory.create(settings)
        )

    def retrieve(
        self,
        keywords: list[str],
        top_k: int,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        cleaned_keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
        if not cleaned_keywords or top_k <= 0:
            return []

        query = " ".join(cleaned_keywords)
        matches = self.bm25_indexer.search(query, top_k=top_k)
        if not matches:
            if trace is not None:
                trace.record_stage(
                    "sparse_retriever.retrieve",
                    keywords=cleaned_keywords,
                    top_k=top_k,
                    result_count=0,
                )
            return []

        ids = [str(match["chunk_id"]) for match in matches]
        payloads = self.vector_store.get_by_ids(ids, trace=trace)
        payload_by_id = {str(item["id"]): item for item in payloads}

        results: list[RetrievalResult] = []
        for match in matches:
            chunk_id = str(match["chunk_id"])
            payload = payload_by_id.get(chunk_id)
            if payload is None:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=float(match["score"]),
                    text=str(payload["text"]),
                    metadata=dict(payload.get("metadata", {})),
                )
            )

        if trace is not None:
            trace.record_stage(
                "sparse_retriever.retrieve",
                keywords=cleaned_keywords,
                top_k=top_k,
                result_count=len(results),
            )

        return results
