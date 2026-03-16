"""Dense retriever built on top of embedding and vector-store abstractions."""

from __future__ import annotations

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class DenseRetriever:
    """Semantic retriever that embeds the query and searches the vector store."""

    def __init__(
        self,
        settings: Settings,
        embedding_client: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = (
            embedding_client if embedding_client is not None else EmbeddingFactory.create(settings)
        )
        self.vector_store = (
            vector_store if vector_store is not None else VectorStoreFactory.create(settings)
        )

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, object] | None = None,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        vectors = self.embedding_client.embed([normalized_query], trace=trace)
        if len(vectors) != 1:
            raise ValueError("dense_retriever: response_error: expected exactly one query vector")

        matches = self.vector_store.query(vectors[0], top_k=top_k, filters=filters, trace=trace)
        results = [
            RetrievalResult(
                chunk_id=match.id,
                score=match.score,
                text=match.text,
                metadata=dict(match.metadata),
            )
            for match in matches
        ]

        if trace is not None:
            trace.record_stage(
                "dense_retriever.retrieve",
                query=normalized_query,
                top_k=top_k,
                result_count=len(results),
                filters=dict(filters or {}),
            )

        return results
