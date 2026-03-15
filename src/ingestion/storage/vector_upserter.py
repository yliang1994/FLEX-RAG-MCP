"""Vector store upsert adapter with deterministic ids."""

from __future__ import annotations

import hashlib

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import ChunkRecord, VectorRecord
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class VectorUpserter:
    """Persist dense chunk records into the configured vector store."""

    def __init__(self, settings: Settings, store: BaseVectorStore | None = None) -> None:
        self.settings = settings
        self.store = store if store is not None else VectorStoreFactory.create(settings)

    def upsert(self, records: list[ChunkRecord], trace: TraceContext | None = None) -> list[str]:
        vector_records = [self._to_vector_record(record) for record in records]
        self.store.upsert(vector_records, trace=trace)
        ids = [record.id for record in vector_records]
        if trace is not None:
            trace.record_stage(
                "vector_upserter.upsert",
                record_count=len(vector_records),
                ids=ids,
            )
        return ids

    def _to_vector_record(self, record: ChunkRecord) -> VectorRecord:
        if not record.dense_vector:
            raise ValueError(f"vector_upserter: missing dense_vector for record {record.id}")
        vector_id = self._stable_id(record)
        metadata = dict(record.metadata)
        metadata.setdefault("original_chunk_id", record.id)
        return VectorRecord(
            id=vector_id,
            text=record.text,
            vector=record.dense_vector,
            metadata=metadata,
        )

    def _stable_id(self, record: ChunkRecord) -> str:
        source_ref = str(
            record.metadata.get("source_path")
            or record.metadata.get("source_ref")
            or record.metadata.get("source")
            or record.id
        )
        chunk_index = record.metadata.get("chunk_index", 0)
        content_hash = hashlib.sha256(record.text.encode("utf-8")).hexdigest()[:8]
        seed = f"{source_ref}:{chunk_index}:{content_hash}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
