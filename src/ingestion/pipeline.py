"""MVP ingestion pipeline orchestrating load, transform, encode, and storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.loader.pdf_loader import PdfLoader


@dataclass(slots=True)
class PipelineResult:
    status: str
    document_id: str | None = None
    chunk_count: int = 0
    vector_ids: list[str] = field(default_factory=list)
    bm25_terms: int = 0
    image_count: int = 0
    trace_id: str | None = None
    trace: TraceContext | None = None


class IngestionPipeline:
    """Serial ingestion pipeline for a single document path."""

    def __init__(
        self,
        settings: Settings,
        *,
        integrity_checker: SQLiteIntegrityChecker | None = None,
        loader: PdfLoader | None = None,
        chunker: DocumentChunker | None = None,
        chunk_refiner: ChunkRefiner | None = None,
        metadata_enricher: MetadataEnricher | None = None,
        image_captioner: ImageCaptioner | None = None,
        batch_processor: BatchProcessor | None = None,
        bm25_indexer: BM25Indexer | None = None,
        vector_upserter: VectorUpserter | None = None,
        image_storage: ImageStorage | None = None,
    ) -> None:
        self.settings = settings
        self.integrity_checker = integrity_checker or SQLiteIntegrityChecker("data/db/file_integrity.db")
        self.loader = loader or PdfLoader()
        self.chunker = chunker or DocumentChunker(settings)
        self.chunk_refiner = chunk_refiner or ChunkRefiner(settings)
        self.metadata_enricher = metadata_enricher or MetadataEnricher(settings)
        self.image_captioner = image_captioner or ImageCaptioner(settings)
        self.batch_processor = batch_processor or BatchProcessor(
            dense_encoder=DenseEncoder(settings),
            sparse_encoder=SparseEncoder(),
        )
        self.bm25_indexer = bm25_indexer or BM25Indexer()
        self.vector_upserter = vector_upserter or VectorUpserter(settings)
        self.image_storage = image_storage or ImageStorage()

    def ingest(self, path: str | Path, collection: str = "default", force: bool = False) -> PipelineResult:
        trace = TraceContext(trace_type="ingestion")
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"input file not found: {source_path}")

        sha256 = self.integrity_checker.compute_sha256(source_path)
        if not force and self.integrity_checker.should_skip(sha256):
            trace.record_stage(
                "skip",
                elapsed_ms=0.0,
                method="sha256",
                source_path=str(source_path),
                reason="unchanged_document",
            )
            trace.finish()
            return PipelineResult(status="skipped", trace_id=trace.trace_id, trace=trace)

        try:
            document = self._timed_stage(
                trace,
                "load",
                method=getattr(self.loader, "__class__", type(self.loader)).__name__,
                callback=lambda: self.loader.load(source_path),
                details_factory=lambda document: {
                    "source_path": str(source_path),
                    "file_size": source_path.stat().st_size,
                    "image_count": len(document.metadata.get("images", []))
                    if isinstance(document.metadata.get("images", []), list)
                    else 0,
                },
            )
            chunks = self._timed_stage(
                trace,
                "split",
                method=getattr(self.chunker.splitter, "__class__", type(self.chunker.splitter)).__name__,
                callback=lambda: self.chunker.split_document(document),
                details_factory=self._split_trace_details,
            )
            chunks = self._timed_stage(
                trace,
                "transform",
                method="chunk_refiner+metadata_enricher+image_captioner",
                callback=lambda: self._transform_chunks(chunks, trace),
                details_factory=lambda transformed_chunks: {
                    "chunk_count": len(transformed_chunks),
                    "steps": ["chunk_refiner", "metadata_enricher", "image_captioner"],
                },
            )
            records = self._timed_stage(
                trace,
                "embed",
                method=getattr(self.batch_processor, "__class__", type(self.batch_processor)).__name__,
                callback=lambda: self.batch_processor.process(chunks, trace=trace),
                details_factory=lambda encoded_records: {
                    "record_count": len(encoded_records),
                    "batch_size": getattr(self.batch_processor, "batch_size", None),
                },
            )
            upsert_result = self._timed_stage(
                trace,
                "upsert",
                method="bm25+vector_store+image_storage",
                callback=lambda: self._upsert_assets(
                    document=document,
                    records=records,
                    collection=collection,
                    trace=trace,
                ),
                details_factory=lambda payload: {
                    "vector_count": len(payload["vector_ids"]),
                    "bm25_terms": payload["bm25_terms"],
                    "image_count": payload["image_count"],
                },
            )
            image_count = upsert_result["image_count"]
            vector_ids = upsert_result["vector_ids"]
            bm25_terms = upsert_result["bm25_terms"]
            self.integrity_checker.mark_success(sha256, str(source_path), chunk_count=len(chunks), file_size=source_path.stat().st_size)
        except Exception as exc:
            trace.record_stage("pipeline.error", error=str(exc))
            trace.finish()
            self.integrity_checker.mark_failed(sha256, str(exc))
            raise RuntimeError(f"pipeline stage failed: {exc}") from exc

        trace.finish()
        return PipelineResult(
            status="ingested",
            document_id=document.id,
            chunk_count=len(chunks),
            vector_ids=vector_ids,
            bm25_terms=bm25_terms,
            image_count=image_count,
            trace_id=trace.trace_id,
            trace=trace,
        )

    def _timed_stage(
        self,
        trace: TraceContext,
        name: str,
        *,
        method: str,
        callback,
        details_factory=None,
    ):
        started_at = perf_counter()
        try:
            result = callback()
        except Exception as exc:
            raise RuntimeError(f"{name}: {exc}") from exc
        details = details_factory(result) if details_factory is not None else {}
        trace.record_stage(
            name,
            elapsed_ms=round((perf_counter() - started_at) * 1000, 3),
            method=method,
            **details,
        )
        return result

    def _transform_chunks(self, chunks: list[Chunk], trace: TraceContext) -> list[Chunk]:
        refined = self.chunk_refiner.transform(chunks, trace=trace)
        enriched = self.metadata_enricher.transform(refined, trace=trace)
        return self.image_captioner.transform(enriched, trace=trace)

    def _upsert_assets(
        self,
        *,
        document,
        records: list[ChunkRecord],
        collection: str,
        trace: TraceContext,
    ) -> dict[str, Any]:
        image_count = self._persist_document_images(document.metadata.get("images", []), collection)
        self.bm25_indexer.build(records)
        vector_ids = self.vector_upserter.upsert(records, trace=trace)
        return {
            "image_count": image_count,
            "vector_ids": vector_ids,
            "bm25_terms": len(self.bm25_indexer.index),
        }

    def _split_trace_details(self, chunks: list[Chunk]) -> dict[str, Any]:
        average_length = 0.0
        if chunks:
            average_length = round(sum(len(chunk.text) for chunk in chunks) / len(chunks), 3)
        return {
            "chunk_count": len(chunks),
            "avg_chunk_length": average_length,
        }

    def _persist_document_images(self, images: Any, collection: str) -> int:
        if not isinstance(images, list):
            return 0

        stored_count = 0
        for image in images:
            if not isinstance(image, dict):
                continue
            image_id = str(image.get("id", "")).strip()
            image_path = str(image.get("path", "")).strip()
            if not image_id or not image_path:
                continue
            stored_path = self.image_storage.save(image_id, image_path, collection=collection)
            image["path"] = stored_path
            stored_count += 1
        return stored_count
