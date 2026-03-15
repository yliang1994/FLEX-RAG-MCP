"""MVP ingestion pipeline orchestrating load, transform, encode, and storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
        trace = TraceContext()
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"input file not found: {source_path}")

        sha256 = self.integrity_checker.compute_sha256(source_path)
        if not force and self.integrity_checker.should_skip(sha256):
            return PipelineResult(status="skipped", trace_id=trace.trace_id)

        try:
            document = self._stage("load", trace, lambda: self.loader.load(source_path))
            image_count = self._stage(
                "store_images",
                trace,
                lambda: self._persist_document_images(document.metadata.get("images", []), collection),
            )
            chunks = self._stage("split", trace, lambda: self.chunker.split_document(document))
            chunks = self._stage("refine", trace, lambda: self.chunk_refiner.transform(chunks, trace=trace))
            chunks = self._stage("metadata", trace, lambda: self.metadata_enricher.transform(chunks, trace=trace))
            chunks = self._stage("caption", trace, lambda: self.image_captioner.transform(chunks, trace=trace))
            records = self._stage("encode", trace, lambda: self.batch_processor.process(chunks, trace=trace))
            self._stage("bm25", trace, lambda: self.bm25_indexer.build(records))
            vector_ids = self._stage("vector_upsert", trace, lambda: self.vector_upserter.upsert(records, trace=trace))
            self.integrity_checker.mark_success(sha256, str(source_path), chunk_count=len(chunks), file_size=source_path.stat().st_size)
        except Exception as exc:
            self.integrity_checker.mark_failed(sha256, str(exc))
            raise RuntimeError(f"pipeline stage failed: {exc}") from exc

        return PipelineResult(
            status="ingested",
            document_id=document.id,
            chunk_count=len(chunks),
            vector_ids=vector_ids,
            bm25_terms=len(self.bm25_indexer.index),
            image_count=image_count,
            trace_id=trace.trace_id,
        )

    def _stage(self, name: str, trace: TraceContext, callback):
        try:
            result = callback()
        except Exception as exc:
            trace.record_stage("pipeline.error", stage=name, error=str(exc))
            raise RuntimeError(f"{name}: {exc}") from exc
        trace.record_stage("pipeline.stage", stage=name)
        return result

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
