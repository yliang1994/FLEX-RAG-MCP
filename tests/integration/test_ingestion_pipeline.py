from __future__ import annotations

from pathlib import Path

from core.settings import load_settings
from ingestion.pipeline import IngestionPipeline
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore
from ingestion.storage.vector_upserter import VectorUpserter

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ingestion_pipeline_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    document_path = tmp_path / "simple.pdf"
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-image")
    document_path.write_text(
        "Simple Title\n\nThis is a technical paragraph.\n\n[[IMAGE:diagram.png]]\n\nAnother paragraph.",
        encoding="utf-8",
    )

    settings = load_settings(REPO_ROOT / "config" / "settings.yaml")
    settings.vector_store.persist_path = str(tmp_path / "data" / "db" / "chroma")

    pipeline = IngestionPipeline(
        settings,
        integrity_checker=SQLiteIntegrityChecker(tmp_path / "data" / "db" / "file_integrity.db"),
        bm25_indexer=BM25Indexer(tmp_path / "data" / "db" / "bm25"),
        image_storage=ImageStorage(tmp_path / "data" / "images", tmp_path / "data" / "db" / "image_index.db"),
        vector_upserter=VectorUpserter(settings, store=ChromaStore(str(tmp_path / "data" / "db" / "chroma"))),
    )

    result = pipeline.ingest(document_path, collection="manuals")

    assert result.status == "ingested"
    assert result.chunk_count > 0
    assert result.vector_ids
    assert result.bm25_terms > 0
    assert result.image_count == 1
    assert result.trace is not None
    assert result.trace.trace_type == "ingestion"
    stage_names = [stage.name for stage in result.trace.stages]
    top_level_stage_names = [name for name in stage_names if name in {"load", "split", "transform", "embed", "upsert"}]
    assert top_level_stage_names == [
        "load",
        "split",
        "transform",
        "embed",
        "upsert",
    ]
    top_level_stages = [stage for stage in result.trace.stages if stage.name in {"load", "split", "transform", "embed", "upsert"}]
    assert all(stage.details["elapsed_ms"] >= 0 for stage in top_level_stages)
    assert top_level_stages[0].details["method"] == "PdfLoader"
    assert top_level_stages[1].details["chunk_count"] == result.chunk_count
    assert top_level_stages[4].details["image_count"] == 1
    assert (tmp_path / "data" / "db" / "bm25" / "index.json").exists()
    assert (tmp_path / "data" / "db" / "chroma" / "records.json").exists()


def test_ingestion_pipeline_skips_unchanged_document(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    document_path = tmp_path / "simple.pdf"
    document_path.write_text("Simple Title\n\nOne paragraph.", encoding="utf-8")

    settings = load_settings(REPO_ROOT / "config" / "settings.yaml")
    settings.vector_store.persist_path = str(tmp_path / "data" / "db" / "chroma")

    pipeline = IngestionPipeline(
        settings,
        integrity_checker=SQLiteIntegrityChecker(tmp_path / "data" / "db" / "file_integrity.db"),
        bm25_indexer=BM25Indexer(tmp_path / "data" / "db" / "bm25"),
        image_storage=ImageStorage(tmp_path / "data" / "images", tmp_path / "data" / "db" / "image_index.db"),
        vector_upserter=VectorUpserter(settings, store=ChromaStore(str(tmp_path / "data" / "db" / "chroma"))),
    )

    first = pipeline.ingest(document_path, collection="manuals")
    second = pipeline.ingest(document_path, collection="manuals")

    assert first.status == "ingested"
    assert second.status == "skipped"
    assert second.trace is not None
    assert second.trace.trace_type == "ingestion"
    assert second.trace.stages[-1].name == "skip"
