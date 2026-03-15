from __future__ import annotations

from pathlib import Path

from core.settings import load_settings
from ingestion.pipeline import IngestionPipeline
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore
from ingestion.storage.vector_upserter import VectorUpserter


def test_ingestion_pipeline_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    document_path = tmp_path / "simple.pdf"
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-image")
    document_path.write_text(
        "Simple Title\n\nThis is a technical paragraph.\n\n[[IMAGE:diagram.png]]\n\nAnother paragraph.",
        encoding="utf-8",
    )

    settings = load_settings(Path("/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/config/settings.yaml"))
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
    assert (tmp_path / "data" / "db" / "bm25" / "index.json").exists()
    assert (tmp_path / "data" / "db" / "chroma" / "records.json").exists()


def test_ingestion_pipeline_skips_unchanged_document(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    document_path = tmp_path / "simple.pdf"
    document_path.write_text("Simple Title\n\nOne paragraph.", encoding="utf-8")

    settings = load_settings(Path("/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/config/settings.yaml"))
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
