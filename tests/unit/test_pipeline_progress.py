from __future__ import annotations

from pathlib import Path

from core.settings import load_settings
from ingestion.pipeline import IngestionPipeline
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


def test_pipeline_progress_callback_receives_top_level_stages(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    document_path = tmp_path / "sample.pdf"
    document_path.write_text("Title\n\nParagraph one.\n\nParagraph two.", encoding="utf-8")

    settings = load_settings(Path(__file__).resolve().parents[2] / "config" / "settings.yaml")
    settings.vector_store.persist_path = str(tmp_path / "data" / "db" / "chroma")

    pipeline = IngestionPipeline(
        settings,
        integrity_checker=SQLiteIntegrityChecker(tmp_path / "data" / "db" / "file_integrity.db"),
        bm25_indexer=BM25Indexer(tmp_path / "data" / "db" / "bm25"),
        image_storage=ImageStorage(tmp_path / "data" / "images", tmp_path / "data" / "db" / "image_index.db"),
        vector_upserter=VectorUpserter(settings, store=ChromaStore(str(tmp_path / "data" / "db" / "chroma"))),
    )

    progress_calls: list[tuple[str, int, int]] = []

    result = pipeline.run(
        path=document_path,
        collection="manuals",
        on_progress=lambda stage_name, current, total: progress_calls.append((stage_name, current, total)),
    )

    assert result.status == "ingested"
    assert progress_calls == [
        ("load", 1, 5),
        ("split", 2, 5),
        ("transform", 3, 5),
        ("embed", 4, 5),
        ("upsert", 5, 5),
    ]


def test_pipeline_progress_callback_is_optional(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    document_path = tmp_path / "sample.pdf"
    document_path.write_text("Title\n\nParagraph.", encoding="utf-8")

    settings = load_settings(Path(__file__).resolve().parents[2] / "config" / "settings.yaml")
    settings.vector_store.persist_path = str(tmp_path / "data" / "db" / "chroma")

    pipeline = IngestionPipeline(
        settings,
        integrity_checker=SQLiteIntegrityChecker(tmp_path / "data" / "db" / "file_integrity.db"),
        bm25_indexer=BM25Indexer(tmp_path / "data" / "db" / "bm25"),
        image_storage=ImageStorage(tmp_path / "data" / "images", tmp_path / "data" / "db" / "image_index.db"),
        vector_upserter=VectorUpserter(settings, store=ChromaStore(str(tmp_path / "data" / "db" / "chroma"))),
    )

    result = pipeline.run(path=document_path, collection="manuals")

    assert result.status == "ingested"
