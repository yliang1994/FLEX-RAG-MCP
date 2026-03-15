from __future__ import annotations

from core.settings import load_settings
from core.types import Document
from ingestion.chunking.document_chunker import DocumentChunker


class FakeSplitter:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def split_text(self, text: str) -> list[str]:
        return list(self.outputs)


def test_document_chunker_creates_deterministic_chunk_ids(monkeypatch) -> None:
    settings = load_settings("config/settings.yaml")
    monkeypatch.setattr(
        "ingestion.chunking.document_chunker.SplitterFactory.create",
        lambda settings: FakeSplitter(["alpha", "beta"]),
    )
    document = Document(
        id="doc-1",
        text="alpha\n\nbeta",
        metadata={"source_path": "docs/demo.pdf", "doc_type": "pdf"},
    )

    chunker = DocumentChunker(settings)
    first = chunker.split_document(document)
    second = chunker.split_document(document)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert len({chunk.id for chunk in first}) == 2
    assert all(chunk.source_ref == "doc-1" for chunk in first)


def test_document_chunker_inherits_metadata_and_chunk_index(monkeypatch) -> None:
    settings = load_settings("config/settings.yaml")
    monkeypatch.setattr(
        "ingestion.chunking.document_chunker.SplitterFactory.create",
        lambda settings: FakeSplitter(["alpha", "beta"]),
    )
    document = Document(
        id="doc-1",
        text="alpha\n\nbeta",
        metadata={"source_path": "docs/demo.pdf", "title": "Demo"},
    )

    chunks = DocumentChunker(settings).split_document(document)

    assert chunks[0].metadata["source_path"] == "docs/demo.pdf"
    assert chunks[0].metadata["title"] == "Demo"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1


def test_document_chunker_distributes_images_only_to_referencing_chunks(monkeypatch) -> None:
    settings = load_settings("config/settings.yaml")
    monkeypatch.setattr(
        "ingestion.chunking.document_chunker.SplitterFactory.create",
        lambda settings: FakeSplitter(["Intro [IMAGE: img-1]", "Tail section"]),
    )
    document = Document(
        id="doc-1",
        text="Intro [IMAGE: img-1]\n\nTail section",
        metadata={
            "source_path": "docs/demo.pdf",
            "images": [
                {"id": "img-1", "path": "data/images/default/img-1.png", "page": 1, "text_offset": 6, "text_length": 14, "position": {}},
                {"id": "img-2", "path": "data/images/default/img-2.png", "page": 2, "text_offset": 30, "text_length": 14, "position": {}},
            ],
        },
    )

    chunks = DocumentChunker(settings).split_document(document)

    assert chunks[0].metadata["image_refs"] == ["img-1"]
    assert [image["id"] for image in chunks[0].metadata["images"]] == ["img-1"]
    assert "images" not in chunks[1].metadata
    assert "image_refs" not in chunks[1].metadata
