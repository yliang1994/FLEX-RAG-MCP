from __future__ import annotations

from dataclasses import asdict

from core import Chunk, ChunkRecord, Document, ImageRef, RetrievalResult


def test_document_and_image_refs_are_serializable() -> None:
    image_ref = ImageRef(
        id="dochash_1_0",
        path="data/images/default/dochash_1_0.png",
        page=1,
        text_offset=42,
        text_length=21,
        position={"x": 10, "y": 20, "width": 320, "height": 240},
    )
    document = Document(
        id="doc-1",
        text="Intro\n\n[IMAGE: dochash_1_0]",
        metadata={"source_path": "docs/demo.pdf", "images": [asdict(image_ref)]},
    )

    payload = asdict(document)

    assert payload["id"] == "doc-1"
    assert payload["metadata"]["source_path"] == "docs/demo.pdf"
    assert payload["metadata"]["images"][0]["id"] == "dochash_1_0"
    assert payload["metadata"]["images"][0]["text_offset"] == 42


def test_chunk_and_chunk_record_keep_stable_fields() -> None:
    chunk = Chunk(
        id="chunk-1",
        text="chunk body",
        metadata={"source_path": "docs/demo.pdf", "image_refs": ["img-1"]},
        start_offset=10,
        end_offset=20,
        source_ref="doc-1#10:20",
    )
    record = ChunkRecord(
        id="chunk-1",
        text="chunk body",
        metadata={"source_path": "docs/demo.pdf"},
        dense_vector=[0.1, 0.2],
        sparse_vector={"chunk": 1.5},
    )

    chunk_payload = asdict(chunk)
    record_payload = asdict(record)

    assert chunk_payload["start_offset"] == 10
    assert chunk_payload["end_offset"] == 20
    assert chunk_payload["metadata"]["image_refs"] == ["img-1"]
    assert record_payload["dense_vector"] == [0.1, 0.2]
    assert record_payload["sparse_vector"] == {"chunk": 1.5}


def test_retrieval_result_is_serializable_with_metadata() -> None:
    result = RetrievalResult(
        chunk_id="chunk-1",
        score=0.95,
        text="answer text",
        metadata={"source_path": "docs/demo.pdf", "page": 3},
    )

    payload = asdict(result)

    assert payload == {
        "chunk_id": "chunk-1",
        "score": 0.95,
        "text": "answer text",
        "metadata": {"source_path": "docs/demo.pdf", "page": 3},
    }
