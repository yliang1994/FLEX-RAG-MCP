from __future__ import annotations

from pathlib import Path

from libs.loader.pdf_loader import PdfLoader


def test_pdf_loader_loads_text_only_pdf_fixture(tmp_path: Path) -> None:
    pdf_path = tmp_path / "simple.pdf"
    pdf_path.write_text("# Simple Document\n\nThis is a plain text PDF fixture.\n", encoding="utf-8")
    loader = PdfLoader(image_output_dir=tmp_path / "data" / "images")

    document = loader.load(pdf_path)

    assert document.metadata["source_path"] == str(pdf_path)
    assert document.metadata["doc_type"] == "pdf"
    assert document.metadata["title"] == "Simple Document"
    assert "plain text PDF fixture" in document.text
    assert "images" not in document.metadata


def test_pdf_loader_extracts_image_marker_and_records_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-png-binary")
    pdf_path = tmp_path / "with_images.pdf"
    pdf_path.write_text(
        "# With Images\n\nIntro paragraph.\n\n[[IMAGE:diagram.png]]\n\nClosing paragraph.\n",
        encoding="utf-8",
    )
    loader = PdfLoader(image_output_dir=tmp_path / "data" / "images")

    document = loader.load(pdf_path)
    images = document.metadata["images"]

    assert len(images) == 1
    assert "[IMAGE: " in document.text
    assert Path(images[0]["path"]).exists()
    assert images[0]["text_length"] == len(f"[IMAGE: {images[0]['id']}]")
    assert document.text[images[0]["text_offset"] : images[0]["text_offset"] + images[0]["text_length"]] == f"[IMAGE: {images[0]['id']}]"
