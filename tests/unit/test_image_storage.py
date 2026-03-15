from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.storage.image_storage import ImageStorage


def test_image_storage_saves_bytes_and_persists_mapping(tmp_path: Path) -> None:
    storage = ImageStorage(
        image_root=tmp_path / "images",
        db_path=tmp_path / "db" / "image_index.db",
    )

    saved_path = storage.save("img-1", b"fake-image", collection="manuals")

    assert Path(saved_path).exists()
    assert storage.get_path("img-1") == saved_path



def test_image_storage_copies_source_file_and_can_be_reloaded(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"png-data")
    db_path = tmp_path / "db" / "image_index.db"
    image_root = tmp_path / "images"

    first = ImageStorage(image_root=image_root, db_path=db_path)
    saved_path = first.save("img-2", str(source), collection="reports")

    second = ImageStorage(image_root=image_root, db_path=db_path)

    assert second.get_path("img-2") == saved_path
    assert Path(saved_path).read_bytes() == b"png-data"



def test_image_storage_updates_existing_mapping(tmp_path: Path) -> None:
    storage = ImageStorage(
        image_root=tmp_path / "images",
        db_path=tmp_path / "db" / "image_index.db",
    )

    first = storage.save("img-3", b"first", collection="c1")
    second = storage.save("img-3", b"second", collection="c2")

    assert first != second
    assert storage.get_path("img-3") == second
    assert Path(second).read_bytes() == b"second"



def test_image_storage_rejects_missing_source_file(tmp_path: Path) -> None:
    storage = ImageStorage(
        image_root=tmp_path / "images",
        db_path=tmp_path / "db" / "image_index.db",
    )

    with pytest.raises(FileNotFoundError, match="image file not found"):
        storage.save("img-4", str(tmp_path / "missing.png"))
