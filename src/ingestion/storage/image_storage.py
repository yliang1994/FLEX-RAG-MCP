"""Image file storage with persistent SQLite image-id index."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path


class ImageStorage:
    """Store image payloads on disk and maintain an image_id to path mapping."""

    def __init__(
        self,
        image_root: str | Path = "data/images",
        db_path: str | Path = "data/db/image_index.db",
    ) -> None:
        self.image_root = Path(image_root)
        self.db_path = Path(db_path)
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def save(self, image_id: str, image: str | bytes, collection: str = "default") -> str:
        if not image_id.strip():
            raise ValueError("image_id must not be empty")

        target_dir = self.image_root / collection
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / self._target_filename(image_id, image)

        if isinstance(image, bytes):
            target_path.write_bytes(image)
        else:
            source_path = Path(image)
            if not source_path.exists():
                raise FileNotFoundError(f"image file not found: {source_path}")
            shutil.copyfile(source_path, target_path)

        self._upsert_mapping(image_id=image_id, path=str(target_path), collection=collection)
        return str(target_path)

    def get_path(self, image_id: str) -> str | None:
        query = "SELECT path FROM image_index WHERE image_id = ?"
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(query, (image_id,)).fetchone()
        return None if row is None else str(row[0])

    def _init_db(self) -> None:
        statement = """
        CREATE TABLE IF NOT EXISTS image_index (
            image_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            collection_name TEXT NOT NULL
        )
        """
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(statement)
            connection.commit()

    def _upsert_mapping(self, image_id: str, path: str, collection: str) -> None:
        statement = """
        INSERT INTO image_index(image_id, path, collection_name)
        VALUES (?, ?, ?)
        ON CONFLICT(image_id) DO UPDATE SET
            path = excluded.path,
            collection_name = excluded.collection_name
        """
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(statement, (image_id, path, collection))
            connection.commit()

    def _target_filename(self, image_id: str, image: str | bytes) -> str:
        if isinstance(image, bytes):
            return f"{image_id}.bin"
        suffix = Path(image).suffix or ".bin"
        return f"{image_id}{suffix}"
