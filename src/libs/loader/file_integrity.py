"""File integrity helpers for incremental ingestion."""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class FileIntegrityChecker(ABC):
    """Abstract interface for file-hash based ingestion deduplication."""

    @abstractmethod
    def compute_sha256(self, path: str | Path) -> str:
        """Return the SHA256 hex digest for the given file path."""

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """Return whether the file has already been processed successfully."""

    @abstractmethod
    def mark_success(self, file_hash: str, file_path: str, **kwargs: Any) -> None:
        """Persist a successful processing record."""

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """Persist a failed processing record."""


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """SQLite-backed integrity checker using WAL mode for safe local writes."""

    def __init__(self, db_path: str | Path = "data/db/ingestion_history.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def compute_sha256(self, path: str | Path) -> str:
        file_path = Path(path)
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ? AND status = 'success'",
                (file_hash,),
            ).fetchone()
        return row is not None

    def mark_success(self, file_hash: str, file_path: str, **kwargs: Any) -> None:
        payload = {
            "file_size": kwargs.get("file_size"),
            "chunk_count": kwargs.get("chunk_count"),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, file_path, file_size, status, error_msg, chunk_count
                ) VALUES (?, ?, ?, 'success', NULL, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    file_size = excluded.file_size,
                    status = 'success',
                    error_msg = NULL,
                    chunk_count = excluded.chunk_count,
                    processed_at = CURRENT_TIMESTAMP
                """,
                (file_hash, file_path, payload["file_size"], payload["chunk_count"]),
            )

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, file_path, file_size, status, error_msg, chunk_count
                ) VALUES (?, '', NULL, 'failed', ?, NULL)
                ON CONFLICT(file_hash) DO UPDATE SET
                    status = 'failed',
                    error_msg = excluded.error_msg,
                    processed_at = CURRENT_TIMESTAMP
                """,
                (file_hash, error_msg),
            )

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'processing')),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_msg TEXT,
                    chunk_count INTEGER
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_at ON ingestion_history(processed_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
