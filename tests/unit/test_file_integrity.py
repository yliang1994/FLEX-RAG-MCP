from __future__ import annotations

import sqlite3

from libs.loader.file_integrity import SQLiteIntegrityChecker


def test_compute_sha256_is_stable_for_same_file(tmp_path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello integrity\n", encoding="utf-8")
    checker = SQLiteIntegrityChecker(tmp_path / "data" / "db" / "ingestion_history.db")

    first = checker.compute_sha256(file_path)
    second = checker.compute_sha256(file_path)

    assert first == second
    assert len(first) == 64


def test_mark_success_enables_skip(tmp_path) -> None:
    db_path = tmp_path / "data" / "db" / "ingestion_history.db"
    checker = SQLiteIntegrityChecker(db_path)

    checker.mark_success(
        file_hash="abc123",
        file_path="docs/manual.pdf",
        file_size=1024,
        chunk_count=7,
    )

    assert checker.should_skip("abc123") is True
    assert db_path.exists()


def test_mark_failed_does_not_enable_skip(tmp_path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "data" / "db" / "ingestion_history.db")

    checker.mark_failed("failed-hash", "network timeout")

    assert checker.should_skip("failed-hash") is False


def test_sqlite_integrity_checker_uses_wal_mode(tmp_path) -> None:
    db_path = tmp_path / "data" / "db" / "ingestion_history.db"
    SQLiteIntegrityChecker(db_path)

    with sqlite3.connect(db_path) as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

    assert journal_mode.lower() == "wal"
