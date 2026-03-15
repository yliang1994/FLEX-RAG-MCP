from __future__ import annotations

import math
from pathlib import Path

from core.types import ChunkRecord
from ingestion.storage.bm25_indexer import BM25Indexer


def make_records() -> list[ChunkRecord]:
    return [
        ChunkRecord(
            id="chunk-1",
            text="apple banana apple",
            metadata={"source": "a"},
            sparse_vector={"apple": 1.0, "banana": 0.5},
        ),
        ChunkRecord(
            id="chunk-2",
            text="banana carrot",
            metadata={"source": "b"},
            sparse_vector={"banana": 1.0, "carrot": 1.0},
        ),
        ChunkRecord(
            id="chunk-3",
            text="durian only",
            metadata={"source": "c"},
            sparse_vector={"durian": 1.0, "only": 1.0},
        ),
    ]


def test_bm25_indexer_build_load_and_query_roundtrip(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path)
    indexer.build(make_records())

    reloaded = BM25Indexer(tmp_path)
    reloaded.load()
    results = reloaded.search("apple banana", top_k=2)

    assert [result["chunk_id"] for result in results] == ["chunk-1", "chunk-2"]
    assert (tmp_path / "index.json").exists()
    assert (tmp_path / "docs.json").exists()


def test_bm25_indexer_computes_expected_idf_for_rare_term(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path)
    indexer.build(make_records())

    expected = math.log((3 - 1 + 0.5) / (1 + 0.5))

    assert indexer.index["apple"]["idf"] == expected


def test_bm25_indexer_supports_incremental_upsert(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path)
    indexer.build(make_records()[:2])
    indexer.upsert(make_records()[2:])

    reloaded = BM25Indexer(tmp_path)
    reloaded.load()
    results = reloaded.search("durian", top_k=1)

    assert results[0]["chunk_id"] == "chunk-3"


def test_bm25_indexer_rebuild_replaces_previous_corpus(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path)
    indexer.build(make_records())
    indexer.build([ChunkRecord(id="chunk-4", text="kiwi", metadata={}, sparse_vector={"kiwi": 1.0})])

    reloaded = BM25Indexer(tmp_path)
    reloaded.load()

    assert reloaded.search("apple") == []
    assert reloaded.search("kiwi", top_k=1)[0]["chunk_id"] == "chunk-4"
