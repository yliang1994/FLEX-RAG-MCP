from __future__ import annotations

from libs.vector_store.chroma_store import ChromaStore
from core.types import VectorRecord


def test_chroma_store_roundtrip_persists_and_queries(tmp_path) -> None:
    persist_path = tmp_path / "chroma"

    store = ChromaStore(str(persist_path))
    inserted = store.upsert(
        [
            VectorRecord(
                id="doc-1",
                text="alpha chunk",
                vector=[1.0, 0.0],
                metadata={"collection": "docs", "source": "alpha"},
            ),
            VectorRecord(
                id="doc-2",
                text="beta chunk",
                vector=[0.0, 1.0],
                metadata={"collection": "docs", "source": "beta"},
            ),
        ]
    )

    reloaded = ChromaStore(str(persist_path))
    matches = reloaded.query([1.0, 0.0], top_k=1, filters={"collection": "docs"})

    assert inserted == 2
    assert (persist_path / "records.json").exists()
    assert len(matches) == 1
    assert matches[0].id == "doc-1"
    assert matches[0].text == "alpha chunk"


def test_chroma_store_roundtrip_respects_top_k_and_filters(tmp_path) -> None:
    store = ChromaStore(str(tmp_path / "chroma"))
    store.upsert(
        [
            VectorRecord(id="a", text="A", vector=[1.0, 0.0], metadata={"collection": "keep"}),
            VectorRecord(id="b", text="B", vector=[0.9, 0.1], metadata={"collection": "keep"}),
            VectorRecord(id="c", text="C", vector=[0.0, 1.0], metadata={"collection": "drop"}),
        ]
    )

    matches = store.query([1.0, 0.0], top_k=2, filters={"collection": "keep"})

    assert [match.id for match in matches] == ["a", "b"]
