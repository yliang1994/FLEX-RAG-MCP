from __future__ import annotations

import pytest

from core.settings import Settings, VectorStoreSettings, load_settings
from core.types import QueryMatch, VectorRecord
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import (
    InlineVectorStoreSettings,
    UnsupportedVectorStoreError,
    VectorStoreFactory,
)


class FakeVectorStore(BaseVectorStore):
    backend_name = "fake"

    def __init__(self, persist_path: str, **kwargs: object) -> None:
        super().__init__(persist_path, **kwargs)
        self.records: list[VectorRecord] = []

    def upsert(self, records: list[VectorRecord], trace: object | None = None) -> int:
        self.records.extend(records)
        return len(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
        trace: object | None = None,
    ) -> list[QueryMatch]:
        return [
            QueryMatch(
                id="fake-1",
                score=1.0,
                text="hello",
                metadata={"collection": "unit"},
            )
        ][:top_k]

    def get_by_ids(self, ids: list[str], trace: object | None = None) -> list[dict[str, object]]:
        return [
            {
                "id": record_id,
                "text": "hello",
                "metadata": {"collection": "unit"},
            }
            for record_id in ids
        ]


def _settings_with_vector_store(backend: str, persist_path: str) -> Settings:
    base = load_settings("config/settings.yaml")
    return Settings(
        llm=base.llm,
        embedding=base.embedding,
        splitter=base.splitter,
        vector_store=VectorStoreSettings(backend=backend, persist_path=persist_path),
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=base.evaluation,
        observability=base.observability,
    )


def test_vector_store_factory_routes_from_settings_object() -> None:
    store = VectorStoreFactory.create(_settings_with_vector_store("chroma", "./data/db/chroma"))

    inserted = store.upsert(
        [
            VectorRecord(
                id="doc-1",
                text="alpha chunk",
                vector=[1.0, 0.0],
                metadata={"collection": "docs", "source": "handbook"},
            ),
            VectorRecord(
                id="doc-2",
                text="beta chunk",
                vector=[0.0, 1.0],
                metadata={"collection": "docs", "source": "guide"},
            ),
        ]
    )
    matches = store.query([1.0, 0.0], top_k=1, filters={"collection": "docs"})

    assert store.backend_name == "chroma"
    assert inserted == 2
    assert len(matches) == 1
    assert matches[0].id == "doc-1"
    assert isinstance(matches[0].score, float)


def test_vector_store_contract_filters_and_shapes_results() -> None:
    store = VectorStoreFactory.create(InlineVectorStoreSettings(backend="stub", persist_path="./tmp"))
    store.upsert(
        [
            VectorRecord(
                id="doc-a",
                text="gamma",
                vector=[1.0, 1.0],
                metadata={"collection": "keep", "lang": "zh"},
            ),
            VectorRecord(
                id="doc-b",
                text="delta",
                vector=[1.0, -1.0],
                metadata={"collection": "drop", "lang": "en"},
            ),
        ]
    )

    matches = store.query([1.0, 1.0], top_k=5, filters={"collection": "keep"})

    assert [match.id for match in matches] == ["doc-a"]
    assert all(isinstance(match, QueryMatch) for match in matches)
    assert matches[0].metadata["lang"] == "zh"


def test_vector_store_factory_supports_custom_registration() -> None:
    VectorStoreFactory.register("fake", FakeVectorStore)

    store = VectorStoreFactory.create(
        InlineVectorStoreSettings(backend="fake", persist_path="./tmp/fake")
    )

    assert isinstance(store, FakeVectorStore)
    assert store.query([1.0], top_k=1)[0].id == "fake-1"


def test_vector_store_factory_rejects_unknown_backend() -> None:
    with pytest.raises(UnsupportedVectorStoreError, match="unsupported-backend"):
        VectorStoreFactory.create(
            InlineVectorStoreSettings(backend="unsupported-backend", persist_path="./tmp")
        )
