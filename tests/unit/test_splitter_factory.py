from __future__ import annotations

import pytest

from core.settings import Settings, SplitterSettings, load_settings
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import (
    InlineSplitterSettings,
    SplitterFactory,
    UnsupportedSplitterError,
)


class FakeSplitter(BaseSplitter):
    method_name = "fake"

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        return [text.upper()]


def _settings_with_splitter(method: str, chunk_size: int, chunk_overlap: int = 0) -> Settings:
    base = load_settings("config/settings.yaml")
    return Settings(
        llm=base.llm,
        embedding=base.embedding,
        splitter=SplitterSettings(
            method=method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        vector_store=base.vector_store,
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=base.evaluation,
        observability=base.observability,
    )


def test_splitter_factory_routes_from_settings_object() -> None:
    splitter = SplitterFactory.create(_settings_with_splitter("recursive", 18, 4))

    chunks = splitter.split_text("alpha beta gamma\ndelta epsilon\n\nzeta eta theta")

    assert splitter.method_name == "recursive"
    assert chunks
    assert all(len(chunk) <= 18 for chunk in chunks)


def test_splitter_factory_normalizes_method_name() -> None:
    splitter = SplitterFactory.create(
        InlineSplitterSettings(method="  FIXED  ", chunk_size=5, chunk_overlap=1)
    )

    chunks = splitter.split_text("abcdefghij")

    assert splitter.method_name == "fixed"
    assert chunks == ["abcde", "efghi", "ij"]


def test_splitter_factory_supports_custom_registration() -> None:
    SplitterFactory.register("fake", FakeSplitter)

    splitter = SplitterFactory.create(
        InlineSplitterSettings(method="fake", chunk_size=10, chunk_overlap=0)
    )

    assert isinstance(splitter, FakeSplitter)
    assert splitter.split_text("hello") == ["HELLO"]


def test_splitter_factory_rejects_unknown_method() -> None:
    with pytest.raises(UnsupportedSplitterError, match="unsupported-method"):
        SplitterFactory.create(
            InlineSplitterSettings(method="unsupported-method", chunk_size=10, chunk_overlap=0)
        )
