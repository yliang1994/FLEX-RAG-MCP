from __future__ import annotations

import json
import pytest

from core.settings import (
    ChunkRefinerSettings,
    EmbeddingSettings,
    EvaluationSettings,
    IngestionSettings,
    LLMSettings,
    MetadataEnricherSettings,
    ObservabilitySettings,
    RetrievalSettings,
    RerankSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
    load_settings,
)
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.llm.base_llm import BaseLLM, LLMResponse


class FakeMetadataLLM(BaseLLM):
    provider_name = "fake-metadata"

    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        super().__init__(model="fake-model")
        self.content = content or json.dumps(
            {
                "title": "LLM Title",
                "summary": "LLM Summary",
                "tags": ["llm", "metadata"],
            }
        )
        self.error = error

    def chat(self, messages):  # type: ignore[override]
        if self.error is not None:
            raise self.error
        return LLMResponse(content=self.content)


def make_settings(use_llm: bool = False) -> Settings:
    return Settings(
        llm=LLMSettings(provider="stub", model="stub-llm"),
        embedding=EmbeddingSettings(provider="stub", model="stub-embedding"),
        splitter=SplitterSettings(method="recursive", chunk_size=500, chunk_overlap=50),
        vector_store=VectorStoreSettings(backend="chroma", persist_path="./data/db/chroma"),
        retrieval=RetrievalSettings(
            sparse_backend="bm25",
            fusion_algorithm="rrf",
            top_k_dense=20,
            top_k_sparse=20,
            top_k_final=10,
        ),
        rerank=RerankSettings(backend="none", model="", top_m=30),
        evaluation=EvaluationSettings(
            backends=["custom"], golden_test_set="./tests/fixtures/golden_test_set.json"
        ),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
        ingestion=IngestionSettings(
            chunk_refiner=ChunkRefinerSettings(use_llm=False),
            metadata_enricher=MetadataEnricherSettings(use_llm=use_llm),
        ),
    )


def test_rule_mode_populates_required_metadata_fields() -> None:
    enricher = MetadataEnricher(make_settings(use_llm=False))
    chunk = Chunk(
        id="chunk-1",
        text="Distributed systems coordinate work across many nodes and require observability.",
        metadata={},
    )

    enriched = enricher.transform([chunk])[0]

    assert enriched.metadata["title"]
    assert enriched.metadata["summary"]
    assert enriched.metadata["tags"]
    assert enriched.metadata["metadata_enriched_by"] == "rule"


def test_llm_mode_overrides_rule_metadata_with_richer_values() -> None:
    enricher = MetadataEnricher(make_settings(use_llm=True), llm=FakeMetadataLLM())
    chunk = Chunk(id="chunk-1", text="Short input text", metadata={})

    enriched = enricher.transform([chunk])[0]

    assert enriched.metadata["title"] == "LLM Title"
    assert enriched.metadata["summary"] == "LLM Summary"
    assert enriched.metadata["tags"] == ["llm", "metadata"]
    assert enriched.metadata["metadata_enriched_by"] == "llm"


def test_llm_failure_falls_back_to_rule_metadata_and_marks_reason() -> None:
    trace = TraceContext()
    enricher = MetadataEnricher(
        make_settings(use_llm=True),
        llm=FakeMetadataLLM(error=RuntimeError("timeout")),
    )
    chunk = Chunk(id="chunk-1", text="Page 2 cache invalidation patterns", metadata={})

    enriched = enricher.transform([chunk], trace=trace)[0]

    assert enriched.metadata["metadata_enriched_by"] == "rule"
    assert enriched.metadata["metadata_enricher_fallback"] == "llm_failed"
    assert any(stage.name == "metadata_enricher.llm_fallback" for stage in trace.stages)


def test_llm_invalid_payload_falls_back_to_rule_metadata() -> None:
    enricher = MetadataEnricher(
        make_settings(use_llm=True),
        llm=FakeMetadataLLM(content=json.dumps({"title": "", "summary": "x", "tags": []})),
    )
    chunk = Chunk(id="chunk-1", text="Useful text for metadata generation", metadata={})

    enriched = enricher.transform([chunk])[0]

    assert enriched.metadata["metadata_enriched_by"] == "rule"
    assert enriched.metadata["metadata_enricher_fallback"] == "llm_failed"
    assert enriched.metadata["title"]
    assert enriched.metadata["summary"]
    assert enriched.metadata["tags"]


@pytest.mark.skipif(
    load_settings("config/settings.yaml").llm.provider == "stub",
    reason="real llm smoke requires a configured non-stub provider",
)
def test_metadata_enricher_real_llm_smoke() -> None:
    settings = load_settings("config/settings.yaml")
    settings.ingestion.metadata_enricher.use_llm = True
    enricher = MetadataEnricher(settings)

    enriched = enricher.transform([Chunk(id="chunk-1", text="Metadata generation text", metadata={})])[0]

    assert enriched.metadata["title"]
    assert enriched.metadata["summary"]
    assert enriched.metadata["tags"]
