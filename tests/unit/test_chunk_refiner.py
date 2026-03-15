from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.settings import (
    ChunkRefinerSettings,
    EmbeddingSettings,
    EvaluationSettings,
    IngestionSettings,
    LLMSettings,
    ObservabilitySettings,
    RetrievalSettings,
    RerankSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.chunk_refiner import ChunkRefiner
from libs.llm.base_llm import BaseLLM, LLMResponse


class FakeLLM(BaseLLM):
    provider_name = "fake"

    def __init__(self, responses: list[str] | None = None, error: Exception | None = None) -> None:
        super().__init__(model="fake-model")
        self.responses = responses or []
        self.error = error
        self.calls: list[str] = []

    def chat(self, messages):  # type: ignore[override]
        self.calls.append(messages[-1].content)
        if self.error is not None:
            raise self.error
        return LLMResponse(content=self.responses.pop(0))


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
        ingestion=IngestionSettings(chunk_refiner=ChunkRefinerSettings(use_llm=use_llm)),
    )


@pytest.fixture
def fixture_cases() -> list[dict[str, object]]:
    fixture_path = Path("tests/fixtures/noisy_chunks.json")
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_rule_based_refine_matches_fixture_expectations(fixture_cases: list[dict[str, object]]) -> None:
    refiner = ChunkRefiner(make_settings())

    for case in fixture_cases:
        refined = refiner._rule_based_refine(case["input"])  # type: ignore[index]
        for expected in case["expected_contains"]:  # type: ignore[index]
            assert expected in refined, case["name"]  # type: ignore[index]
        for unexpected in case["expected_absent"]:  # type: ignore[index]
            assert unexpected not in refined, case["name"]  # type: ignore[index]


def test_transform_marks_rule_metadata_when_llm_disabled() -> None:
    refiner = ChunkRefiner(make_settings(use_llm=False))
    chunk = Chunk(id="chunk-1", text="Page 1\n\nUseful   text.", metadata={"source": "pdf"})

    refined_chunk = refiner.transform([chunk])[0]

    assert refined_chunk.text == "Useful text."
    assert refined_chunk.metadata["refined_by"] == "rule"
    assert refined_chunk.metadata["source"] == "pdf"


def test_transform_uses_llm_output_when_enabled() -> None:
    llm = FakeLLM(responses=["Clean rewritten text."])
    refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)
    chunk = Chunk(id="chunk-1", text="Messy   text", metadata={})

    refined_chunk = refiner.transform([chunk])[0]

    assert refined_chunk.text == "Clean rewritten text."
    assert refined_chunk.metadata["refined_by"] == "llm"
    assert "{text}" not in llm.calls[0]
    assert "Messy text" in llm.calls[0]


def test_transform_falls_back_to_rule_when_llm_fails() -> None:
    trace = TraceContext()
    llm = FakeLLM(error=RuntimeError("boom"))
    refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)
    chunk = Chunk(id="chunk-1", text="Page 7\n\nMessy   text", metadata={})

    refined_chunk = refiner.transform([chunk], trace=trace)[0]

    assert refined_chunk.text == "Messy text"
    assert refined_chunk.metadata["refined_by"] == "rule"
    assert any(stage.name == "chunk_refiner.llm_fallback" for stage in trace.stages)


def test_transform_preserves_original_chunk_when_rule_processing_crashes(monkeypatch: pytest.MonkeyPatch) -> None:
    refiner = ChunkRefiner(make_settings())
    chunk = Chunk(id="chunk-1", text="Original text", metadata={})

    def raise_error(_: str) -> str:
        raise ValueError("bad rule")

    monkeypatch.setattr(refiner, "_rule_based_refine", raise_error)

    refined_chunk = refiner.transform([chunk])[0]

    assert refined_chunk.text == "Original text"
    assert refined_chunk.metadata["refined_by"] == "original"
    assert refined_chunk.metadata["refine_error"] == "bad rule"


def test_load_prompt_appends_text_placeholder_when_missing(tmp_path: Path) -> None:
    prompt_path = tmp_path / "chunk_refinement.txt"
    prompt_path.write_text("Rewrite this chunk.", encoding="utf-8")

    refiner = ChunkRefiner(make_settings(), prompt_path=prompt_path)

    assert refiner.prompt_template.endswith("{text}")


def test_trace_records_transform_summary() -> None:
    trace = TraceContext()
    refiner = ChunkRefiner(make_settings())
    chunks = [Chunk(id="chunk-1", text="Text", metadata={})]

    refiner.transform(chunks, trace=trace)

    assert trace.stages[-1].name == "chunk_refiner.transform"
    assert trace.stages[-1].details["chunk_count"] == 1
