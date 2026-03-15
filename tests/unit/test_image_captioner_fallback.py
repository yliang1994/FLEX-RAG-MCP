from __future__ import annotations

from pathlib import Path

from core.settings import (
    ChunkRefinerSettings,
    EmbeddingSettings,
    EvaluationSettings,
    ImageCaptionerSettings,
    IngestionSettings,
    LLMSettings,
    MetadataEnricherSettings,
    ObservabilitySettings,
    RetrievalSettings,
    RerankSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.image_captioner import ImageCaptioner
from libs.llm.base_vision_llm import BaseVisionLLM, VisionResponse


class FakeVisionLLM(BaseVisionLLM):
    provider_name = "fake-vision"

    def __init__(self, fail: bool = False) -> None:
        super().__init__(model="fake-vision")
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def chat_with_image(self, text: str, image_path: str | bytes, trace=None):  # type: ignore[override]
        self.calls.append((text, str(image_path)))
        if self.fail:
            raise RuntimeError("vision unavailable")
        return VisionResponse(content=f"caption for {Path(str(image_path)).name}")


def make_settings(enabled: bool = False) -> Settings:
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
            metadata_enricher=MetadataEnricherSettings(use_llm=False),
            image_captioner=ImageCaptionerSettings(enabled=enabled),
        ),
    )


def make_chunk() -> Chunk:
    return Chunk(
        id="chunk-1",
        text="This chunk references an image.",
        metadata={
            "image_refs": ["img-1"],
            "images": [{"id": "img-1", "path": "/tmp/image-1.png"}],
        },
    )


def test_image_captioner_generates_captions_when_enabled() -> None:
    vision_llm = FakeVisionLLM()
    captioner = ImageCaptioner(make_settings(enabled=True), vision_llm=vision_llm)

    enriched = captioner.transform([make_chunk()])[0]

    assert enriched.metadata["image_captions"] == {"img-1": "caption for image-1.png"}
    assert "has_unprocessed_images" not in enriched.metadata
    assert vision_llm.calls


def test_image_captioner_marks_unprocessed_when_disabled() -> None:
    captioner = ImageCaptioner(make_settings(enabled=False))

    enriched = captioner.transform([make_chunk()])[0]

    assert enriched.metadata["has_unprocessed_images"] is True
    assert "image_captions" not in enriched.metadata


def test_image_captioner_marks_unprocessed_when_vision_llm_fails() -> None:
    trace = TraceContext()
    captioner = ImageCaptioner(make_settings(enabled=True), vision_llm=FakeVisionLLM(fail=True))

    enriched = captioner.transform([make_chunk()], trace=trace)[0]

    assert enriched.metadata["has_unprocessed_images"] is True
    assert "image_captions" not in enriched.metadata
    assert any(stage.name == "image_captioner.fallback" for stage in trace.stages)


def test_image_captioner_ignores_chunks_without_images() -> None:
    captioner = ImageCaptioner(make_settings(enabled=True), vision_llm=FakeVisionLLM())
    chunk = Chunk(id="chunk-2", text="No images here.", metadata={})

    enriched = captioner.transform([chunk])[0]

    assert enriched.metadata == {}
