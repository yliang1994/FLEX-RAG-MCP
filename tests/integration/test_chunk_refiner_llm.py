from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import load_settings
from core.types import Chunk
from ingestion.transform.chunk_refiner import ChunkRefiner
from libs.llm.base_llm import BaseLLM, LLMResponse


class RecordingLLM(BaseLLM):
    provider_name = "recording"

    def __init__(self) -> None:
        super().__init__(model="recording")

    def chat(self, messages):  # type: ignore[override]
        return LLMResponse(content=messages[-1].content.replace("Messy text", "Clean text"))


def test_chunk_refiner_runs_with_injected_llm() -> None:
    settings = load_settings(Path("config/settings.yaml"))
    settings.ingestion.chunk_refiner.use_llm = True
    refiner = ChunkRefiner(settings, llm=RecordingLLM())

    refined = refiner.transform([Chunk(id="chunk-1", text="Messy   text", metadata={})])[0]

    assert refined.metadata["refined_by"] == "llm"
    assert "Clean text" in refined.text


def test_chunk_refiner_gracefully_falls_back_when_llm_configuration_is_invalid() -> None:
    settings = load_settings(Path("config/settings.yaml"))
    settings.ingestion.chunk_refiner.use_llm = True

    class BrokenLLM(BaseLLM):
        provider_name = "broken"

        def __init__(self) -> None:
            super().__init__(model="broken")

        def chat(self, messages):  # type: ignore[override]
            raise RuntimeError("invalid model")

    refiner = ChunkRefiner(settings, llm=BrokenLLM())
    refined = refiner.transform([Chunk(id="chunk-1", text="Page 2\n\nMessy   text", metadata={})])[0]

    assert refined.metadata["refined_by"] == "rule"
    assert refined.text == "Messy text"


@pytest.mark.skipif(
    load_settings(Path("config/settings.yaml")).llm.provider == "stub",
    reason="real llm integration requires a configured non-stub provider",
)
def test_chunk_refiner_real_llm_smoke() -> None:
    settings = load_settings(Path("config/settings.yaml"))
    settings.ingestion.chunk_refiner.use_llm = True
    refiner = ChunkRefiner(settings)

    refined = refiner.transform([Chunk(id="chunk-1", text="Messy   text", metadata={})])[0]

    assert refined.text.strip()
