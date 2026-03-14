from __future__ import annotations

import re

import pytest

from core.settings import LLMSettings, Settings, load_settings
from libs.llm.base_vision_llm import BaseVisionLLM, VisionResponse
from libs.llm.llm_factory import LLMFactory, UnsupportedLLMProviderError


class FakeVisionLLM(BaseVisionLLM):
    provider_name = "fake-vision"

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: object | None = None,
    ) -> VisionResponse:
        processed = self.preprocess_image(image_path)
        kind = "bytes" if isinstance(processed, bytes) else "path"
        return VisionResponse(content=f"{text}:{kind}", raw={"provider": self.provider_name})


def _settings_with_llm(provider: str, model: str) -> Settings:
    base = load_settings("config/settings.yaml")
    return Settings(
        llm=LLMSettings(provider=provider, model=model, api_key="secret"),
        embedding=base.embedding,
        splitter=base.splitter,
        vector_store=base.vector_store,
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=base.evaluation,
        observability=base.observability,
    )


def test_vision_llm_factory_routes_from_settings_object() -> None:
    vision_llm = LLMFactory.create_vision_llm(_settings_with_llm("azure", "gpt-4o"))

    response = vision_llm.chat_with_image("describe this", "/tmp/example.png")

    assert vision_llm.provider_name == "azure"
    assert response.content == "[azure-vision:gpt-4o] describe this"
    assert response.raw["image_kind"] == "path"


def test_vision_llm_factory_supports_custom_registration() -> None:
    LLMFactory.register_vision("fake", FakeVisionLLM)

    vision_llm = LLMFactory.create_vision_llm(LLMSettings(provider="fake", model="vision-test"))

    assert isinstance(vision_llm, FakeVisionLLM)
    assert vision_llm.chat_with_image("look", b"img").content == "look:bytes"


def test_vision_llm_factory_rejects_unknown_provider() -> None:
    with pytest.raises(
        UnsupportedLLMProviderError,
        match=re.escape("Unsupported vision llm provider: unsupported. Supported providers: azure, fake, stub"),
    ):
        LLMFactory.create_vision_llm(LLMSettings(provider="unsupported", model="x"))


def test_base_vision_llm_exposes_preprocess_extension_point() -> None:
    class PreprocessingVisionLLM(FakeVisionLLM):
        def preprocess_image(self, image: str | bytes) -> str | bytes:
            return b"compressed"

    vision_llm = PreprocessingVisionLLM(model="vision-test")

    response = vision_llm.chat_with_image("look", "/tmp/original.png")

    assert response.content == "look:bytes"
