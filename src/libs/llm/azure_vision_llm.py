"""Minimal Azure vision-capable LLM provider."""

from __future__ import annotations

from libs.llm.base_vision_llm import BaseVisionLLM, VisionResponse


class AzureVisionLLM(BaseVisionLLM):
    """Placeholder provider used to validate factory routing for vision models."""

    provider_name = "azure"

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: object | None = None,
    ) -> VisionResponse:
        processed_image = self.preprocess_image(image_path)
        image_kind = "bytes" if isinstance(processed_image, bytes) else "path"
        return VisionResponse(
            content=f"[azure-vision:{self.model}] {text}",
            raw={"provider": self.provider_name, "image_kind": image_kind},
        )
