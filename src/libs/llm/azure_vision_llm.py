"""Minimal Azure vision-capable LLM provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from libs.llm.base_vision_llm import BaseVisionLLM, VisionResponse


Transport = Callable[[dict[str, Any]], dict[str, Any]]


class AzureVisionLLM(BaseVisionLLM):
    """Mock-friendly Azure Vision provider with multimodal input normalization."""

    provider_name = "azure"

    def __init__(self, model: str, api_key: str = "", **kwargs: Any) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.azure_endpoint = str(kwargs.get("azure_endpoint", "https://example-resource.openai.azure.com"))
        self.api_version = str(kwargs.get("api_version", "2024-10-21"))
        self.deployment_name = str(kwargs.get("deployment_name", model))
        self.max_image_size = int(kwargs.get("max_image_size", 2048))
        self._transport: Transport = kwargs.get("transport", self._default_transport)

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: object | None = None,
    ) -> VisionResponse:
        processed_image = self.preprocess_image(image_path)
        image_kind = self._detect_image_kind(image_path)
        payload = {
            "deployment_name": self.deployment_name,
            "api_version": self.api_version,
            "text": text,
            "image": processed_image,
            "image_kind": image_kind,
        }

        try:
            response_payload = self._transport(payload)
        except TimeoutError as exc:
            raise RuntimeError("azure_vision: timeout_error: request timed out") from exc
        except PermissionError as exc:
            raise RuntimeError("azure_vision: auth_error: authentication failed") from exc
        except OSError as exc:
            raise RuntimeError("azure_vision: connection_error: failed to reach azure endpoint") from exc

        try:
            content = str(response_payload["content"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError("azure_vision: response_error: invalid response payload") from exc

        return VisionResponse(
            content=content,
            raw={
                "provider": self.provider_name,
                "image_kind": image_kind,
                "azure_endpoint": self.azure_endpoint,
                "api_version": self.api_version,
                "deployment_name": self.deployment_name,
            },
        )

    def preprocess_image(self, image: str | bytes) -> str | bytes:
        if isinstance(image, bytes):
            return image[: self.max_image_size]

        image_path = Path(image)
        if image_path.exists():
            data = image_path.read_bytes()
            return data[: self.max_image_size]
        return image

    def _detect_image_kind(self, image: str | bytes) -> str:
        if isinstance(image, bytes):
            return "bytes"
        if str(image).startswith("data:"):
            return "base64"
        return "path"

    def _default_transport(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"content": f"[azure-vision:{self.model}] {payload['text']}"}
