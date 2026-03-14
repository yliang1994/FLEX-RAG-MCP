from __future__ import annotations

import re

import pytest

from libs.llm.azure_vision_llm import AzureVisionLLM


def test_azure_vision_llm_supports_image_path_input(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"1234567890")
    captured: dict[str, object] = {}

    def transport(payload: dict[str, object]) -> dict[str, object]:
        captured.update(payload)
        return {"content": "path response"}

    vision_llm = AzureVisionLLM(
        model="gpt-4o",
        azure_endpoint="https://rag-test.openai.azure.com",
        api_version="2024-10-21",
        deployment_name="vision-deploy",
        transport=transport,
        max_image_size=4,
    )

    response = vision_llm.chat_with_image("describe", str(image_path))

    assert captured["image_kind"] == "path"
    assert captured["image"] == b"1234"
    assert response.content == "path response"
    assert response.raw["deployment_name"] == "vision-deploy"


def test_azure_vision_llm_supports_base64_and_bytes_inputs() -> None:
    captured: list[str] = []

    def transport(payload: dict[str, object]) -> dict[str, object]:
        captured.append(str(payload["image_kind"]))
        return {"content": "ok"}

    vision_llm = AzureVisionLLM(model="gpt-4o", transport=transport)

    vision_llm.chat_with_image("describe", "data:image/png;base64,AAA")
    vision_llm.chat_with_image("describe", b"raw-bytes")

    assert captured == ["base64", "bytes"]


@pytest.mark.parametrize(
    ("exc", "expected_error"),
    [
        (TimeoutError("timed out"), "azure_vision: timeout_error: request timed out"),
        (PermissionError("denied"), "azure_vision: auth_error: authentication failed"),
        (OSError("offline"), "azure_vision: connection_error: failed to reach azure endpoint"),
    ],
)
def test_azure_vision_llm_reports_azure_specific_errors(exc: Exception, expected_error: str) -> None:
    def failing_transport(payload: dict[str, object]) -> dict[str, object]:
        raise exc

    vision_llm = AzureVisionLLM(model="gpt-4o", transport=failing_transport)

    with pytest.raises(RuntimeError, match=re.escape(expected_error)):
        vision_llm.chat_with_image("describe", b"img")


def test_azure_vision_llm_rejects_invalid_response_payload() -> None:
    vision_llm = AzureVisionLLM(model="gpt-4o", transport=lambda payload: {"bad": True})

    with pytest.raises(
        RuntimeError,
        match=re.escape("azure_vision: response_error: invalid response payload"),
    ):
        vision_llm.chat_with_image("describe", b"img")
