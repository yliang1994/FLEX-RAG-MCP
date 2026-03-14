"""Minimal Ollama LLM provider."""

from __future__ import annotations

import json
from typing import Any, Callable
from urllib import error, request

from libs.llm.base_llm import BaseLLM, ChatMessage, LLMResponse
from libs.llm.openai_llm import _validate_messages


Transport = Callable[[dict[str, Any]], dict[str, Any]]


class OllamaLLM(BaseLLM):
    """Placeholder implementation for local Ollama models."""

    provider_name = "ollama"

    def __init__(
        self,
        model: str,
        api_key: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.base_url = str(kwargs.get("base_url", "http://localhost:11434"))
        self.timeout = float(kwargs.get("timeout", 10.0))
        self._transport: Transport = kwargs.get("transport", self._default_transport)

    def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        _validate_messages(self.provider_name, messages)
        payload = {
            "model": self.model,
            "messages": [{"role": item.role, "content": item.content} for item in messages],
            "stream": False,
        }

        try:
            response_payload = self._transport(payload)
        except TimeoutError as exc:
            raise RuntimeError("ollama: timeout_error: request to local ollama endpoint timed out") from exc
        except OSError as exc:
            raise RuntimeError("ollama: connection_error: failed to reach local ollama endpoint") from exc

        try:
            content = str(response_payload["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError("ollama: response_error: invalid response payload") from exc

        return LLMResponse(
            content=content,
            raw={
                "provider": self.provider_name,
                "message_count": len(messages),
                "base_url": self.base_url,
            },
        )

    def _default_transport(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a local echo payload unless tests replace the transport."""

        return {"message": {"content": f"[ollama:{payload['model']}] {payload['messages'][-1]['content']}"}}

    def _http_transport(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        endpoint = f"{self.base_url.rstrip('/')}/api/chat"
        req = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, TimeoutError):
                raise TimeoutError from exc
            raise OSError from exc
