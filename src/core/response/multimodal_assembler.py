"""Assemble MCP image content blocks from retrieval results."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from core.types import RetrievalResult


class MultimodalAssembler:
    """Append image content blocks for retrieval results that reference images."""

    def assemble(self, retrieval_results: list[RetrievalResult]) -> list[dict[str, object]]:
        content: list[dict[str, object]] = []
        seen_paths: set[Path] = set()

        for result in retrieval_results:
            images = result.metadata.get("images")
            if not isinstance(images, list):
                continue

            for image in images:
                if not isinstance(image, dict):
                    continue
                raw_path = image.get("path")
                if not isinstance(raw_path, str) or not raw_path.strip():
                    continue

                image_path = Path(raw_path)
                if image_path in seen_paths or not image_path.exists():
                    continue
                seen_paths.add(image_path)

                mime_type, _ = mimetypes.guess_type(image_path.name)
                content.append(
                    {
                        "type": "image",
                        "mimeType": mime_type or "application/octet-stream",
                        "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                    }
                )

        return content
