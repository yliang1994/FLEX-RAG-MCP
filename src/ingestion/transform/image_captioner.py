"""Image captioner with optional Vision LLM support and non-blocking fallback."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory


DEFAULT_PROMPT = """Describe the image identified by {image_id} in the context of this chunk.
Return one concise caption sentence that helps retrieval.

Chunk context:
{text}
"""


class ImageCaptioner(BaseTransform):
    """Generate image captions when image refs and a Vision LLM are available."""

    def __init__(
        self,
        settings: Settings,
        vision_llm: BaseVisionLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.enabled = settings.ingestion.image_captioner.enabled
        self.vision_llm = vision_llm if vision_llm is not None else self._build_vision_llm()
        self.prompt_template = self._load_prompt(prompt_path)

    def transform(self, chunks: list[Chunk], trace: TraceContext | None = None) -> list[Chunk]:
        transformed: list[Chunk] = []

        for chunk in chunks:
            metadata = deepcopy(chunk.metadata)
            image_entries = self._resolve_image_entries(metadata)

            if not image_entries:
                transformed.append(chunk)
                continue

            if not self.enabled or self.vision_llm is None:
                metadata["has_unprocessed_images"] = True
                transformed.append(self._copy_chunk(chunk, metadata))
                continue

            captions: dict[str, str] = {}
            failed = False
            for image_entry in image_entries:
                image_id = str(image_entry.get("id", "")).strip()
                image_path = str(image_entry.get("path", "")).strip()
                if not image_id or not image_path:
                    failed = True
                    continue

                prompt = self.prompt_template.format(text=chunk.text, image_id=image_id)
                try:
                    response = self.vision_llm.chat_with_image(prompt, image_path, trace=trace)
                except Exception as exc:
                    failed = True
                    if trace is not None:
                        trace.record_stage(
                            "image_captioner.fallback",
                            chunk_id=chunk.id,
                            image_id=image_id,
                            reason=str(exc),
                        )
                    continue
                captions[image_id] = response.content.strip()

            if captions:
                metadata["image_captions"] = captions
            if failed or len(captions) != len(image_entries):
                metadata["has_unprocessed_images"] = True
            transformed.append(self._copy_chunk(chunk, metadata))

        if trace is not None:
            trace.record_stage(
                "image_captioner.transform",
                chunk_count=len(chunks),
                enabled=self.enabled and self.vision_llm is not None,
            )
        return transformed

    def _build_vision_llm(self) -> BaseVisionLLM | None:
        if not self.enabled:
            return None
        return LLMFactory.create_vision_llm(self.settings)

    def _resolve_image_entries(self, metadata: dict) -> list[dict]:
        images = metadata.get("images")
        if isinstance(images, list):
            return [image for image in images if isinstance(image, dict)]
        return []

    def _load_prompt(self, prompt_path: str | Path | None) -> str:
        resolved = Path(prompt_path) if prompt_path is not None else Path("config/prompts/image_captioning.txt")
        if resolved.exists():
            template = resolved.read_text(encoding="utf-8").strip()
            if "{text}" not in template:
                template = f"{template}\n\n{{text}}"
            if "{image_id}" not in template:
                template = f"{template}\n\nImage: {{image_id}}"
            return template
        return DEFAULT_PROMPT

    def _copy_chunk(self, chunk: Chunk, metadata: dict) -> Chunk:
        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )
