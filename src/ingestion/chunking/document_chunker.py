"""Document-to-chunk adapter built on top of libs.splitter."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy

from core.settings import Settings
from core.types import Chunk, Document
from libs.splitter.splitter_factory import SplitterFactory


IMAGE_PLACEHOLDER_RE = re.compile(r"\[IMAGE:\s*(?P<image_id>[^\]]+)\]")


class DocumentChunker:
    """Adapt a Document object into Chunk objects with stable metadata."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.splitter = SplitterFactory.create(settings)

    def split_document(self, document: Document) -> list[Chunk]:
        chunk_texts = self.splitter.split_text(document.text)
        chunks: list[Chunk] = []
        search_start = 0

        for index, chunk_text in enumerate(chunk_texts):
            start_offset = document.text.find(chunk_text, search_start)
            if start_offset == -1:
                start_offset = search_start
            end_offset = start_offset + len(chunk_text)
            search_start = end_offset

            chunks.append(
                Chunk(
                    id=self._generate_chunk_id(document.id, index, chunk_text),
                    text=chunk_text,
                    metadata=self._inherit_metadata(document, index, chunk_text),
                    start_offset=start_offset,
                    end_offset=end_offset,
                    source_ref=document.id,
                )
            )
        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{text_hash}"

    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str) -> dict:
        metadata = deepcopy(document.metadata)
        metadata["chunk_index"] = chunk_index

        document_images = metadata.pop("images", None)
        image_refs = [match.group("image_id").strip() for match in IMAGE_PLACEHOLDER_RE.finditer(chunk_text)]
        if image_refs and isinstance(document_images, list):
            matched_images = [image for image in document_images if image.get("id") in image_refs]
            if matched_images:
                metadata["images"] = matched_images
            metadata["image_refs"] = image_refs

        return metadata
