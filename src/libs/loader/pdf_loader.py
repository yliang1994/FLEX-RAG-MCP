"""Minimal PDF loader shell for normalized markdown-like output."""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from core.types import Document, ImageRef
from libs.loader.base_loader import BaseLoader


IMAGE_MARKER_RE = re.compile(r"\[\[IMAGE:(?P<name>[^\]]+)\]\]")


class PdfLoader(BaseLoader):
    """Load test-friendly PDF fixtures into the shared Document contract."""

    def __init__(self, image_output_dir: str | Path = "data/images") -> None:
        self.image_output_dir = Path(image_output_dir)

    def load(self, path: str | Path) -> Document:
        source_path = Path(path)
        if source_path.suffix.lower() != ".pdf":
            raise ValueError(f"PdfLoader only supports .pdf files: {source_path}")

        raw_text = source_path.read_text(encoding="utf-8")
        doc_hash = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:12]
        normalized_text, image_refs = self._extract_images(raw_text, source_path, doc_hash)
        title = self._extract_title(normalized_text, source_path)

        metadata = {
            "source_path": str(source_path),
            "doc_type": "pdf",
            "title": title,
        }
        if image_refs:
            metadata["images"] = [asdict(image_ref) for image_ref in image_refs]

        return Document(id=doc_hash, text=normalized_text, metadata=metadata)

    def _extract_title(self, text: str, source_path: Path) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped.lstrip("# ")
        return source_path.stem

    def _extract_images(self, text: str, source_path: Path, doc_hash: str) -> tuple[str, list[ImageRef]]:
        output_parts: list[str] = []
        image_refs: list[ImageRef] = []
        last_index = 0

        for seq, match in enumerate(IMAGE_MARKER_RE.finditer(text)):
            output_parts.append(text[last_index:match.start()])
            image_name = match.group("name").strip()
            source_image = source_path.parent / image_name
            image_id = f"{doc_hash}_1_{seq}"
            dest_dir = self.image_output_dir / doc_hash
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{image_id}{source_image.suffix or '.png'}"
            if source_image.exists():
                shutil.copyfile(source_image, dest_path)

            placeholder = f"[IMAGE: {image_id}]"
            current_offset = len("".join(output_parts))
            output_parts.append(placeholder)
            image_refs.append(
                ImageRef(
                    id=image_id,
                    path=str(dest_path),
                    page=1,
                    text_offset=current_offset,
                    text_length=len(placeholder),
                    position={},
                )
            )
            last_index = match.end()

        output_parts.append(text[last_index:])
        return "".join(output_parts), image_refs
