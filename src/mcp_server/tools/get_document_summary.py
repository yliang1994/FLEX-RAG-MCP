"""MCP tool for loading a document summary by document id."""

from __future__ import annotations

import json
from pathlib import Path


def get_document_summary(
    doc_id: str,
    records_path: str | Path = "data/db/chroma/records.json",
) -> dict[str, object]:
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("doc_id must be a non-empty string")

    record = _load_record(doc_id.strip(), Path(records_path))
    metadata = dict(record.get("metadata", {}))
    text = str(record.get("text", "")).strip()

    title = str(metadata.get("title") or metadata.get("document_id") or doc_id).strip()
    summary = str(metadata.get("summary") or _build_summary(text) or title).strip()
    tags = _normalize_tags(metadata.get("tags"))

    payload = {
        "doc_id": doc_id,
        "title": title,
        "summary": summary,
        "tags": tags,
        "source": str(metadata.get("source_path") or metadata.get("source") or "unknown"),
    }
    return {
        "content": [
            {
                "type": "text",
                "text": _build_markdown(payload),
            }
        ],
        "structuredContent": payload,
    }


def _load_record(doc_id: str, records_path: Path) -> dict[str, object]:
    if not records_path.exists():
        raise ValueError(f"document store not found: {records_path}")

    payload = json.loads(records_path.read_text(encoding="utf-8"))
    for item in payload:
        if str(item.get("id")) == doc_id:
            return item
    raise ValueError(f"doc_id not found: {doc_id}")


def _normalize_tags(raw_tags: object) -> list[str]:
    if isinstance(raw_tags, list):
        return [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    return []


def _build_summary(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def _build_markdown(payload: dict[str, object]) -> str:
    tags = ", ".join(str(tag) for tag in payload["tags"]) or "-"
    return (
        f"Title: {payload['title']}\n\n"
        f"Summary: {payload['summary']}\n\n"
        f"Tags: {tags}\n"
        f"Source: {payload['source']}"
    )
