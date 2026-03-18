"""MCP tool for listing local document collections."""

from __future__ import annotations

from pathlib import Path


def list_collections(root_dir: str | Path = "data/documents") -> dict[str, object]:
    base_dir = Path(root_dir)
    collections = _discover_collections(base_dir)

    if not collections:
        message = "当前没有可用集合，请先在 data/documents/ 下准备集合目录并完成数据摄取。"
        return {
            "content": [
                {
                    "type": "text",
                    "text": message,
                }
            ],
            "structuredContent": {
                "collections": [],
                "count": 0,
            },
        }

    lines = ["可用集合：", ""]
    for item in collections:
        lines.append(f"- {item['name']} ({item['document_count']} docs)")

    return {
        "content": [
            {
                "type": "text",
                "text": "\n".join(lines),
            }
        ],
        "structuredContent": {
            "collections": collections,
            "count": len(collections),
        },
    }


def _discover_collections(base_dir: Path) -> list[dict[str, object]]:
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    collections: list[dict[str, object]] = []
    for collection_dir in sorted(
        (path for path in base_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        document_count = sum(1 for path in collection_dir.rglob("*") if path.is_file())
        collections.append(
            {
                "name": collection_dir.name,
                "path": str(collection_dir),
                "document_count": document_count,
            }
        )
    return collections
