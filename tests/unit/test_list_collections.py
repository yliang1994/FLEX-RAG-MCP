from __future__ import annotations

from pathlib import Path

from mcp_server.tools.list_collections import list_collections


def test_list_collections_returns_sorted_collection_names(tmp_path: Path) -> None:
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    (alpha / "a.pdf").write_text("a", encoding="utf-8")
    (alpha / "b.md").write_text("b", encoding="utf-8")
    (beta / "notes.txt").write_text("c", encoding="utf-8")

    response = list_collections(tmp_path)

    assert response["content"][0]["type"] == "text"
    assert response["structuredContent"] == {
        "collections": [
            {
                "name": "alpha",
                "path": str(alpha),
                "document_count": 2,
            },
            {
                "name": "beta",
                "path": str(beta),
                "document_count": 1,
            },
        ],
        "count": 2,
    }


def test_list_collections_returns_friendly_message_for_missing_root(tmp_path: Path) -> None:
    response = list_collections(tmp_path / "missing")

    assert response == {
        "content": [
            {
                "type": "text",
                "text": "当前没有可用集合，请先在 data/documents/ 下准备集合目录并完成数据摄取。",
            }
        ],
        "structuredContent": {
            "collections": [],
            "count": 0,
        },
    }
