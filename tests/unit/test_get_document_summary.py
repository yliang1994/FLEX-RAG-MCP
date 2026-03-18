from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_server.tools.get_document_summary import get_document_summary


def test_get_document_summary_returns_structured_payload(tmp_path: Path) -> None:
    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps(
            [
                {
                    "id": "doc-1",
                    "text": "Hybrid retrieval combines dense and sparse search for better recall.",
                    "vector": [0.1, 0.2, 0.3],
                    "metadata": {
                        "title": "Hybrid Search Overview",
                        "summary": "A short explanation of hybrid search.",
                        "tags": ["retrieval", "hybrid"],
                        "source_path": "docs/hybrid.pdf",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    response = get_document_summary("doc-1", records_path=records_path)

    assert response["structuredContent"] == {
        "doc_id": "doc-1",
        "title": "Hybrid Search Overview",
        "summary": "A short explanation of hybrid search.",
        "tags": ["retrieval", "hybrid"],
        "source": "docs/hybrid.pdf",
    }
    assert "Title: Hybrid Search Overview" in response["content"][0]["text"]


def test_get_document_summary_falls_back_to_text_when_metadata_missing(tmp_path: Path) -> None:
    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps(
            [
                {
                    "id": "doc-2",
                    "text": "This record has no explicit summary metadata but should still produce a readable summary.",
                    "vector": [0.1, 0.2, 0.3],
                    "metadata": {},
                }
            ]
        ),
        encoding="utf-8",
    )

    response = get_document_summary("doc-2", records_path=records_path)

    assert response["structuredContent"]["doc_id"] == "doc-2"
    assert response["structuredContent"]["title"] == "doc-2"
    assert response["structuredContent"]["tags"] == []
    assert response["structuredContent"]["summary"].startswith("This record has no explicit summary metadata")


def test_get_document_summary_rejects_unknown_doc_id(tmp_path: Path) -> None:
    records_path = tmp_path / "records.json"
    records_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="doc_id not found: missing-doc"):
        get_document_summary("missing-doc", records_path=records_path)
