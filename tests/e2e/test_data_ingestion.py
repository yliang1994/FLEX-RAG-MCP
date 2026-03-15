from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: stub",
                "  model: stub-llm",
                "  api_key: ''",
                "embedding:",
                "  provider: stub",
                "  model: stub-embedding",
                "splitter:",
                "  method: recursive",
                "  chunk_size: 500",
                "  chunk_overlap: 50",
                "vector_store:",
                "  backend: chroma",
                "  persist_path: ./data/db/chroma",
                "retrieval:",
                "  sparse_backend: bm25",
                "  fusion_algorithm: rrf",
                "  top_k_dense: 20",
                "  top_k_sparse: 20",
                "  top_k_final: 10",
                "rerank:",
                "  backend: none",
                "  model: ''",
                "  top_m: 30",
                "evaluation:",
                "  backends: [custom]",
                "  golden_test_set: ./tests/fixtures/golden_test_set.json",
                "observability:",
                "  enabled: true",
                "  log_file: ./logs/traces.jsonl",
                "ingestion:",
                "  chunk_refiner:",
                "    use_llm: false",
                "  metadata_enricher:",
                "    use_llm: false",
                "  image_captioner:",
                "    enabled: false",
            ]
        ),
        encoding="utf-8",
    )


def test_ingest_cli_runs_and_skips_on_second_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "ingest.py"
    config_path = tmp_path / "settings.yaml"
    document_path = tmp_path / "sample.pdf"
    image_path = tmp_path / "diagram.png"

    write_config(config_path)
    image_path.write_bytes(b"fake-image")
    document_path.write_text(
        "CLI Title\n\nCLI body paragraph.\n\n[[IMAGE:diagram.png]]",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--path",
        str(document_path),
        "--collection",
        "manuals",
    ]

    first = subprocess.run(command, cwd=tmp_path, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    first_payload = json.loads(first.stdout)
    assert first_payload["status"] == "ingested"
    assert (tmp_path / "data" / "db" / "chroma" / "records.json").exists()
    assert (tmp_path / "data" / "db" / "bm25" / "index.json").exists()

    second = subprocess.run(command, cwd=tmp_path, check=False, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    second_payload = json.loads(second.stdout)
    assert second_payload["status"] == "skipped"
