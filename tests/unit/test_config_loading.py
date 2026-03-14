from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import SettingsError, load_settings


def test_load_settings_reads_minimal_config(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: stub",
                "  model: stub-llm",
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
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.llm.provider == "stub"
    assert settings.splitter.method == "recursive"
    assert settings.retrieval.top_k_final == 10
    assert settings.observability.enabled is True


def test_load_settings_reports_missing_field(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: stub",
                "  model: stub-llm",
                "embedding:",
                "  provider: ''",
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
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="embedding.provider"):
        load_settings(config_path)
