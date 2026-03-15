"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SettingsError(ValueError):
    """Raised when settings cannot be loaded or validated."""


@dataclass(slots=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str = ""


@dataclass(slots=True)
class EmbeddingSettings:
    provider: str
    model: str


@dataclass(slots=True)
class SplitterSettings:
    method: str
    chunk_size: int
    chunk_overlap: int


@dataclass(slots=True)
class VectorStoreSettings:
    backend: str
    persist_path: str


@dataclass(slots=True)
class RetrievalSettings:
    sparse_backend: str
    fusion_algorithm: str
    top_k_dense: int
    top_k_sparse: int
    top_k_final: int


@dataclass(slots=True)
class RerankSettings:
    backend: str
    model: str
    top_m: int


@dataclass(slots=True)
class EvaluationSettings:
    backends: list[str]
    golden_test_set: str


@dataclass(slots=True)
class ObservabilitySettings:
    enabled: bool
    log_file: str


@dataclass(slots=True)
class ChunkRefinerSettings:
    use_llm: bool = False


@dataclass(slots=True)
class MetadataEnricherSettings:
    use_llm: bool = False


@dataclass(slots=True)
class IngestionSettings:
    chunk_refiner: ChunkRefinerSettings = field(default_factory=ChunkRefinerSettings)
    metadata_enricher: MetadataEnricherSettings = field(default_factory=MetadataEnricherSettings)


@dataclass(slots=True)
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    splitter: SplitterSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.lstrip("-").isdigit():
        return int(value)
    return value


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2 != 0:
            raise SettingsError(f"Invalid indentation at line {lineno}")

        line = raw_line.strip()
        if ":" not in line:
            raise SettingsError(f"Expected key/value at line {lineno}")

        key, raw_value = line.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise SettingsError(f"Unexpected indentation at line {lineno}")

        current = stack[-1][1]
        value = raw_value.strip()
        if value == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value)

    return root


def validate_settings(settings: Settings) -> None:
    required: list[tuple[str, Any]] = [
        ("llm.provider", settings.llm.provider),
        ("llm.model", settings.llm.model),
        ("embedding.provider", settings.embedding.provider),
        ("embedding.model", settings.embedding.model),
        ("splitter.method", settings.splitter.method),
        ("splitter.chunk_size", settings.splitter.chunk_size),
        ("splitter.chunk_overlap", settings.splitter.chunk_overlap),
        ("vector_store.backend", settings.vector_store.backend),
        ("vector_store.persist_path", settings.vector_store.persist_path),
        ("retrieval.sparse_backend", settings.retrieval.sparse_backend),
        ("retrieval.fusion_algorithm", settings.retrieval.fusion_algorithm),
        ("evaluation.backends", settings.evaluation.backends),
        ("evaluation.golden_test_set", settings.evaluation.golden_test_set),
        ("observability.log_file", settings.observability.log_file),
    ]
    for field_path, value in required:
        if value in ("", None, []):
            raise SettingsError(f"Missing required field: {field_path}")


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path)
    if not config_path.exists():
        raise SettingsError(f"Settings file not found: {config_path}")

    raw = _parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    try:
        settings = Settings(
            llm=LLMSettings(**raw["llm"]),
            embedding=EmbeddingSettings(**raw["embedding"]),
            splitter=SplitterSettings(**raw["splitter"]),
            vector_store=VectorStoreSettings(**raw["vector_store"]),
            retrieval=RetrievalSettings(**raw["retrieval"]),
            rerank=RerankSettings(**raw["rerank"]),
            evaluation=EvaluationSettings(**raw["evaluation"]),
            observability=ObservabilitySettings(**raw["observability"]),
            ingestion=IngestionSettings(
                chunk_refiner=ChunkRefinerSettings(
                    **raw.get("ingestion", {}).get("chunk_refiner", {})
                ),
                metadata_enricher=MetadataEnricherSettings(
                    **raw.get("ingestion", {}).get("metadata_enricher", {})
                ),
            ),
        )
    except KeyError as exc:
        raise SettingsError(f"Missing required section: {exc.args[0]}") from exc
    except TypeError as exc:
        raise SettingsError(f"Invalid settings shape: {exc}") from exc

    validate_settings(settings)
    return settings
