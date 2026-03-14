"""Factory for creating vector stores from settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from core.settings import Settings, VectorStoreSettings
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.chroma_store import ChromaStore


class UnsupportedVectorStoreError(ValueError):
    """Raised when the configured vector store backend has no implementation."""


@dataclass(slots=True)
class InlineVectorStoreSettings:
    """Fallback settings shape for tests or local construction."""

    backend: str
    persist_path: str


class VectorStoreFactory:
    """Resolve backend names to concrete vector store implementations."""

    _registry: dict[str, Type[BaseVectorStore]] = {
        "chroma": ChromaStore,
        "stub": ChromaStore,
    }

    @classmethod
    def create(
        cls, settings: Settings | VectorStoreSettings | InlineVectorStoreSettings
    ) -> BaseVectorStore:
        vector_settings = settings.vector_store if isinstance(settings, Settings) else settings
        backend = vector_settings.backend.strip().lower()
        store_cls = cls._registry.get(backend)
        if store_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise UnsupportedVectorStoreError(
                f"Unsupported vector store backend: {vector_settings.backend}. "
                f"Supported backends: {supported}"
            )
        return store_cls(persist_path=vector_settings.persist_path)

    @classmethod
    def register(cls, backend: str, store_cls: Type[BaseVectorStore]) -> None:
        """Allow tests or extensions to register additional vector stores."""

        cls._registry[backend.strip().lower()] = store_cls
