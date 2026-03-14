"""Factory for creating splitter strategies from settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from core.settings import Settings, SplitterSettings
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.fixed_length_splitter import FixedLengthSplitter
from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.semantic_splitter import SemanticSplitter


class UnsupportedSplitterError(ValueError):
    """Raised when the configured splitter method has no implementation."""


@dataclass(slots=True)
class InlineSplitterSettings:
    """Fallback settings shape for tests or local construction."""

    method: str
    chunk_size: int
    chunk_overlap: int = 0


class SplitterFactory:
    """Resolve splitter methods to concrete implementations."""

    _registry: dict[str, Type[BaseSplitter]] = {
        "fixed": FixedLengthSplitter,
        "recursive": RecursiveSplitter,
        "semantic": SemanticSplitter,
    }

    @classmethod
    def create(cls, settings: Settings | SplitterSettings | InlineSplitterSettings) -> BaseSplitter:
        splitter_settings = settings.splitter if isinstance(settings, Settings) else settings
        method = splitter_settings.method.strip().lower()
        splitter_cls = cls._registry.get(method)
        if splitter_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise UnsupportedSplitterError(
                f"Unsupported splitter method: {splitter_settings.method}. "
                f"Supported methods: {supported}"
            )
        return splitter_cls(
            chunk_size=splitter_settings.chunk_size,
            chunk_overlap=splitter_settings.chunk_overlap,
        )

    @classmethod
    def register(cls, method: str, splitter_cls: Type[BaseSplitter]) -> None:
        """Allow tests or extensions to register additional splitter strategies."""

        cls._registry[method.strip().lower()] = splitter_cls
