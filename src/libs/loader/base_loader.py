"""Base abstractions for pluggable document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.types import Document


class BaseLoader(ABC):
    """Common interface all source loaders must implement."""

    @abstractmethod
    def load(self, path: str | Path) -> Document:
        """Load a source file into the normalized Document contract."""
