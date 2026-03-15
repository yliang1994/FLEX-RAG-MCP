"""Minimal trace context used by ingestion and retrieval components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class TraceStage:
    """Structured record for one pipeline stage."""

    name: str
    details: dict[str, Any] = field(default_factory=dict)


class TraceContext:
    """Collect stage-level execution details for a single request."""

    def __init__(self) -> None:
        self.trace_id = str(uuid4())
        self.stages: list[TraceStage] = []

    def record_stage(self, name: str, **details: Any) -> None:
        """Append a stage event to the in-memory trace."""

        self.stages.append(TraceStage(name=name, details=details))

