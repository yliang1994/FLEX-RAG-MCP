"""Trace collection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.trace.trace_context import TraceContext


@dataclass(slots=True)
class TraceCollector:
    """Collect finished trace payloads for downstream persistence."""

    traces: list[dict[str, Any]] = field(default_factory=list)

    def collect(self, trace: TraceContext) -> None:
        trace.finish()
        self.traces.append(trace.to_dict())
