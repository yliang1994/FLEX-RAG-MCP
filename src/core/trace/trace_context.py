"""Trace context primitives shared by ingestion and retrieval flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class TraceStage:
    """Structured record for one pipeline stage."""

    name: str
    recorded_at: str
    elapsed_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "recorded_at": self.recorded_at,
            "elapsed_ms": self.elapsed_ms,
            "details": self.details,
        }


class TraceContext:
    """Collect stage-level execution details for a single request."""

    _VALID_TRACE_TYPES = {"query", "ingestion"}

    def __init__(
        self,
        trace_type: str = "query",
        *,
        now_factory: Callable[[], datetime] = _utc_now,
        timer: Callable[[], float] = perf_counter,
    ) -> None:
        if trace_type not in self._VALID_TRACE_TYPES:
            raise ValueError(f"unsupported trace_type: {trace_type}")

        self.trace_type = trace_type
        self._now_factory = now_factory
        self._timer = timer
        self._started_at_dt = now_factory()
        self._started_at_timer = timer()
        self.trace_id = str(uuid4())
        self.started_at = self._serialize_datetime(self._started_at_dt)
        self.finished_at: str | None = None
        self.total_elapsed_ms: float | None = None
        self.stages: list[TraceStage] = []

    def record_stage(self, name: str, **details: Any) -> None:
        """Append a stage event to the in-memory trace."""

        elapsed_ms = details.get("elapsed_ms")
        normalized_elapsed = float(elapsed_ms) if isinstance(elapsed_ms, (int, float)) else None
        self.stages.append(
            TraceStage(
                name=name,
                recorded_at=self._serialize_datetime(self._now_factory()),
                elapsed_ms=normalized_elapsed,
                details=details,
            )
        )

    def finish(self) -> None:
        """Mark the trace as finished and freeze total elapsed time."""

        if self.finished_at is not None:
            return

        finished_dt = self._now_factory()
        self.finished_at = self._serialize_datetime(finished_dt)
        self.total_elapsed_ms = round((self._timer() - self._started_at_timer) * 1000, 3)

    def elapsed_ms(self, stage_name: str | None = None) -> float:
        """Return total elapsed time or the summed elapsed time of one stage."""

        if stage_name is None:
            if self.total_elapsed_ms is not None:
                return self.total_elapsed_ms
            return round((self._timer() - self._started_at_timer) * 1000, 3)

        total = 0.0
        for stage in self.stages:
            if stage.name == stage_name and stage.elapsed_ms is not None:
                total += stage.elapsed_ms
        return round(total, 3)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace into a JSON-friendly dictionary."""

        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": self.elapsed_ms(),
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @staticmethod
    def _serialize_datetime(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
