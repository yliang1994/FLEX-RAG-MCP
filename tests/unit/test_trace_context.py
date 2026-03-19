from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from core.trace.trace_collector import TraceCollector
from core.trace.trace_context import TraceContext


class FakeClock:
    def __init__(self, timestamps: list[datetime], timers: list[float]) -> None:
        self._timestamps = iter(timestamps)
        self._timers = iter(timers)

    def now(self) -> datetime:
        return next(self._timestamps)

    def timer(self) -> float:
        return next(self._timers)


def test_trace_context_finish_and_to_dict_are_json_serializable() -> None:
    clock = FakeClock(
        timestamps=[
            datetime(2026, 3, 19, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 19, 1, 0, 1, tzinfo=timezone.utc),
            datetime(2026, 3, 19, 1, 0, 2, tzinfo=timezone.utc),
        ],
        timers=[10.0, 11.25],
    )
    trace = TraceContext(trace_type="ingestion", now_factory=clock.now, timer=clock.timer)

    trace.record_stage("load", elapsed_ms=125.5, chunk_count=3)
    trace.finish()

    payload = trace.to_dict()

    assert payload["trace_type"] == "ingestion"
    assert payload["started_at"] == "2026-03-19T01:00:00Z"
    assert payload["finished_at"] == "2026-03-19T01:00:02Z"
    assert payload["total_elapsed_ms"] == 1250.0
    assert payload["stages"] == [
        {
            "name": "load",
            "recorded_at": "2026-03-19T01:00:01Z",
            "elapsed_ms": 125.5,
            "details": {
                "elapsed_ms": 125.5,
                "chunk_count": 3,
            },
        }
    ]
    json.dumps(payload)


def test_elapsed_ms_returns_stage_and_total_elapsed() -> None:
    clock = FakeClock(
        timestamps=[
            datetime(2026, 3, 19, 2, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 19, 2, 0, 1, tzinfo=timezone.utc),
            datetime(2026, 3, 19, 2, 0, 2, tzinfo=timezone.utc),
            datetime(2026, 3, 19, 2, 0, 3, tzinfo=timezone.utc),
        ],
        timers=[20.0, 21.0],
    )
    trace = TraceContext(now_factory=clock.now, timer=clock.timer)

    trace.record_stage("dense", elapsed_ms=40.0)
    trace.record_stage("dense", elapsed_ms=60.0)
    trace.finish()

    assert trace.elapsed_ms("dense") == 100.0
    assert trace.elapsed_ms("missing") == 0.0
    assert trace.elapsed_ms() == 1000.0


def test_trace_collector_collects_finished_payload() -> None:
    clock = FakeClock(
        timestamps=[
            datetime(2026, 3, 19, 3, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 19, 3, 0, 1, tzinfo=timezone.utc),
        ],
        timers=[30.0, 30.5],
    )
    trace = TraceContext(now_factory=clock.now, timer=clock.timer)
    collector = TraceCollector()

    collector.collect(trace)

    assert len(collector.traces) == 1
    assert collector.traces[0]["finished_at"] == "2026-03-19T03:00:01Z"
    assert collector.traces[0]["total_elapsed_ms"] == 500.0


def test_trace_context_rejects_unknown_trace_type() -> None:
    with pytest.raises(ValueError, match="unsupported trace_type"):
        TraceContext(trace_type="other")
