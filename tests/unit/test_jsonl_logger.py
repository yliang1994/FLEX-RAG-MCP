from __future__ import annotations

import json
import logging
from pathlib import Path

from observability.logger import JSONFormatter, get_trace_logger, write_trace


def test_json_formatter_serializes_dict_payload() -> None:
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="trace.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg={"trace_id": "trace-1", "trace_type": "query"},
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert payload["trace_id"] == "trace-1"
    assert payload["trace_type"] == "query"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "trace.test"


def test_write_trace_appends_jsonl_record(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "traces.jsonl"

    write_trace(
        {
            "trace_id": "trace-2",
            "trace_type": "ingestion",
            "stages": [],
        },
        log_file,
    )

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["trace_id"] == "trace-2"
    assert payload["trace_type"] == "ingestion"


def test_get_trace_logger_reuses_same_file_handler(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "traces.jsonl"

    logger_a = get_trace_logger(log_file)
    logger_b = get_trace_logger(log_file)

    assert logger_a is logger_b
    assert len(logger_a.handlers) == 1
