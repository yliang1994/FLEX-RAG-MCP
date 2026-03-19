"""Logging helpers for human-readable stderr logs and trace JSONL output."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Serialize log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = self._coerce_payload(record)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _coerce_payload(self, record: logging.LogRecord) -> dict[str, Any]:
        message = record.msg
        if isinstance(message, dict):
            payload = dict(message)
        else:
            payload = {"message": record.getMessage()}

        payload.setdefault("level", record.levelname)
        payload.setdefault("logger", record.name)
        return payload


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger(name)


def get_trace_logger(log_file: str | Path = "logs/traces.jsonl") -> logging.Logger:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger_name = f"observability.trace.{path.resolve()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    target_path = path.resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == target_path:
            return logger

    file_handler = logging.FileHandler(target_path, encoding="utf-8")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    return logger


def write_trace(trace_dict: dict[str, Any], log_file: str | Path = "logs/traces.jsonl") -> None:
    logger = get_trace_logger(log_file)
    logger.info(trace_dict)
