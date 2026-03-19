#!/usr/bin/env python3
"""CLI entrypoint for offline data ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from core.settings import load_settings
from ingestion.pipeline import IngestionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ingestion pipeline for one document path.")
    parser.add_argument("--path", required=True, help="Path to the source document")
    parser.add_argument("--collection", default="default", help="Target collection name")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if the file hash already exists")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "settings.yaml"),
        help="Path to settings.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.config)
        pipeline = IngestionPipeline(settings)
        result = pipeline.ingest(path=args.path, collection=args.collection, force=args.force)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 1

    payload = asdict(result)
    if result.trace is not None:
        payload["trace"] = result.trace.to_dict()
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
