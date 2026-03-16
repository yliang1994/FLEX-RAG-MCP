#!/usr/bin/env python3
"""CLI entrypoint for hybrid retrieval queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.reranker import Reranker
from core.settings import Settings, load_settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the local knowledge hub.")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--top-k", type=int, default=None, help="Number of results to return")
    parser.add_argument("--collection", help="Optional collection metadata filter")
    parser.add_argument("--verbose", action="store_true", help="Print trace stages and details")
    parser.add_argument("--no-rerank", action="store_true", help="Skip the rerank stage")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "settings.yaml"),
        help="Path to settings.yaml",
    )
    return parser


def _resolve_top_k(settings: Settings, requested_top_k: int | None) -> int:
    if requested_top_k is not None:
        if requested_top_k <= 0:
            raise ValueError("--top-k must be positive")
        return requested_top_k
    return settings.retrieval.top_k_final


def _format_result(index: int, result: RetrievalResult) -> str:
    source = str(result.metadata.get("source_path") or result.metadata.get("source") or "-")
    page = result.metadata.get("page", "-")
    snippet = " ".join(result.text.strip().split())
    if len(snippet) > 140:
        snippet = f"{snippet[:137]}..."
    return (
        f"{index}. score={result.score:.4f} source={source} page={page}\n"
        f"   chunk_id={result.chunk_id}\n"
        f"   text={snippet}"
    )


def _print_verbose(trace: TraceContext) -> None:
    print("\nTrace:")
    for stage in trace.stages:
        print(f"- {stage.name}: {json.dumps(stage.details, ensure_ascii=True, sort_keys=True)}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.config)
        top_k = _resolve_top_k(settings, args.top_k)
        trace = TraceContext()
        filters = {"collection": args.collection} if args.collection else None

        hybrid_search = HybridSearch(settings)
        candidates = hybrid_search.search(args.query, top_k=top_k, filters=filters, trace=trace)

        if args.no_rerank:
            results = candidates
        else:
            results = Reranker(settings).rerank(args.query, candidates, trace=trace)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 1

    if not results:
        print("未找到相关文档，请先运行 ingest.py 摄取数据，或检查查询条件。")
        if args.verbose:
            _print_verbose(trace)
        return 0

    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    for index, result in enumerate(results, start=1):
        print(_format_result(index, result))

    if args.verbose:
        _print_verbose(trace)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
