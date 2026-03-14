"""Entry point for the Modular RAG MCP server."""

from __future__ import annotations

from core.settings import load_settings
from observability.logger import get_logger


def main() -> int:
    settings = load_settings("config/settings.yaml")
    logger = get_logger(__name__)
    logger.info(
        "Loaded settings for llm=%s embedding=%s",
        settings.llm.provider,
        settings.embedding.provider,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
