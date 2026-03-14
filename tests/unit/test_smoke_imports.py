from __future__ import annotations

import importlib


def test_core_packages_import() -> None:
    for module_name in [
        "mcp_server",
        "core",
        "ingestion",
        "libs",
        "observability",
    ]:
        assert importlib.import_module(module_name) is not None
