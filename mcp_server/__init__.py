"""Root shim for the src-layout mcp_server package."""

from __future__ import annotations

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "src" / "mcp_server")]
