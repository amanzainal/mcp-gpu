"""Command-line entry point for the mcp-gpu server.

Usage::

    mcp-gpu              # serve real nvidia-smi data over stdio
    mcp-gpu --mock       # serve synthetic data (no GPU required)
    MCP_GPU_MOCK=1 mcp-gpu

The server speaks the Model Context Protocol over stdio, so it is normally
launched by an MCP client (Claude Desktop / Claude Code) rather than by hand.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-gpu",
        description="MCP server exposing local NVIDIA GPU stats and controls.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Serve synthetic GPU data (no nvidia-smi required). "
        "Equivalent to setting MCP_GPU_MOCK=1.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mcp-gpu {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Import lazily so --help / --version work even without the mcp SDK present.
    from . import server

    if args.mock:
        server.set_mock(True)

    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
