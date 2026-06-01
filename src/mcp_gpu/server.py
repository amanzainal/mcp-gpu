"""MCP server wiring GPU telemetry tools onto the official MCP SDK.

A module-level flag (``set_mock``) lets the CLI force mock mode for every tool
without threading the flag through each MCP call.
"""

from __future__ import annotations

from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from . import nvidia
from .summary import format_summary

mcp = FastMCP("mcp-gpu")

# When True, every tool serves synthetic data regardless of the environment.
_FORCE_MOCK = False


def set_mock(enabled: bool) -> None:
    """Force (or unforce) mock mode for all tools. Called by the CLI."""
    global _FORCE_MOCK
    _FORCE_MOCK = bool(enabled)


@mcp.tool()
def list_gpus() -> List[Dict[str, Any]]:
    """List all local NVIDIA GPUs with memory, utilization and temperature.

    Returns one object per GPU with index, name, total/used/free memory (MiB),
    memory-used percent, GPU utilization percent and temperature (Celsius).
    """
    gpus = nvidia.query_gpus(mock=_FORCE_MOCK)
    return [g.to_dict() for g in gpus]


@mcp.tool()
def gpu_processes() -> List[Dict[str, Any]]:
    """List compute processes currently running on the local NVIDIA GPUs.

    Returns one object per process with the owning GPU UUID, PID, process name
    and the GPU memory it is using (MiB).
    """
    procs = nvidia.query_processes(mock=_FORCE_MOCK)
    return [p.to_dict() for p in procs]


@mcp.tool()
def gpu_summary() -> str:
    """Return one compact, human-readable summary string for all GPUs."""
    gpus = nvidia.query_gpus(mock=_FORCE_MOCK)
    return format_summary(gpus)


@mcp.tool()
def set_power_limit(index: int, watts: int) -> str:
    """Set the power limit (in watts) for the GPU at ``index``.

    NOTE: on real hardware this shells out to ``nvidia-smi -i <index> -pl
    <watts>`` and typically requires root/sudo. In mock mode it only reports
    the command that would have run.
    """
    return nvidia.set_power_limit(index, watts, mock=_FORCE_MOCK)


def run() -> None:
    """Run the server over stdio transport."""
    mcp.run(transport="stdio")
