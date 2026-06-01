"""Human-readable formatting for GPU telemetry."""

from __future__ import annotations

from typing import List

from .nvidia import GpuInfo


def format_gpu_line(gpu: GpuInfo) -> str:
    """One compact line per GPU, e.g.::

    GPU 0 (Synthetic GPU Model A): 42% util, 8192/24576 MiB (33.3%), 55C
    """
    return (
        f"GPU {gpu.index} ({gpu.name}): "
        f"{gpu.utilization_pct}% util, "
        f"{gpu.memory_used_mb}/{gpu.memory_total_mb} MiB "
        f"({gpu.memory_used_pct}%), "
        f"{gpu.temperature_c}C"
    )


def format_summary(gpus: List[GpuInfo]) -> str:
    """Return one compact multi-line summary string for all GPUs."""
    if not gpus:
        return "No NVIDIA GPUs detected."
    header = f"{len(gpus)} GPU(s) detected:"
    lines = [format_gpu_line(g) for g in gpus]
    return "\n".join([header, *lines])
