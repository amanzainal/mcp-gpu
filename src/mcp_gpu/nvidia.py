"""Core GPU telemetry layer.

This module is deliberately split into pure parsing functions (string in,
dataclass out) and thin subprocess wrappers around ``nvidia-smi``.  Keeping the
parsing pure makes it trivial to unit-test against fixture CSV and to drive a
mock mode on machines with no GPU.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import List

# ---------------------------------------------------------------------------
# Query field definitions
# ---------------------------------------------------------------------------
# The exact fields passed to ``nvidia-smi --query-gpu=...``.  Order matters:
# it must match the column order assumed by ``parse_gpu_csv``.
GPU_QUERY_FIELDS = [
    "index",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "temperature.gpu",
]

# Fields passed to ``nvidia-smi --query-compute-apps=...``.
PROC_QUERY_FIELDS = [
    "gpu_uuid",
    "pid",
    "process_name",
    "used_memory",
]

# Common flags: machine-readable CSV with no header and no unit suffixes.
CSV_FORMAT = "csv,noheader,nounits"


class NvidiaSmiNotFound(RuntimeError):
    """Raised when ``nvidia-smi`` is not on PATH and mock mode is off."""


class NvidiaSmiError(RuntimeError):
    """Raised when ``nvidia-smi`` runs but exits non-zero."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_pct: int
    temperature_c: int

    @property
    def memory_free_mb(self) -> int:
        return max(self.memory_total_mb - self.memory_used_mb, 0)

    @property
    def memory_used_pct(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return round(100.0 * self.memory_used_mb / self.memory_total_mb, 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["memory_free_mb"] = self.memory_free_mb
        d["memory_used_pct"] = self.memory_used_pct
        return d


@dataclass(frozen=True)
class GpuProcess:
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mb: int

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pure parsers (no subprocess, fully unit-testable)
# ---------------------------------------------------------------------------
def _clean(cell: str) -> str:
    return cell.strip()


def _to_int(cell: str) -> int:
    """Parse an integer cell, tolerating ``[N/A]`` / ``[Not Supported]``."""
    text = _clean(cell)
    if not text or text.startswith("["):
        return 0
    # values may arrive as floats like "75.0"
    return int(float(text))


def parse_gpu_csv(csv_text: str) -> List[GpuInfo]:
    """Parse ``--query-gpu`` CSV (noheader,nounits) into GpuInfo rows."""
    gpus: List[GpuInfo] = []
    for line in csv_text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(GPU_QUERY_FIELDS):
            # Malformed / partial line; skip rather than crash.
            continue
        gpus.append(
            GpuInfo(
                index=_to_int(parts[0]),
                name=_clean(parts[1]),
                memory_total_mb=_to_int(parts[2]),
                memory_used_mb=_to_int(parts[3]),
                utilization_pct=_to_int(parts[4]),
                temperature_c=_to_int(parts[5]),
            )
        )
    return gpus


def parse_proc_csv(csv_text: str) -> List[GpuProcess]:
    """Parse ``--query-compute-apps`` CSV (noheader,nounits) into rows.

    ``nvidia-smi`` prints a single line such as ``No running processes found``
    (without commas) when nothing is using the GPU; such lines are ignored.
    """
    procs: List[GpuProcess] = []
    for line in csv_text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(PROC_QUERY_FIELDS):
            continue
        procs.append(
            GpuProcess(
                gpu_uuid=_clean(parts[0]),
                pid=_to_int(parts[1]),
                process_name=_clean(parts[2]),
                used_memory_mb=_to_int(parts[3]),
            )
        )
    return procs


# ---------------------------------------------------------------------------
# Mock data (synthetic, obviously fake — runs on GPU-less machines / CI)
# ---------------------------------------------------------------------------
def mock_gpu_csv() -> str:
    """Return synthetic ``--query-gpu`` CSV for two fake GPUs."""
    return (
        "0, Synthetic GPU Model A, 24576, 8192, 42, 55\n"
        "1, Synthetic GPU Model A, 24576, 1024, 3, 38\n"
    )


def mock_proc_csv() -> str:
    """Return synthetic ``--query-compute-apps`` CSV for two fake processes."""
    return (
        "GPU-00000000-mock-0000-0000-000000000000, 12345, example-trainer, 8192\n"
        "GPU-11111111-mock-1111-1111-111111111111, 23456, example-inference, 1024\n"
    )


def mock_enabled(flag: bool = False) -> bool:
    """True if mock mode is requested via the ``--mock`` flag or env var."""
    if flag:
        return True
    return os.environ.get("MCP_GPU_MOCK", "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Subprocess layer
# ---------------------------------------------------------------------------
def have_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def _run(args: List[str]) -> str:
    """Run ``nvidia-smi`` with ``args`` and return stdout, raising on failure."""
    if not have_nvidia_smi():
        raise NvidiaSmiNotFound(
            "nvidia-smi was not found on PATH. Install the NVIDIA driver, or "
            "run with --mock (or MCP_GPU_MOCK=1) to emit synthetic data."
        )
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - exercised only on broken installs
        raise NvidiaSmiError(f"Failed to execute nvidia-smi: {exc}") from exc
    if proc.returncode != 0:
        raise NvidiaSmiError(
            f"nvidia-smi exited {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def query_gpus(*, mock: bool = False) -> List[GpuInfo]:
    if mock_enabled(mock):
        return parse_gpu_csv(mock_gpu_csv())
    csv_text = _run(
        [f"--query-gpu={','.join(GPU_QUERY_FIELDS)}", f"--format={CSV_FORMAT}"]
    )
    return parse_gpu_csv(csv_text)


def query_processes(*, mock: bool = False) -> List[GpuProcess]:
    if mock_enabled(mock):
        return parse_proc_csv(mock_proc_csv())
    csv_text = _run(
        [f"--query-compute-apps={','.join(PROC_QUERY_FIELDS)}", f"--format={CSV_FORMAT}"]
    )
    return parse_proc_csv(csv_text)


# ---------------------------------------------------------------------------
# Power limit control
# ---------------------------------------------------------------------------
def power_limit_command(index: int, watts: int) -> List[str]:
    """Build the argv for setting a GPU power limit (does not execute it).

    Exposed separately so tests can assert the exact command without root.
    """
    return ["nvidia-smi", "-i", str(int(index)), "-pl", str(int(watts))]


def set_power_limit(index: int, watts: int, *, mock: bool = False) -> str:
    """Set a GPU power limit. Requires root/sudo on real hardware.

    Returns a human-readable status string.  In mock mode no command is run.
    """
    index = int(index)
    watts = int(watts)
    if watts <= 0:
        raise ValueError("watts must be a positive integer")

    cmd = power_limit_command(index, watts)
    if mock_enabled(mock):
        return f"[mock] would run: {' '.join(cmd)}"

    if not have_nvidia_smi():
        raise NvidiaSmiNotFound(
            "nvidia-smi was not found on PATH. Cannot set power limit."
        )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        hint = ""
        if "permission" in stderr.lower() or "insufficient" in stderr.lower():
            hint = " (this usually requires root; try running the server under sudo)"
        raise NvidiaSmiError(
            f"Failed to set power limit on GPU {index}: {stderr or proc.stdout.strip()}{hint}"
        )
    return proc.stdout.strip() or f"Set power limit of GPU {index} to {watts} W."
