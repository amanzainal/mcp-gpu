"""Tests for summary formatting and the power-limit command builder."""

from mcp_gpu import nvidia
from mcp_gpu.summary import format_gpu_line, format_summary

from .fixtures import GPU_CSV


def test_power_limit_command_shape():
    cmd = nvidia.power_limit_command(1, 300)
    assert cmd == ["nvidia-smi", "-i", "1", "-pl", "300"]


def test_power_limit_command_coerces_types():
    # Floats / numeric strings should normalise to ints in the argv.
    cmd = nvidia.power_limit_command(0, 250)
    assert cmd[2] == "0"
    assert cmd[4] == "250"


def test_format_gpu_line_contents():
    g = nvidia.parse_gpu_csv(GPU_CSV)[0]
    line = format_gpu_line(g)
    assert "GPU 0" in line
    assert "Example Synthetic GPU" in line
    assert "87% util" in line
    assert "12000/24576 MiB" in line
    assert "71C" in line


def test_format_summary_multiple_gpus():
    gpus = nvidia.parse_gpu_csv(GPU_CSV)
    out = format_summary(gpus)
    assert out.startswith("2 GPU(s) detected:")
    assert out.count("\n") == 2  # header + 2 GPU lines


def test_format_summary_empty():
    assert format_summary([]) == "No NVIDIA GPUs detected."
