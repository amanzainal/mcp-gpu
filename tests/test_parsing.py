"""Tests for the pure nvidia-smi CSV parsers."""

from mcp_gpu import nvidia

from .fixtures import (
    GPU_CSV,
    GPU_CSV_WITH_NA,
    PROC_CSV,
    PROC_CSV_EMPTY,
)


def test_parse_gpu_csv_basic():
    gpus = nvidia.parse_gpu_csv(GPU_CSV)
    assert len(gpus) == 2

    g0 = gpus[0]
    assert g0.index == 0
    assert g0.name == "Example Synthetic GPU"
    assert g0.memory_total_mb == 24576
    assert g0.memory_used_mb == 12000
    assert g0.utilization_pct == 87
    assert g0.temperature_c == 71

    # Derived fields.
    assert g0.memory_free_mb == 24576 - 12000
    assert g0.memory_used_pct == round(100 * 12000 / 24576, 1)

    g1 = gpus[1]
    assert g1.index == 1
    assert g1.utilization_pct == 0
    assert g1.memory_free_mb == 24576 - 256


def test_parse_gpu_csv_handles_na():
    gpus = nvidia.parse_gpu_csv(GPU_CSV_WITH_NA)
    assert len(gpus) == 1
    # [N/A] utilization should coerce to 0 rather than crash.
    assert gpus[0].utilization_pct == 0
    assert gpus[0].temperature_c == 60


def test_parse_gpu_csv_skips_blank_and_malformed_lines():
    text = "\n0, Example Synthetic GPU, 8192, 1024, 10, 40\nbad,line\n\n"
    gpus = nvidia.parse_gpu_csv(text)
    assert len(gpus) == 1
    assert gpus[0].memory_total_mb == 8192


def test_to_dict_includes_derived_fields():
    g = nvidia.parse_gpu_csv(GPU_CSV)[0]
    d = g.to_dict()
    assert d["memory_free_mb"] == 24576 - 12000
    assert "memory_used_pct" in d
    assert d["index"] == 0


def test_parse_proc_csv_basic():
    procs = nvidia.parse_proc_csv(PROC_CSV)
    assert len(procs) == 2
    assert procs[0].pid == 4242
    assert procs[0].process_name == "example-trainer"
    assert procs[0].used_memory_mb == 12000
    assert procs[1].gpu_uuid.startswith("GPU-bbbbbbbb")


def test_parse_proc_csv_empty_message_yields_no_rows():
    # "No running processes found" has no commas -> not enough columns -> skipped.
    procs = nvidia.parse_proc_csv(PROC_CSV_EMPTY)
    assert procs == []
