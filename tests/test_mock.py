"""Tests for the mock-mode path (runs with no GPU / nvidia-smi)."""

import pytest

from mcp_gpu import nvidia


def test_mock_enabled_flag():
    assert nvidia.mock_enabled(True) is True


def test_mock_enabled_env(monkeypatch):
    monkeypatch.setenv("MCP_GPU_MOCK", "1")
    assert nvidia.mock_enabled() is True
    monkeypatch.setenv("MCP_GPU_MOCK", "true")
    assert nvidia.mock_enabled() is True
    monkeypatch.setenv("MCP_GPU_MOCK", "0")
    assert nvidia.mock_enabled() is False
    monkeypatch.delenv("MCP_GPU_MOCK", raising=False)
    assert nvidia.mock_enabled() is False


def test_query_gpus_mock_returns_synthetic_rows(monkeypatch):
    monkeypatch.delenv("MCP_GPU_MOCK", raising=False)
    gpus = nvidia.query_gpus(mock=True)
    assert len(gpus) == 2
    assert all(g.memory_total_mb > 0 for g in gpus)
    # Synthetic names must be obviously fake.
    assert all("Synthetic" in g.name for g in gpus)


def test_query_processes_mock_returns_synthetic_rows():
    procs = nvidia.query_processes(mock=True)
    assert len(procs) == 2
    assert all(p.gpu_uuid.startswith("GPU-") for p in procs)
    assert {p.process_name for p in procs} == {"example-trainer", "example-inference"}


def test_set_power_limit_mock_does_not_execute():
    msg = nvidia.set_power_limit(0, 350, mock=True)
    assert "[mock]" in msg
    assert "nvidia-smi -i 0 -pl 350" in msg


def test_set_power_limit_rejects_nonpositive_watts():
    with pytest.raises(ValueError):
        nvidia.set_power_limit(0, 0, mock=True)
