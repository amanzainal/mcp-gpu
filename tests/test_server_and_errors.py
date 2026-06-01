"""Tests for the server tool layer and the missing-nvidia-smi error path."""

import pytest

from mcp_gpu import nvidia, server


def test_missing_nvidia_smi_raises_clear_error(monkeypatch):
    monkeypatch.delenv("MCP_GPU_MOCK", raising=False)
    monkeypatch.setattr(nvidia, "have_nvidia_smi", lambda: False)
    with pytest.raises(nvidia.NvidiaSmiNotFound) as exc:
        nvidia.query_gpus(mock=False)
    # Error must point the user at the fix.
    assert "nvidia-smi" in str(exc.value)
    assert "--mock" in str(exc.value)


def test_set_power_limit_missing_smi_raises(monkeypatch):
    monkeypatch.delenv("MCP_GPU_MOCK", raising=False)
    monkeypatch.setattr(nvidia, "have_nvidia_smi", lambda: False)
    with pytest.raises(nvidia.NvidiaSmiNotFound):
        nvidia.set_power_limit(0, 200, mock=False)


def test_server_tools_in_mock_mode():
    server.set_mock(True)
    try:
        gpus = server.list_gpus()
        assert isinstance(gpus, list) and len(gpus) == 2
        assert isinstance(gpus[0], dict)

        procs = server.gpu_processes()
        assert isinstance(procs, list) and len(procs) == 2

        summary = server.gpu_summary()
        assert isinstance(summary, str)
        assert "GPU 0" in summary

        msg = server.set_power_limit(1, 300)
        assert "[mock]" in msg
    finally:
        server.set_mock(False)


def test_query_gpus_uses_real_path_when_not_mock(monkeypatch):
    """When not mocked, query_gpus should call the subprocess runner."""
    monkeypatch.delenv("MCP_GPU_MOCK", raising=False)
    monkeypatch.setattr(nvidia, "have_nvidia_smi", lambda: True)
    captured = {}

    def fake_run(args):
        captured["args"] = args
        return "0, Example Synthetic GPU, 8192, 1024, 5, 40\n"

    monkeypatch.setattr(nvidia, "_run", fake_run)
    gpus = nvidia.query_gpus(mock=False)
    assert len(gpus) == 1
    # Verify the exact query fields requested.
    joined = " ".join(captured["args"])
    assert "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu" in joined
    assert "--format=csv,noheader,nounits" in joined
