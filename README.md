# mcp-gpu

> An MCP server that exposes local NVIDIA GPU stats and controls to any agent.

`mcp-gpu` is a small [Model Context Protocol](https://modelcontextprotocol.io)
server that lets any MCP client (Claude Desktop, Claude Code, or your own agent)
read live NVIDIA GPU telemetry — memory, utilization, temperature, running
processes — and adjust the per-GPU power limit, all by shelling out to
`nvidia-smi`. It speaks MCP over **stdio**.

## Why it exists

Agents that run local training, inference, or rendering jobs are flying blind:
they can't see whether a GPU is saturated, out of memory, or running hot, and
they can't throttle power without a human at a terminal. `mcp-gpu` gives an
agent a tiny, well-scoped window into the local GPUs — and nothing else.

It is built to run **anywhere**, including machines with no GPU at all: a
built-in **mock mode** emits obviously-synthetic GPU rows so the server (and its
test suite) work on a laptop or in CI.

## Features

- `list_gpus()` — index, name, total/used/free memory (MiB), memory-used %,
  utilization %, temperature (C) for every GPU.
- `gpu_processes()` — every compute process: owning GPU UUID, PID, process name,
  and GPU memory used (MiB).
- `gpu_summary()` — one compact, human-readable string for all GPUs.
- `set_power_limit(index, watts)` — set a GPU's power cap (requires sudo/root on
  real hardware; see below).
- **Mock mode** (`--mock` or `MCP_GPU_MOCK=1`) — synthetic data, no GPU needed.
- Graceful, actionable error when `nvidia-smi` is missing.

## Install

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <your-fork-url> mcp-gpu
cd mcp-gpu
uv sync
```

That installs the official `mcp` SDK and the package itself, exposing the
`mcp-gpu` entry point inside the project environment.

## Usage

The server is normally launched by an MCP client, but you can run it by hand:

```bash
# Serve real nvidia-smi data over stdio
uv run mcp-gpu

# Serve synthetic data — no GPU required (great for trying it out / CI)
uv run mcp-gpu --mock

# Same, via env var
MCP_GPU_MOCK=1 uv run mcp-gpu
```

### Example tool output

`list_gpus()` (mock mode) returns structured rows:

```json
[
  {
    "index": 0,
    "name": "Synthetic GPU Model A",
    "memory_total_mb": 24576,
    "memory_used_mb": 8192,
    "utilization_pct": 42,
    "temperature_c": 55,
    "memory_free_mb": 16384,
    "memory_used_pct": 33.3
  },
  {
    "index": 1,
    "name": "Synthetic GPU Model A",
    "memory_total_mb": 24576,
    "memory_used_mb": 1024,
    "utilization_pct": 3,
    "temperature_c": 38,
    "memory_free_mb": 23552,
    "memory_used_pct": 4.2
  }
]
```

`gpu_summary()` returns one compact string:

```text
2 GPU(s) detected:
GPU 0 (Synthetic GPU Model A): 42% util, 8192/24576 MiB (33.3%), 55C
GPU 1 (Synthetic GPU Model A): 3% util, 1024/24576 MiB (4.2%), 38C
```

`set_power_limit(0, 350)` in mock mode reports the command it *would* run:

```text
[mock] would run: nvidia-smi -i 0 -pl 350
```

### Setting the power limit (sudo required)

On real hardware, `set_power_limit(index, watts)` runs:

```bash
nvidia-smi -i <index> -pl <watts>
```

That command requires **root/sudo**. If you want an agent to use this tool, run
the server (or just that command) with sufficient privileges; otherwise the tool
returns a clear permission error. Power-limit changes are not persistent across
reboots.

## MCP client configuration

Add an entry to your MCP client config (for Claude Desktop this is
`claude_desktop_config.json`; Claude Code uses `.mcp.json` or `claude mcp add`).
Use an absolute path to the cloned repo:

```json
{
  "mcpServers": {
    "mcp-gpu": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/mcp-gpu", "mcp-gpu"],
      "env": {
        "MCP_GPU_MOCK": "0"
      }
    }
  }
}
```

Set `MCP_GPU_MOCK` to `1` (or add `"--mock"` to `args`) to serve synthetic data
on a machine with no GPU.

## How it works

- **Telemetry** comes from `nvidia-smi --query-gpu=...` and
  `--query-compute-apps=...` with `--format=csv,noheader,nounits`. The output is
  parsed by pure functions (string in, dataclass out), which keeps parsing fully
  unit-testable and independent of any real GPU.
- **Mock mode** swaps the subprocess call for a fixed synthetic CSV string, so
  the exact same parsing path runs on GPU-less machines and in CI.
- **Control** (`set_power_limit`) builds the `nvidia-smi -i <i> -pl <w>` argv via
  a dedicated, testable helper and then executes it.
- The MCP layer is the official `mcp` SDK's `FastMCP`, served over **stdio**.

```
MCP client  <--stdio-->  FastMCP server  -->  nvidia.py  -->  nvidia-smi
                                                  |
                                            (mock CSV in mock mode)
```

## Development

```bash
uv sync
uv run pytest -q
```

The tests assert parsing of fixture `nvidia-smi` CSV (including `[N/A]` cells and
the empty-process message), the mock-data path, summary formatting, the
power-limit command shape, and the missing-`nvidia-smi` error.

## Roadmap

- `set_persistence_mode` and `set_gpu_clocks` controls.
- Per-process kill / signal (guarded, opt-in).
- Optional NVML (`pynvml`) backend to avoid the `nvidia-smi` fork per call.
- MCP resources exposing a live, pollable GPU snapshot.
- AMD ROCm (`rocm-smi`) backend behind the same tool surface.

## License

MIT — see [LICENSE](LICENSE).
