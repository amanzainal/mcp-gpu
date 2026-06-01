"""Synthetic nvidia-smi CSV fixtures.

These mimic the exact output of:
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits

All values are obviously fake. No real device names or paths.
"""

# Two GPUs, one busy, one idle. nvidia-smi emits a leading space after commas.
GPU_CSV = (
    "0, Example Synthetic GPU, 24576, 12000, 87, 71\n"
    "1, Example Synthetic GPU, 24576, 256, 0, 34\n"
)

# A GPU that reports [N/A] for a field (e.g. virtualized / unsupported sensor).
GPU_CSV_WITH_NA = "0, Example Synthetic GPU, 16384, 4096, [N/A], 60\n"

# Compute-apps query with two processes.
PROC_CSV = (
    "GPU-aaaaaaaa-0000-0000-0000-000000000000, 4242, example-trainer, 12000\n"
    "GPU-bbbbbbbb-1111-1111-1111-111111111111, 5353, example-server, 256\n"
)

# When nothing is running, nvidia-smi prints a single comma-free line.
PROC_CSV_EMPTY = "No running processes found\n"
