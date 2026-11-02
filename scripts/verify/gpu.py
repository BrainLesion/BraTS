"""GPU-free gate ensuring the verification container never contends for a GPU."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Optional

from brats.utils.cuda import normalize_cuda_devices

SmiRunner = Callable[[list[str], int], subprocess.CompletedProcess]


class GpuError(RuntimeError):
    """Raised when the GPU state cannot be determined."""


@dataclass(frozen=True)
class _GpuState:
    index: int
    uuid: str
    utilization: int
    memory_used_mb: int


def parse_device_ids(cuda_devices: str) -> list[int]:
    """Parse a comma-separated CUDA device string into integer indices.

    Args:
        cuda_devices (str): Comma-separated device ids, e.g. ``"0,1"``

    Returns:
        List[int]: Parsed device indices

    Raises:
        GpuError: If the string is empty or a device id is not an integer
    """
    try:
        normalized = normalize_cuda_devices(cuda_devices)
        return [int(part) for part in normalized.split(",")]
    except ValueError as e:
        raise GpuError(f"Invalid CUDA device id in {cuda_devices!r}") from e


def _default_runner(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, check=False
    )


def _run_smi(query_type: str, query: str, runner: SmiRunner, timeout: int) -> str:
    args = [
        "nvidia-smi",
        f"{query_type}={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = runner(args, timeout)
    except FileNotFoundError as e:
        raise GpuError("nvidia-smi not found on PATH") from e
    except (OSError, subprocess.SubprocessError) as e:
        raise GpuError(f"Failed to run nvidia-smi: {e}") from e
    if proc.returncode != 0:
        raise GpuError(f"nvidia-smi failed: {proc.stderr.strip()}")
    return proc.stdout


def _gpu_states(runner: SmiRunner, timeout: int) -> dict[int, _GpuState]:
    raw = _run_smi(
        "--query-gpu", "index,uuid,utilization.gpu,memory.used", runner, timeout
    )
    states: dict[int, _GpuState] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise GpuError(f"Unexpected nvidia-smi gpu row: {line!r}")
        try:
            states[int(parts[0])] = _GpuState(
                index=int(parts[0]),
                uuid=parts[1],
                utilization=int(parts[2]),
                memory_used_mb=int(parts[3]),
            )
        except ValueError as e:
            raise GpuError(f"Could not parse nvidia-smi gpu row: {line!r}") from e
    if not states:
        raise GpuError("nvidia-smi reported no GPUs")
    return states


def _busy_uuids(runner: SmiRunner, timeout: int) -> set[str]:
    raw = _run_smi("--query-compute-apps", "gpu_uuid,pid,used_memory", runner, timeout)
    busy: set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise GpuError(f"Unexpected nvidia-smi compute row: {line!r}")
        busy.add(parts[0])
    return busy


def _check_once(
    ids: list[int], runner: SmiRunner, timeout: int, idle_mem_mb: int
) -> bool:
    states = _gpu_states(runner, timeout)
    busy = _busy_uuids(runner, timeout)
    for device_id in ids:
        state = states.get(device_id)
        if state is None:
            raise GpuError(f"GPU index {device_id} does not exist")
        if state.uuid in busy:
            return False
        if state.memory_used_mb > idle_mem_mb:
            return False
        if state.utilization != 0:
            return False
    return True


def ensure_free(
    ids: list[int],
    *,
    idle_mem_mb: int = 500,
    wait_s: int = 600,
    poll_s: int = 20,
    timeout: int = 30,
    runner: Optional[SmiRunner] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Wait until the requested GPUs are idle.

    A GPU is considered free when there is no compute process on it, its memory
    usage is at most ``idle_mem_mb`` and its utilization is zero. The gate never
    interrupts other users; it only waits and reports.

    Args:
        ids (List[int]): Host GPU indices to check
        idle_mem_mb (int): Maximum tolerated memory usage in MB
        wait_s (int): Maximum time to wait for the GPUs to free up
        poll_s (int): Poll interval in seconds
        timeout (int): Timeout for each nvidia-smi invocation
        runner (Optional[SmiRunner]): Injectable nvidia-smi runner (for tests)
        sleep (Callable): Injectable sleep (for tests)

    Returns:
        bool: True if the GPUs are free, False if still busy after ``wait_s``

    Raises:
        GpuError: If nvidia-smi is missing, fails or reports an invalid device
    """
    smi_runner = runner or _default_runner
    if runner is None and shutil.which("nvidia-smi") is None:
        raise GpuError("nvidia-smi not found on PATH")

    deadline = time.monotonic() + max(wait_s, 0)
    while True:
        if _check_once(ids, smi_runner, timeout, idle_mem_mb):
            return True
        if time.monotonic() >= deadline:
            return False
        sleep(poll_s)
