from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass

from room3d_worker.common.errors import StageExecutionError


@dataclass(frozen=True)
class StageResult:
    command: tuple[str, ...]
    returncode: int
    elapsed_sec: float
    stdout: str
    stderr: str


def _normalize_cuda_device(cuda_device: str) -> str:
    normalized = cuda_device.strip()
    if not normalized:
        raise StageExecutionError("cuda_device must not be empty or whitespace")
    if normalized.lower().startswith("cuda:"):
        normalized = normalized[5:]
    if not normalized:
        raise StageExecutionError("cuda_device must include a device index or UUID")
    return normalized


def available_gpu_count() -> int:
    """Return number of visible CUDA GPUs without failing in CPU-only environments."""
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except (ImportError, RuntimeError, OSError):
        pass
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return 0
    try:
        out = subprocess.run(
            [nvidia_smi, "-L"], check=False, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if out.returncode != 0:
        return 0
    return sum(1 for line in out.stdout.splitlines() if line.strip().startswith("GPU "))


def resolve_physical_gpu(preferred_gpu: int, gpu_count: int | None = None) -> int | None:
    """Map logical GPU ordinal (0=diffusion, 1=SAM/3DGS) to available hardware."""
    count = available_gpu_count() if gpu_count is None else gpu_count
    if count <= 0:
        return None
    if 0 <= preferred_gpu < count:
        return preferred_gpu
    return 0


def run_isolated_stage(
    cmd: list[str], cuda_device: str | None, env_extra: dict[str, str] | None, timeout_sec: int
) -> StageResult:
    if not cmd or timeout_sec <= 0:
        raise StageExecutionError("Stage command must be non-empty and timeout_sec must be positive")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = _normalize_cuda_device(cuda_device)
    started = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, env=env, check=False, capture_output=True, text=True, timeout=timeout_sec, shell=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StageExecutionError(f"Stage command failed to execute {cmd!r}: {exc}") from exc
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise StageExecutionError(
            f"Stage command exited {result.returncode}: {cmd!r}; stderr={result.stderr.strip()}"
        )
    return StageResult(tuple(cmd), result.returncode, elapsed, result.stdout, result.stderr)


def run_stage_on_gpu(
    cmd: list[str],
    preferred_gpu: int | None,
    env_extra: dict[str, str] | None,
    timeout_sec: int,
    *,
    gpu_count: int | None = None,
) -> StageResult:
    """Run an isolated stage subprocess using resolve_physical_gpu; subprocess sees cuda:0."""
    extra = dict(env_extra or {})
    cuda_device: str | None = None
    if preferred_gpu is not None:
        extra["ROOM3D_ASSIGNED_GPU"] = str(preferred_gpu)
        physical = resolve_physical_gpu(preferred_gpu, gpu_count=gpu_count)
        if physical is not None:
            cuda_device = str(physical)
            extra["ROOM3D_LOGICAL_CUDA_DEVICE"] = "cuda:0"
    return run_isolated_stage(cmd, cuda_device, extra, timeout_sec)
