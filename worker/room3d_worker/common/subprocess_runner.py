from __future__ import annotations

import importlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path

from room3d_worker.common.cancellation import cancellation_event, check_cancelled
from room3d_worker.common.errors import StageExecutionError
from room3d_worker.common.progress import report_progress


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


def _failure_message(context: str, stderr: str) -> str:
    # Native workers can log INFO lines after a fatal check and bury its cause.
    print(f"{context}\n{stderr}", file=sys.stderr, flush=True)
    context = context[:900]
    prefix = f"{context}; stderr tail="
    budget = 3200 - len(prefix)
    diagnostic = re.sub(r"(?m)^I\d{8}\s[^\n]*(?:\n|$)", "", stderr)
    if len(diagnostic) <= budget:
        return prefix + diagnostic
    fatal = re.findall(
        r"(?m)^.*(?:F\d{8}\s|Check failed|FATAL|SIGABRT|SIGSEGV|"
        r"terminate called|what\(\):|Fatal Python error).*$", diagnostic,
    )
    marker = "\n... stderr truncated ...\n"
    head = "\n".join(fatal)[:budget // 2] if fatal else diagnostic[:budget // 2]
    return prefix + head + marker + diagnostic[-(budget - len(head) - len(marker)):]


def run_isolated_stage(
    cmd: list[str], cuda_device: str | None, env_extra: dict[str, str] | None, timeout_sec: int,
    *, stall_timeout_sec: float | None = None,
) -> StageResult:
    if not cmd or timeout_sec <= 0:
        raise StageExecutionError("Stage command must be non-empty and timeout_sec must be positive")
    if stall_timeout_sec is not None and stall_timeout_sec <= 0:
        raise StageExecutionError("stall_timeout_sec must be positive")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = _normalize_cuda_device(cuda_device)
    env["QT_QPA_PLATFORM"] = "offscreen"
    started = time.perf_counter()
    try:
        result = _run_cancellable(cmd, env, timeout_sec, stall_timeout_sec) if cancellation_event() is not None or stall_timeout_sec is not None else subprocess.run(
            cmd, env=env, check=False, capture_output=True, text=True, timeout=timeout_sec, shell=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raw_stderr = exc.stderr if isinstance(exc, subprocess.TimeoutExpired) else None
        stderr = raw_stderr.decode("utf-8", errors="replace") if isinstance(raw_stderr, bytes) else raw_stderr or ""
        message = _failure_message(f"Stage command failed to execute {cmd!r}: {exc}", stderr)
        raise StageExecutionError(message) from exc
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        message = _failure_message(f"Stage command exited {result.returncode}: {cmd!r}", result.stderr)
        raise StageExecutionError(message)
    return StageResult(tuple(cmd), result.returncode, elapsed, result.stdout, result.stderr)


def _terminate_tree(process: subprocess.Popen[bytes]) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def _run_cancellable(
    cmd: list[str], env: dict[str, str], timeout: int, stall_timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    check_cancelled()
    watchdog = _StageWatchdog(env, timeout, stall_timeout)
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=os.name != "nt")
    captured = _StageCapture(process)
    try:
        while True:
            check_cancelled()
            watchdog.observe(captured)
            returncode = process.poll()
            if returncode is not None and captured.finished():
                stdout, stderr = captured.text()
                return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
            if watchdog.expired() or watchdog.stalled():
                _terminate_tree(process)
                captured.join()
                stdout, stderr = captured.text()
                if watchdog.stalled():
                    raise StageExecutionError(_failure_message(
                        f"Stage stalled: no subprocess output or progress for {stall_timeout}s (absolute budget {timeout}s)", stderr,
                    ))
                raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
            time.sleep(0.1)
    except BaseException:
        _terminate_tree(process)
        raise
    finally:
        # A killed Python stage may leave COLMAP/CUDA children holding its output pipes.
        if process.poll() is None:
            _terminate_tree(process)
        process.wait()
        captured.join()


class _StageWatchdog:
    def __init__(self, env: dict[str, str], timeout: int, stall_timeout: float | None) -> None:
        self.env, self.timeout, self.stall_timeout = env, timeout, stall_timeout
        self.started = self.last_activity = self.last_loading_report = time.monotonic()
        self.observed = (0, 0, 0)
        self.loading_finished = False
        self.progress_path = Path(env["ROOM3D_PROGRESS_FILE"]) if env.get("ROOM3D_PROGRESS_FILE") else None

    def observe(self, captured: _StageCapture) -> None:
        captured.check_errors()
        error_file = self.env.get("ROOM3D_RELAY_ERROR_FILE")
        if error_file and Path(error_file).exists():
            raise StageExecutionError("Progress/checkpoint relay failed: " + Path(error_file).read_text(encoding="utf-8"))
        path = self.progress_path
        file_size = path.stat().st_size if path and path.exists() else 0
        activity = (*captured.sizes(), file_size)
        if activity == self.observed:
            return
        if file_size != self.observed[2] and path:
            with path.open("rb") as stream:
                stream.seek(max(0, file_size - 8192))
                tail = stream.read()
            self.loading_finished |= b'"phase":"model_ready"' in tail or b'"denoising_step"' in tail
        self.last_activity, self.observed = time.monotonic(), activity
        if not self.loading_finished and self.env.get("ROOM3D_STAGE_PROGRESS_ID") and self.last_activity - self.last_loading_report >= 2:
            _report_loading(captured.stderr_tail(), self.last_activity - self.started)
            self.last_loading_report = self.last_activity

    def expired(self) -> bool:
        return time.monotonic() - self.started >= self.timeout

    def stalled(self) -> bool:
        return self.stall_timeout is not None and time.monotonic() - self.last_activity >= self.stall_timeout


class _StageCapture:
    """Drain both pipes continuously; Windows communicate() hides partial output until EOF."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.buffers: list[list[bytes]] = [[], []]
        self.counts = [0, 0]
        self.threads: list[threading.Thread] = []
        self.streams = (process.stdout, process.stderr)
        self.errors: list[OSError] = []
        for index, stream in enumerate((process.stdout, process.stderr)):
            assert stream is not None
            thread = threading.Thread(target=self._drain, args=(stream.fileno(), index), daemon=True)
            self.threads.append(thread)
            thread.start()

    def _drain(self, descriptor: int, index: int) -> None:
        try:
            while chunk := os.read(descriptor, 4096):
                self.buffers[index].append(chunk)
                self.counts[index] += len(chunk)
        except OSError as exc:
            self.errors.append(exc)

    def check_errors(self) -> None:
        if self.errors:
            raise StageExecutionError(f"Cannot read subprocess output: {self.errors[0]}") from self.errors[0]

    def sizes(self) -> tuple[int, int]:
        return self.counts[0], self.counts[1]

    def stderr_tail(self) -> bytes:
        return b"".join(self.buffers[1][-4:])[-4000:]

    def text(self) -> tuple[str, str]:
        return (b"".join(self.buffers[0]).decode("utf-8", errors="replace"),
                b"".join(self.buffers[1]).decode("utf-8", errors="replace"))

    def finished(self) -> bool:
        return all(not thread.is_alive() for thread in self.threads)

    def join(self) -> None:
        for thread in self.threads:
            thread.join()
        for stream in self.streams:
            if stream:
                stream.close()


def _report_loading(stderr: bytes | str | None, elapsed: float) -> None:
    text = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr or ""
    matches = list(re.finditer(r"(Fetching \d+ files|Loading pipeline components|Loading weights)[^\r\n]*", text[-4000:]))
    if not matches:
        return
    phase = matches[-1].group(1)
    counts = re.search(r"(\d+)/(\d+)", matches[-1].group())
    measured = f" {counts.group()}" if counts else ""
    # Only expose measured loader output, never a timer pretending to advance the model.
    report_progress("06_diffusion", 0, phase + measured + " (loader active)",
                    phase="download" if phase.startswith("Fetching") else "model_load", elapsed_sec=round(elapsed, 1))


def run_stage_on_gpu(
    cmd: list[str],
    preferred_gpu: int | None,
    env_extra: dict[str, str] | None,
    timeout_sec: int,
    *,
    gpu_count: int | None = None,
    stall_timeout_sec: float | None = None,
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
    return run_isolated_stage(cmd, cuda_device, extra, timeout_sec, stall_timeout_sec=stall_timeout_sec)
