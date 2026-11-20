from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from room3d_worker.common.cancellation import check_cancelled


@dataclass(frozen=True)
class UploadEvent:
    phase: str
    message: str
    telemetry: dict[str, float | int | str]


_sink: ContextVar[Callable[[UploadEvent], None] | None] = ContextVar("upload_progress", default=None)


@contextmanager
def upload_progress_sink(callback: Callable[[UploadEvent], None]) -> Iterator[None]:
    token = _sink.set(callback)
    try:
        yield
    finally:
        _sink.reset(token)


class UploadProgress:
    def __init__(self, total: int, parts: int, resumed: int = 0) -> None:
        self.total, self.parts, self.resumed = total, parts, resumed
        self.started = time.monotonic()
        self.budget = float(os.environ.get("ROOM3D_UPLOAD_BUDGET_SEC", "7200"))
        if not math.isfinite(self.budget) or self.budget <= 0:
            raise ValueError("ROOM3D_UPLOAD_BUDGET_SEC must be finite and positive")

    def report(self, phase: str, confirmed: int, completed: int, message: str, attempt: int = 1) -> None:
        check_cancelled()
        elapsed = max(time.monotonic() - self.started, 1e-6)
        if elapsed >= self.budget:
            raise TimeoutError("Artifact upload budget exhausted; verified server parts are retained for retry")
        callback = _sink.get()
        if callback is None:
            return
        telemetry: dict[str, float | int | str] = {
            "phase": phase, "upload_confirmed_bytes": confirmed, "upload_total_bytes": self.total,
            "upload_parts_completed": completed, "upload_parts_total": self.parts,
            "upload_attempt": attempt, "upload_resumed_bytes": self.resumed,
            "upload_elapsed_sec": round(elapsed, 1),
        }
        speed = max(0, confirmed - self.resumed) / elapsed
        if speed > 0:
            telemetry["upload_speed_bytes_sec"] = round(speed, 1)
            if phase == "uploading":
                telemetry["upload_eta_sec"] = round(max(0, self.total - confirmed) / speed, 1)
        callback(UploadEvent(phase, message, telemetry))
