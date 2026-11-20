from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from room3d_worker.common.progress import report_progress


class GenerationProgress:
    def __init__(self, total: int, frames: int, resumed: int = 0) -> None:
        self.total, self.frames, self.completed = total, frames, resumed
        self.resumed = resumed
        self.started = time.monotonic()
        self.label = ""

    def begin(self, indices: list[int]) -> None:
        self.label = ", ".join(str(idx) for idx in indices)
        self.step(0, 1)

    def step(self, step: int, steps: int) -> None:
        fraction = step / max(steps, 1)
        elapsed = time.monotonic() - self.started
        measured_work = self.completed - self.resumed + fraction
        telemetry: dict[str, float | int | str] = {
            "phase": "inference",
            "operations_completed": self.completed, "operations_total": self.total,
            "frames_to_inpaint": self.frames, "denoising_step": step, "denoising_steps": steps,
            "elapsed_sec": round(elapsed, 1), "operations_resumed": self.resumed,
        }
        if measured_work > 0:
            remaining = max(0, self.total - self.completed - fraction)
            telemetry["eta_sec"] = round(elapsed / measured_work * remaining, 1)
        report_progress("06_diffusion", 10 + 80 * (self.completed + fraction) / max(self.total, 1),
                        f"Inference {min(self.completed + 1, self.total)}/{self.total}; frames {self.label}; step {step}/{steps}",
                        **telemetry)

    def finish(self) -> None:
        self.completed += 1
        self.step(0, 1)


_generation: ContextVar[GenerationProgress | None] = ContextVar("generation_progress", default=None)


def generation_step(step: int, steps: int) -> None:
    if tracker := _generation.get():
        tracker.step(step, steps)


@contextmanager
def generation_scope(tracker: GenerationProgress) -> Iterator[None]:
    token = _generation.set(tracker)
    try:
        yield
    finally:
        _generation.reset(token)
