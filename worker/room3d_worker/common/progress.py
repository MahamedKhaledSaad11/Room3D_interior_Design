from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from pydantic import BaseModel, Field

from room3d_worker.common.cancellation import check_cancelled


class ProgressEvent(BaseModel):
    stage: str
    percent: float = Field(ge=0, le=100, allow_inf_nan=False)
    message: str
    telemetry: dict[str, float | int | str] = Field(default_factory=dict)


_sink: ContextVar[Callable[[ProgressEvent], None] | None] = ContextVar("progress_sink", default=None)


def report_progress(stage: str, percent: float, message: str, **telemetry: float | int | str) -> None:
    check_cancelled()
    event = ProgressEvent(stage=stage, percent=percent, message=message, telemetry=telemetry)
    destination = os.environ.get("ROOM3D_PROGRESS_FILE")
    if destination:
        with Path(destination).open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")
    elif callback := _sink.get():
        callback(event)


@contextmanager
def progress_sink(callback: Callable[[ProgressEvent], None]) -> Iterator[None]:
    token = _sink.set(callback)
    try:
        yield
    finally:
        _sink.reset(token)


@contextmanager
def relay_subprocess_progress(
    path: Path, on_poll: Callable[[], None] | None = None,
) -> Iterator[dict[str, str]]:
    callback = _sink.get()
    path.write_text("", encoding="utf-8")
    failed_path = path.with_suffix(".relay-error")
    failed_path.unlink(missing_ok=True)
    stopped = threading.Event()
    errors: list[Exception] = []

    def relay() -> None:
        try:
            with path.open(encoding="utf-8") as stream:
                while True:
                    latest: dict[str, ProgressEvent] = {}
                    while True:
                        position = stream.tell()
                        line = stream.readline()
                        if not line.endswith("\n"):
                            stream.seek(position)
                            break
                        event = ProgressEvent.model_validate_json(line)
                        latest[event.stage] = event
                    if on_poll:
                        on_poll()
                    if callback:
                        for event in latest.values():
                            callback(event)
                    if stopped.is_set():
                        if not latest:
                            return
                        continue
                    stopped.wait(1.0)
        except Exception as exc:
            errors.append(exc)
            failed_path.write_text(str(exc), encoding="utf-8")

    thread = threading.Thread(target=relay, name="stage-progress", daemon=True)
    thread.start()
    try:
        yield {"ROOM3D_PROGRESS_FILE": str(path.resolve()), "ROOM3D_RELAY_ERROR_FILE": str(failed_path.resolve())}
    finally:
        stopped.set()
        thread.join()
    if errors:
        raise errors[0]
