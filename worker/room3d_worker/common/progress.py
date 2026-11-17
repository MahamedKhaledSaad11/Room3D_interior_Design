from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from pydantic import BaseModel, Field


class ProgressEvent(BaseModel):
    stage: str
    percent: float = Field(ge=0, le=100, allow_inf_nan=False)
    message: str
    telemetry: dict[str, float | int | str] = Field(default_factory=dict)


_sink: ContextVar[Callable[[ProgressEvent], None] | None] = ContextVar("progress_sink", default=None)


def report_progress(stage: str, percent: float, message: str, **telemetry: float | int | str) -> None:
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
def relay_subprocess_progress(path: Path) -> Iterator[dict[str, str]]:
    callback = _sink.get()
    path.write_text("", encoding="utf-8")
    stopped = threading.Event()
    errors: list[Exception] = []

    def relay() -> None:
        try:
            with path.open(encoding="utf-8") as stream:
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if line.endswith("\n"):
                        if callback:
                            callback(ProgressEvent.model_validate_json(line))
                        continue
                    stream.seek(position)
                    if stopped.is_set():
                        return
                    stopped.wait(0.25)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=relay, name="stage-progress", daemon=True)
    thread.start()
    try:
        yield {"ROOM3D_PROGRESS_FILE": str(path.resolve())}
    finally:
        stopped.set()
        thread.join()
    if errors:
        raise errors[0]
