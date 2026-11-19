from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class TaskInterruptedError(RuntimeError):
    """The owner paused or stopped this task."""


_cancellation: ContextVar[threading.Event | None] = ContextVar("task_cancellation", default=None)


def cancellation_event() -> threading.Event | None:
    return _cancellation.get()


def check_cancelled() -> None:
    event = cancellation_event()
    if event is not None and event.is_set():
        raise TaskInterruptedError("Project processing interrupted by user")


@contextmanager
def cancellation_scope(event: threading.Event) -> Iterator[None]:
    token = _cancellation.set(event)
    try:
        yield
    finally:
        _cancellation.reset(token)
