from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from room3d_worker.client import WorkerClient
from room3d_worker.common.cancellation import TaskInterruptedError, cancellation_scope
from room3d_worker.common.subprocess_runner import run_isolated_stage
from room3d_worker.runner import WorkerRunner


def test_cancel_terminates_stage_and_child_process(tmp_path: Path) -> None:
    marker = tmp_path / "orphan.txt"
    child = f"import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).write_text('orphan')"
    script = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(20)"
    stopped = threading.Event()
    timer = threading.Timer(0.5, stopped.set)
    timer.start()
    started = time.monotonic()
    try:
        with cancellation_scope(stopped), pytest.raises(TaskInterruptedError):
            run_isolated_stage([sys.executable, "-c", script], None, None, 10)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 5
    time.sleep(2)
    assert not marker.exists(), "A child of the interrupted stage kept running"


def test_worker_acknowledges_pause_without_completing_or_failing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    heartbeats = 0

    def bridge(request: httpx.Request) -> httpx.Response:
        nonlocal heartbeats
        calls.append(request.url.path)
        if request.url.path.endswith("/claim"):
            return httpx.Response(200, json={"task_id": "task", "attempt_id": "attempt", "project_id": "room",
                                           "task_type": "reconstruct", "payload": {}})
        if request.url.path.endswith("/heartbeat"):
            heartbeats += 1
            return httpx.Response(200, json={"control_action": "pause" if heartbeats > 1 else None})
        return httpx.Response(200, json={"ok": True})

    with WorkerClient("http://testserver", "secret", transport=httpx.MockTransport(bridge)) as client:
        runner = WorkerRunner(client, work_root=tmp_path, heartbeat_interval_sec=0.05)

        def execute(task: object) -> tuple[dict[str, object], None]:
            run_isolated_stage([sys.executable, "-c", "import time; time.sleep(20)"], None, None, 10)
            return {}, None

        monkeypatch.setattr(runner, "execute_claim", execute)
        assert runner.run_once()
    assert "/api/tasks/task/interrupted" in calls
    assert "/api/tasks/task/complete" not in calls
    assert "/api/tasks/task/fail" not in calls
