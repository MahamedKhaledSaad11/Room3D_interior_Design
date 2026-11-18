from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

from room3d_worker.client import TaskClaim, WorkerClient
from room3d_worker.common.progress import ProgressEvent, progress_sink, relay_subprocess_progress
from room3d_worker.common.subprocess_runner import run_stage_on_gpu
from room3d_worker.runner import WorkerRunner


@pytest.mark.parametrize(("task_type", "handler"), [
    ("reconstruct", "_handle_week1"), ("segment_track", "_handle_track"),
    ("diffusion_preview", "_handle_diffusion_preview"),
    ("diffusion_propagate", "_handle_diffusion_apply"), ("rebuild", "_handle_rebuild"),
])
def test_daemon_dispatches_all_live_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_type: str, handler: str,
) -> None:
    requests: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return httpx.Response(200, json={"ok": True})

    with WorkerClient("https://backend.test", "secret", transport=httpx.MockTransport(transport)) as client:
        runner = WorkerRunner(client, work_root=tmp_path)
        monkeypatch.setattr(runner, handler, lambda task: ({"handled": task.task_type}, None))
        task = TaskClaim("task", "attempt", "project", "v1", task_type, {})
        result, _ = runner.execute_claim(task)
    assert result == {"handled": task_type}
    assert requests == ["/api/tasks/task/progress"] * 2


def test_isolated_stage_relays_real_progress_to_parent(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    script = (
        "from room3d_worker.common.progress import report_progress; "
        "report_progress('07_rebuild',25,'step 1',iteration=1); "
        "report_progress('07_rebuild',100,'step 4',iteration=4)"
    )
    with progress_sink(events.append):
        with relay_subprocess_progress(tmp_path / "stage.jsonl") as environment:
            run_stage_on_gpu([sys.executable, "-c", script], None, environment, 15)
    assert events and events[-1].percent == 100
    assert [event.percent for event in events] == sorted(event.percent for event in events)
    assert events[-1].telemetry == {"iteration": 4}


def test_tunnel_progress_interruption_retries_without_restarting_task() -> None:
    calls = 0

    def transient(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503 if calls == 1 else 200, json={"ok": True})

    task = TaskClaim("task", "attempt", "project", "v1", "rebuild", {})
    with WorkerClient("https://backend.test", "secret", transport=httpx.MockTransport(transient), retry_delay=0) as client:
        assert client.progress(task, "07_rebuild", 50, "step 2500")["ok"]
    assert calls == 2


def test_progress_relay_coalesces_backlog_and_delivers_final_event(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    path = tmp_path / "burst.jsonl"
    with progress_sink(events.append), relay_subprocess_progress(path):
        with path.open("a", encoding="utf-8") as stream:
            for percent in range(1, 101):
                stream.write(ProgressEvent(stage="07_rebuild", percent=percent, message=f"step {percent}").model_dump_json() + "\n")
    assert events[-1].percent == 100
    assert len(events) <= 3
