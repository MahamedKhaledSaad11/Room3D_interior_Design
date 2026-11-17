from __future__ import annotations

import hashlib
import io
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

import httpx

from room3d_worker.common.subprocess_runner import StageResult, run_stage_on_gpu


class WorkerClientError(RuntimeError):
    """The backend rejected or could not complete a worker operation."""


class StaleAttemptError(WorkerClientError):
    """The task attempt fence no longer belongs to this worker."""


class RetryableWorkerError(WorkerClientError):
    """A transport interruption or temporary upstream failure can be retried."""


@dataclass(frozen=True)
class TaskClaim:
    task_id: str
    attempt_id: str
    project_id: str | None
    version_id: str | None
    task_type: str
    payload: dict[str, Any]
    worker_id: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TaskClaim:
        return cls(
            payload["task_id"],
            payload["attempt_id"],
            payload.get("project_id"),
            payload.get("version_id"),
            payload["task_type"],
            payload.get("payload", {}),
        )


def split_storage_key(key: str) -> tuple[str, str, str]:
    """Split `<project_id>/<scope>/<rest>` storage keys into router path components."""
    parts = key.lstrip("/").split("/", 2)
    if len(parts) != 3 or not all(parts):
        raise WorkerClientError(f"Invalid storage key '{key}'; expected '<project_id>/<scope>/<path>'")
    return parts[0], parts[1], parts[2]


class WorkerClient:
    BATCH_GPU = {
        "diffusion_preview": "cuda:0",
        "diffusion_apply": "cuda:0",
        "reconstruct_week1": "cuda:1",
        "build_week1": "cuda:1",
        "pose_benchmark": "cuda:1",
        "track_propagate": "cuda:1",
        "rebuild_week2": "cuda:1",
        "reconstruct": "cuda:1",
        "segment_track": "cuda:1",
        "diffusion_propagate": "cuda:0",
        "rebuild": "cuda:1",
    }
    INTERACTIVE_TASKS = {"segment_init", "segment_refine"}

    def __init__(
        self,
        base_url: str,
        worker_token: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
        upload_chunk_bytes: int = 8 * 1024 * 1024,
        upload_attempts: int = 4,
        retry_delay: float = 1.0,
    ) -> None:
        if upload_chunk_bytes <= 0 or upload_attempts < 1 or retry_delay < 0:
            raise ValueError("Upload chunk size and attempts must be positive; delay must be non-negative")
        self._upload_chunk_bytes = upload_chunk_bytes
        self._upload_attempts = upload_attempts
        self._retry_delay = retry_delay
        headers = {"X-Worker-Token": worker_token, "Accept": "application/json"}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> WorkerClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            raise RetryableWorkerError(f"Backend {method} {path} transport failed: {exc}") from exc
        if response.status_code == 409:
            raise StaleAttemptError(f"Backend rejected stale attempt for {path}")
        if response.status_code in {408, 429} or response.status_code >= 500:
            raise RetryableWorkerError(f"Backend {method} {path} temporarily failed: {response.status_code}")
        if response.status_code >= 400:
            raise WorkerClientError(
                f"Backend {method} {path} failed with {response.status_code}: {response.text}"
            )
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    def claim(
        self, worker_id: str, capabilities: list[str], *, interactive_only: bool = False
    ) -> TaskClaim | None:
        payload = self._request("POST",
            "/api/tasks/claim",
            json={
                "worker_id": worker_id,
                "capabilities": capabilities,
                "interactive_only": interactive_only,
            },
        )
        if not payload:
            return None
        task = TaskClaim.from_payload(payload)
        return TaskClaim(
            task.task_id,
            task.attempt_id,
            task.project_id,
            task.version_id,
            task.task_type,
            task.payload,
            worker_id,
        )

    def heartbeat(self, task: TaskClaim, worker_id: str | None = None) -> dict[str, Any]:
        payload = {"attempt_id": task.attempt_id, "worker_id": worker_id or task.worker_id or ""}
        return self._retry_post(f"/api/tasks/{task.task_id}/heartbeat", body=payload, timeout=30.0)

    def progress(
        self,
        task: TaskClaim,
        stage: str,
        progress_pct: float,
        message: str,
        telemetry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "attempt_id": task.attempt_id,
            "stage": stage,
            "progress_pct": progress_pct,
            "message": message,
        }
        if task.worker_id:
            payload["worker_id"] = task.worker_id
        if telemetry is not None:
            payload["telemetry"] = telemetry
        return self._retry_post(f"/api/tasks/{task.task_id}/progress", body=payload, timeout=30.0)

    def complete(
        self,
        task: TaskClaim,
        result_payload: dict[str, Any],
        result_manifest_key: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"attempt_id": task.attempt_id, "result_payload": result_payload}
        if task.worker_id:
            payload["worker_id"] = task.worker_id
        if result_manifest_key is not None:
            payload["result_manifest_key"] = result_manifest_key
        return self._request("POST", f"/api/tasks/{task.task_id}/complete", json=payload)

    def fail(
        self, task: TaskClaim, error_code: str, error_message: str, retryable: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "attempt_id": task.attempt_id,
            "error_code": error_code,
            "error_message": error_message,
            "retryable": retryable,
        }
        if task.worker_id:
            payload["worker_id"] = task.worker_id
        return self._request("POST", f"/api/tasks/{task.task_id}/fail", json=payload)

    def upload(
        self, project_id: str, version_id: str, path: str, source: BinaryIO | Path
    ) -> dict[str, Any]:
        handle = source.open("rb") if isinstance(source, Path) else source
        try:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(0)
            root = f"/api/storage/{project_id}/{version_id}"
            if size <= self._upload_chunk_bytes:
                return self._retry_post(f"{root}/upload", path, handle)
            return self._upload_chunked(root, path, handle, size)
        finally:
            if isinstance(source, Path):
                handle.close()

    def _retry_post(
        self, endpoint: str, filename: str | None = None, stream: BinaryIO | None = None,
        body: dict[str, object] | None = None, *, timeout: float = 120.0,
    ) -> dict[str, Any]:
        for attempt in range(self._upload_attempts):
            try:
                if stream is not None:
                    stream.seek(0)
                    return self._request("POST", endpoint, timeout=timeout,
                                         files={"file": (filename, stream, "application/octet-stream")})
                return self._request("POST", endpoint, json=body, timeout=timeout)
            except RetryableWorkerError:
                if attempt + 1 == self._upload_attempts:
                    raise
                time.sleep(self._retry_delay * 2 ** attempt)
        raise WorkerClientError(f"Request retries exhausted for {endpoint}")

    def _upload_chunked(self, root: str, path: str, handle: BinaryIO, size: int) -> dict[str, Any]:
        endpoint = f"{root}/uploads/{uuid.uuid4()}"
        digest = hashlib.sha256()
        count = 0
        while chunk := handle.read(self._upload_chunk_bytes):
            digest.update(chunk)
            self._retry_post(f"{endpoint}/parts/{count}", "part", io.BytesIO(chunk))
            count += 1
        return self._retry_post(f"{endpoint}/complete", body={
            "path": path, "part_count": count, "size": size, "sha256": digest.hexdigest(),
        })

    def upload_key(self, key: str, source: BinaryIO | Path) -> dict[str, Any]:
        project_id, scope, path = split_storage_key(key)
        return self.upload(project_id, scope, path, source)

    def download(self, project_id: str, version_id: str, path: str, destination: Path) -> None:
        with self._http.stream("GET", f"/api/storage/{project_id}/{version_id}/{path}") as response:
            if response.status_code >= 400:
                raise WorkerClientError(f"Storage download failed with {response.status_code}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)

    def download_key(self, key: str, destination: Path) -> None:
        project_id, scope, path = split_storage_key(key)
        self.download(project_id, scope, path, destination)

    @staticmethod
    def _chunks(handle: BinaryIO, chunk_size: int) -> Iterator[bytes]:
        while chunk := handle.read(chunk_size):
            yield chunk

    def dispatch_batch(
        self,
        task_type: str,
        command: list[str],
        *,
        timeout_sec: int,
        env_extra: dict[str, str] | None = None,
    ) -> StageResult:
        try:
            device = self.BATCH_GPU[task_type]
        except KeyError as exc:
            raise WorkerClientError(f"Unsupported batch task type: {task_type}") from exc
        preferred_gpu = int(device.split(":", 1)[1]) if ":" in device else 0
        return run_stage_on_gpu(command, preferred_gpu, env_extra, timeout_sec)

    def run_interactive_loop(
        self,
        worker_id: str,
        handler: Callable[[TaskClaim], dict[str, Any]],
        stop_event: threading.Event | None = None,
    ) -> None:
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            task = self.claim(worker_id, ["segment_init", "segment_refine"], interactive_only=True)
            if task is None:
                stop_event.wait(0.5)
                continue
            try:
                self.complete(task, handler(task))
            except StaleAttemptError:
                continue
            except Exception as exc:
                self.fail(task, type(exc).__name__, str(exc), retryable=True)
