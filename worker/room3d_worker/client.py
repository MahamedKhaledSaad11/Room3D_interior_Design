from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import shutil
import stat
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator, cast
from zipfile import ZIP_DEFLATED, ZipFile

import httpx

from room3d_worker.common.assets import (
    AssetInfo,
    AssetRevisionError,
    asset_revision,
    safe_asset_path,
)
from room3d_worker.common.cancellation import check_cancelled
from room3d_worker.common.subprocess_runner import StageResult, run_stage_on_gpu
from room3d_worker.common.upload_progress import UploadProgress


class WorkerClientError(RuntimeError):
    """The backend rejected or could not complete a worker operation."""


class StaleAttemptError(WorkerClientError):
    """The task attempt fence no longer belongs to this worker."""


class RetryableWorkerError(WorkerClientError):
    """A transport interruption or temporary upstream failure can be retried."""


def _upload_setting(name: str, default: float) -> float:
    value = float(os.environ.get(f"ROOM3D_UPLOAD_{name}", str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"ROOM3D_UPLOAD_{name} must be finite and positive")
    return value


def _retry_wait(delay: float) -> None:
    end = time.monotonic() + delay
    while time.monotonic() < end:
        check_cancelled()
        time.sleep(min(0.1, max(0, end - time.monotonic())))
    check_cancelled()


def _stream_digest(stream: BinaryIO) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        check_cancelled()
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _part_matches(status: dict[str, Any], index: int, chunk: bytes) -> bool:
    entry = (status.get("parts") or {}).get(str(index)) or {}
    return entry.get("size") == len(chunk) and entry.get("sha256") == hashlib.sha256(chunk).hexdigest()


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
        upload_chunk_bytes: int | None = None,
        upload_attempts: int | None = None,
        retry_delay: float = 1.0,
    ) -> None:
        upload_chunk_bytes = upload_chunk_bytes if upload_chunk_bytes is not None else int(os.environ.get("ROOM3D_UPLOAD_CHUNK_BYTES", "1048576"))
        upload_attempts = upload_attempts if upload_attempts is not None else int(os.environ.get("ROOM3D_UPLOAD_ATTEMPTS", "6"))
        if upload_chunk_bytes <= 0 or upload_attempts < 1 or retry_delay < 0:
            raise ValueError("Upload chunk size and attempts must be positive; delay must be non-negative")
        self._upload_chunk_bytes = upload_chunk_bytes
        self._upload_attempts = upload_attempts
        self._retry_delay = retry_delay
        self._upload_timeout = httpx.Timeout(
            connect=_upload_setting("CONNECT_TIMEOUT_SEC", 15), read=_upload_setting("READ_TIMEOUT_SEC", 120),
            write=_upload_setting("WRITE_TIMEOUT_SEC", 60), pool=15,
        )
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
        check_cancelled()
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

    def interrupted(self, task: TaskClaim) -> dict[str, Any]:
        return self._request("POST", f"/api/tasks/{task.task_id}/interrupted", json={
            "attempt_id": task.attempt_id, "worker_id": task.worker_id or "",
        })

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
                return self._upload_small(f"{root}/upload", path, handle, size)
            return self._upload_chunked(root, path, handle, size)
        finally:
            if isinstance(source, Path):
                handle.close()

    def _retry_post(
        self, endpoint: str, filename: str | None = None, stream: BinaryIO | None = None,
        body: dict[str, object] | None = None, *, timeout: float | httpx.Timeout = 120.0,
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
                _retry_wait(self._retry_delay * 2 ** attempt)
        raise WorkerClientError(f"Request retries exhausted for {endpoint}")

    def _upload_chunked(
        self, root: str, path: str, handle: BinaryIO, size: int, *, extract_to: str | None = None,
    ) -> dict[str, Any]:
        digest = _stream_digest(handle)
        identity = json.dumps([root, path, extract_to, size, digest, self._upload_chunk_bytes])
        endpoint = f"{root}/uploads/{uuid.uuid5(uuid.NAMESPACE_URL, identity)}"
        count = (size + self._upload_chunk_bytes - 1) // self._upload_chunk_bytes
        body: dict[str, object] = {
            "path": path, "part_count": count, "size": size, "sha256": digest,
        }
        if extract_to is not None:
            body["extract_to"] = extract_to
        status = self._upload_status(endpoint)
        progress = UploadProgress(size, count)
        confirmed = 0
        handle.seek(0)
        for index in range(count):
            chunk = handle.read(self._upload_chunk_bytes)
            if status.get("receipt") or _part_matches(status, index, chunk):
                progress.resumed += len(chunk)
            else:
                self._upload_part(endpoint, index, chunk, progress, confirmed)
            confirmed += len(chunk)
            progress.report("uploading", confirmed, index + 1, f"Uploaded {index + 1}/{count} parts")
        progress.report("upload_verifying", confirmed, count, "All bytes received; verifying and publishing artifacts")
        result = self._retry_post(f"{endpoint}/complete", body=body, timeout=self._upload_timeout)
        progress.report("upload_complete", confirmed, count, "Artifact upload verified")
        return result

    def _upload_status(self, endpoint: str, index: int | None = None) -> dict[str, Any]:
        for attempt in range(self._upload_attempts):
            try:
                return self._request("GET", f"{endpoint}/status", timeout=self._upload_timeout,
                                     params={"part": index} if index is not None else {})
            except RetryableWorkerError:
                if attempt + 1 == self._upload_attempts:
                    raise
                _retry_wait(self._retry_delay * 2 ** attempt)
        raise RetryableWorkerError("Upload status retries exhausted")

    def _upload_part(self, endpoint: str, index: int, chunk: bytes, progress: UploadProgress, confirmed: int) -> None:
        for attempt in range(self._upload_attempts):
            progress.report("uploading", confirmed, index, f"Sending part {index + 1}/{progress.parts}; waiting for acknowledgement", attempt + 1)
            try:
                self._request("POST", f"{endpoint}/parts/{index}", timeout=self._upload_timeout,
                              files={"file": ("part", io.BytesIO(chunk), "application/octet-stream")})
                return
            except RetryableWorkerError as exc:
                progress.report("upload_retry", confirmed, index, f"Connection interrupted on part {index + 1}; checking received bytes", attempt + 1)
                if _part_matches(self._upload_status(endpoint, index), index, chunk):
                    return
                if attempt + 1 == self._upload_attempts:
                    raise RetryableWorkerError(f"Part {index + 1}/{progress.parts} failed after {self._upload_attempts} attempts; acknowledged parts are retained") from exc
                logging.getLogger(__name__).warning("Retrying upload part %s after %s", index, type(exc).__name__)
                _retry_wait(self._retry_delay * 2 ** attempt)

    def _upload_small(self, endpoint: str, filename: str, handle: BinaryIO, size: int) -> dict[str, Any]:
        progress = UploadProgress(size, 1)
        progress.report("uploading", 0, 0, "Sending artifact; waiting for acknowledgement")
        result = self._retry_post(endpoint, filename, handle, timeout=self._upload_timeout)
        progress.report("upload_complete", size, 1, "Artifact upload verified")
        return result

    def bundle_index(self, prefix: str, paths: list[str]) -> list[AssetInfo]:
        result = self._request("GET", f"/api/storage/{prefix}/bundle-index",
                               params=[("paths", path) for path in paths])
        return [AssetInfo.model_validate(item) for item in result["objects"]]

    def download_bundle(
        self, prefix: str, paths: list[str], destination: Path, objects: list[AssetInfo] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="room3d-download-", dir=destination.parent) as staging:
            archive = Path(staging) / "bundle.zip"
            selected = [item for item in objects or [] if any(item.path == path or item.path.startswith(path + "/") for path in paths)]
            headers = {"If-Match": asset_revision(selected)} if objects is not None else {}
            self._download(f"/api/storage/{prefix}/bundle.zip", archive, headers=headers,
                           params=[("paths", path) for path in paths], timeout=300.0, on_progress=on_progress)
            with ZipFile(archive) as bundle:
                entries = bundle.infolist()
                if sum(item.file_size for item in entries) > 8_000_000_000 or len(entries) > 20_000:
                    raise WorkerClientError("Downloaded bundle exceeds expanded size or file count limit")
                names = [safe_asset_path(item.filename) for item in entries]
                if len(set(names)) != len(names) or any(stat.S_ISLNK(item.external_attr >> 16) for item in entries):
                    raise WorkerClientError("Downloaded bundle contains duplicate paths or links")
                for item, name in zip(entries, names, strict=True):
                    if item.is_dir():
                        continue
                    target = destination / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(item) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)

    def upload_bundle(self, prefix: str, files: list[tuple[str, Path]]) -> dict[str, Any]:
        if not files:
            return {"objects": []}
        with cast(BinaryIO, tempfile.TemporaryFile(dir=files[0][1].parent)) as stream:
            with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
                for name, path in sorted(files):
                    # Stable ZIP metadata keeps upload identities valid across worker restarts.
                    with path.open("rb") as source, archive.open(safe_asset_path(name), "w", force_zip64=True) as target:
                        while chunk := source.read(1024 * 1024):
                            check_cancelled()
                            target.write(chunk)
            size = stream.tell()
            stream.seek(0)
            if size <= self._upload_chunk_bytes:
                return self._upload_small(f"/api/storage/{prefix}/upload-bundle", "bundle.zip", stream, size)
            project, scope, *rest = prefix.split("/")
            return self._upload_chunked(f"/api/storage/{project}/{scope}", "bundle.zip", stream, size,
                                        extract_to="/".join(rest))

    def upload_key(self, key: str, source: BinaryIO | Path) -> dict[str, Any]:
        project_id, scope, path = split_storage_key(key)
        return self.upload(project_id, scope, path, source)

    def download(self, project_id: str, version_id: str, path: str, destination: Path) -> None:
        self._download(f"/api/storage/{project_id}/{version_id}/{path}", destination)

    def _download(
        self, endpoint: str, destination: Path, *, on_progress: Callable[[int, int], None] | None = None,
        **kwargs: Any,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(self._upload_attempts):
            fd, pending = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
            try:
                with os.fdopen(fd, "wb") as output, self._http.stream("GET", endpoint, **kwargs) as response:
                    if response.status_code == 412:
                        raise AssetRevisionError("Assets changed; refresh the bundle index")
                    if response.status_code in {408, 429} or response.status_code >= 500:
                        raise RetryableWorkerError(f"Storage download temporarily failed with {response.status_code}")
                    if response.status_code >= 400:
                        raise WorkerClientError(f"Storage download failed with {response.status_code}")
                    received, reported = 0, time.monotonic()
                    total = int(response.headers.get("content-length", "0"))
                    for chunk in response.iter_bytes():
                        check_cancelled()
                        output.write(chunk)
                        received += len(chunk)
                        if on_progress and time.monotonic() - reported >= 1:
                            on_progress(received, total)
                            reported = time.monotonic()
                    if on_progress:
                        on_progress(received, total)
                os.replace(pending, destination)
                return
            except (httpx.TransportError, RetryableWorkerError) as exc:
                if attempt + 1 == self._upload_attempts:
                    raise RetryableWorkerError(f"Storage download retries exhausted: {endpoint}") from exc
                time.sleep(self._retry_delay * 2 ** attempt)
            finally:
                Path(pending).unlink(missing_ok=True)

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
