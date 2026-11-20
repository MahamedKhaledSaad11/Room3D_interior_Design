from __future__ import annotations

import hashlib
import io
import json
import uuid
import zipfile
from collections.abc import Iterator
from pathlib import Path

import av
import httpx
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from room3d_worker.client import RetryableWorkerError, WorkerClient, WorkerClientError
from room3d_worker.common.upload_progress import UploadEvent, upload_progress_sink
from room3d_worker.common.asset_cache import DiskAssetCache
from room3d_worker.common.assets import AssetInfo
from room3d_worker.common.manifest import SceneManifest
from room3d_worker.pipeline.s02_poses import BackendOutput, PoseRequest, write_colmap_binary
from room3d_worker.pipeline.s05_track import PromptPoint
from room3d_worker.pipeline.s07_rebuild import GaussianState, RebuildRequest
from room3d_worker.runner import Sam2SessionManager, WorkerBackends, WorkerRunner, _StorageAccessor
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db import Base, get_db
from app.main import app
from app.models import Project, Task


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    engine = create_engine(f"sqlite:///{tmp_path / 'uploads.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine)
    settings = Settings(local_storage_dir=str(tmp_path / "storage"), max_upload_bytes=1,
                        worker_token="dev-worker-token", auth_required=False)
    for module in ("app.routers.artifact_uploads", "app.routers.scenes", "app.routers.bundles", "app.security",
                   "app.routers.projects", "app.routers.edits", "app.services.queue"):
        monkeypatch.setattr(f"{module}.get_settings", lambda: settings)

    def database() -> Iterator[Session]:
        with factory() as session:
            yield session

    with factory() as session:
        session.add(Project(id="upload-room", name="Upload Room"))
        session.commit()
    app.dependency_overrides[get_db] = database
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


@pytest.mark.parametrize("filename", ["scene.spz", "masks.zip", "edited_frames.zip"])
def test_artifacts_survive_chunk_and_commit_retries(api: TestClient, filename: str) -> None:
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr("000000.png", b"mask-or-frame" * 1000)
    payload = content.getvalue()
    failures = {"part": False, "commit": False}

    def bridge(request: httpx.Request) -> httpx.Response:
        response = api.request(request.method, str(request.url), headers=dict(request.headers), content=request.content)
        key = "part" if "/parts/0" in request.url.path else "commit" if request.url.path.endswith("/complete") else None
        if key and not failures[key]:
            failures[key] = True
            raise httpx.ReadError("Tunnel dropped the acknowledgement", request=request)
        return httpx.Response(response.status_code, content=response.content, headers=dict(response.headers))

    with WorkerClient("http://testserver", "dev-worker-token", transport=httpx.MockTransport(bridge),
                      upload_chunk_bytes=2048, retry_delay=0) as worker:
        result = worker.upload("upload-room", "v1", filename, io.BytesIO(payload))
    assert failures == {"part": True, "commit": True}
    assert result["bytes"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    stored = api.get(f"/api/storage/upload-room/v1/{filename}")
    assert stored.content == payload


def test_upload_rejects_incomplete_corrupt_and_unsafe_artifacts(api: TestClient) -> None:
    root = f"/api/storage/upload-room/v1/uploads/{uuid.uuid4()}"
    headers = {"X-Worker-Token": "dev-worker-token"}
    body = {"path": "scene.spz", "part_count": 1, "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    assert api.post(f"{root}/complete", json=body, headers=headers).status_code == 422
    assert api.post(f"{root}/parts/0", files={"file": ("part", b"bad")}, headers=headers).status_code == 200
    assert api.post(f"{root}/complete", json=body, headers=headers).status_code == 422
    assert api.post(f"{root}/complete", json={**body, "path": "../escape"}, headers=headers).status_code == 400
    assert api.post(f"{root}/complete", json=body).status_code == 401
    assert api.get("/api/storage/upload-room/v1/scene.spz").status_code == 404


def _forward(api: TestClient, request: httpx.Request) -> httpx.Response:
    response = api.request(request.method, str(request.url), headers=dict(request.headers), content=request.content)
    return httpx.Response(response.status_code, content=response.content, headers=dict(response.headers))


@pytest.mark.parametrize("corrupt", [False, True])
def test_new_worker_resumes_only_verified_parts(api: TestClient, corrupt: bool) -> None:
    posts: list[str] = []
    fail = True
    endpoint = ""

    def bridge(request: httpx.Request) -> httpx.Response:
        nonlocal endpoint
        if request.method == "POST" and "/parts/" in request.url.path:
            posts.append(request.url.path)
            endpoint = request.url.path.rsplit("/parts/", 1)[0]
            if fail and request.url.path.endswith("/parts/1"):
                raise httpx.WriteTimeout("Connection stalled", request=request)
        return _forward(api, request)

    payload = np.random.default_rng(2).bytes(9000)
    with WorkerClient("http://testserver", "dev-worker-token", transport=httpx.MockTransport(bridge),
                      upload_chunk_bytes=2048, upload_attempts=1, retry_delay=0) as worker:
        with pytest.raises(RetryableWorkerError, match="acknowledged parts are retained"):
            worker.upload("upload-room", "v1", "scene.spz", io.BytesIO(payload))
    headers = {"X-Worker-Token": "dev-worker-token"}
    if corrupt:
        assert api.post(f"{endpoint}/parts/0", files={"file": ("part", b"bad")}, headers=headers).status_code == 200
    status = api.get(f"{endpoint}/status", headers=headers)
    assert status.headers["cache-control"] == "no-store"
    assert set(status.json()["parts"]) == {"0"}
    assert api.get(f"{endpoint}/status").status_code == 401
    fail = False
    events: list[UploadEvent] = []
    with upload_progress_sink(events.append), WorkerClient(
        "http://testserver", "dev-worker-token", transport=httpx.MockTransport(bridge), upload_chunk_bytes=2048, retry_delay=0,
    ) as worker:
        worker.upload("upload-room", "v1", "scene.spz", io.BytesIO(payload))
    assert sum(path.endswith("/parts/0") for path in posts) == (2 if corrupt else 1)
    assert events[-1].telemetry["upload_resumed_bytes"] == (0 if corrupt else 2048)
    assert events[-1].telemetry["upload_confirmed_bytes"] == len(payload)
    assert api.get("/api/storage/upload-room/v1/scene.spz").content == payload


def test_completed_upload_is_not_reused_after_target_was_overwritten(api: TestClient) -> None:
    posts: list[str] = []

    def bridge(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and "/parts/" in request.url.path:
            posts.append(request.url.path)
        return _forward(api, request)

    with WorkerClient("http://testserver", "dev-worker-token", transport=httpx.MockTransport(bridge),
                      upload_chunk_bytes=1024, retry_delay=0) as worker:
        for content in (b"first" * 500, b"second" * 500, b"first" * 500):
            worker.upload("upload-room", "v1", "scene.spz", io.BytesIO(content))
            assert api.get("/api/storage/upload-room/v1/scene.spz").content == content
    assert posts[0] == posts[6], "Returning to old content must restore it, not trust a stale commit receipt"


def test_bundle_resume_is_independent_of_file_order_and_mtime(api: TestClient, tmp_path: Path) -> None:
    import os

    files = [("images/a.png", tmp_path / "a.png"), ("images/b.png", tmp_path / "b.png")]
    for index, (_, path) in enumerate(files):
        path.write_bytes(np.random.default_rng(index).bytes(3000))
    first_parts: list[str] = []
    fail = True

    def bridge(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and "/parts/" in request.url.path:
            if request.url.path.endswith("/parts/0"):
                first_parts.append(request.url.path)
            if fail and request.url.path.endswith("/parts/1"):
                raise httpx.ReadTimeout("Disconnected", request=request)
        return _forward(api, request)

    with WorkerClient("http://testserver", "dev-worker-token", transport=httpx.MockTransport(bridge),
                      upload_chunk_bytes=1024, upload_attempts=1, retry_delay=0) as worker:
        with pytest.raises(RetryableWorkerError):
            worker.upload_bundle("upload-room/v1", files)
    fail = False
    for _, path in files:
        os.utime(path, (1000000000, 1000000000))
    with WorkerClient("http://testserver", "dev-worker-token", transport=httpx.MockTransport(bridge),
                      upload_chunk_bytes=1024, retry_delay=0) as worker:
        worker.upload_bundle("upload-room/v1", list(reversed(files)))
    assert len(first_parts) == 1
    for name, path in files:
        assert api.get(f"/api/storage/upload-room/v1/{name}").content == path.read_bytes()


def test_upload_cancellation_interrupts_retries_and_keeps_partial_bytes(api: TestClient) -> None:
    import threading
    from room3d_worker.common.cancellation import TaskInterruptedError, cancellation_scope

    cancelled = threading.Event()
    posts: list[str] = []

    def bridge(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/parts/1"):
            posts.append(request.url.path)
            cancelled.set()
            raise httpx.WriteTimeout("Stalled", request=request)
        return _forward(api, request)

    with WorkerClient("http://testserver", "dev-worker-token", transport=httpx.MockTransport(bridge),
                      upload_chunk_bytes=1024, retry_delay=10) as worker:
        with cancellation_scope(cancelled), pytest.raises(TaskInterruptedError):
            worker.upload("upload-room", "v1", "scene.spz", io.BytesIO(b"x" * 3000))
    assert len(posts) == 1
    endpoint = posts[0].rsplit("/parts/", 1)[0]
    status = api.get(f"{endpoint}/status", headers={"X-Worker-Token": "dev-worker-token"}).json()
    assert status["parts"]["0"]["size"] == 1024
    assert api.get("/api/storage/upload-room/v1/scene.spz").status_code == 404


def test_upload_uses_separate_connection_write_and_read_limits(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROOM3D_UPLOAD_CONNECT_TIMEOUT_SEC", "7")
    monkeypatch.setenv("ROOM3D_UPLOAD_WRITE_TIMEOUT_SEC", "13")
    monkeypatch.setenv("ROOM3D_UPLOAD_READ_TIMEOUT_SEC", "23")
    timeouts: list[dict[str, float]] = []

    def bridge(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"])
        return _forward(api, request)

    with WorkerClient("http://testserver", "dev-worker-token", transport=httpx.MockTransport(bridge),
                      upload_chunk_bytes=1024, retry_delay=0) as worker:
        worker.upload("upload-room", "v1", "scene.spz", io.BytesIO(b"x" * 3000))
    assert all(timeout["connect"] == 7 and timeout["write"] == 13 and timeout["read"] == 23 for timeout in timeouts)


def test_worker_does_not_retry_permanent_upload_rejections() -> None:
    calls = 0

    def reject(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(403, text="forbidden")

    with WorkerClient("http://testserver", "wrong", transport=httpx.MockTransport(reject), retry_delay=0) as worker:
        with pytest.raises(WorkerClientError, match="403"):
            worker.upload("upload-room", "v1", "scene.spz", io.BytesIO(b"splats"))
    assert calls == 1


@pytest.mark.parametrize("prefix", ["upload-room/v1", "upload-room/edits/edit-a"])
@pytest.mark.parametrize("chunk_size", [2_000_000, 256])
def test_lossless_bundle_roundtrip_cache_reuse_and_invalidation(
    api: TestClient, tmp_path: Path, prefix: str, chunk_size: int,
) -> None:
    requests: list[tuple[str, str]] = []
    lost_commit = False

    def bridge(request: httpx.Request) -> httpx.Response:
        nonlocal lost_commit
        requests.append((request.method, request.url.path))
        response = api.request(request.method, str(request.url), headers=dict(request.headers), content=request.content)
        if request.url.path.endswith("/complete") and not lost_commit:
            lost_commit = True
            raise httpx.ReadError("Lost bundle commit acknowledgement", request=request)
        return httpx.Response(response.status_code, content=response.content, headers=dict(response.headers))

    files: list[tuple[str, Path]] = []
    payloads = {f"images/frame_{i:06d}.jpg": np.random.default_rng(i).bytes(600) for i in range(12)}
    payloads.update({"depths/000007.npy": b"float32-depth-bytes", "manifest.json": b'{"version":"v1"}'})
    for name, payload in payloads.items():
        path = tmp_path / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files.append((name, path))
    cache = DiskAssetCache(tmp_path / "cache")
    with WorkerClient("http://testserver", "dev-worker-token", transport=httpx.MockTransport(bridge),
                      upload_chunk_bytes=chunk_size, retry_delay=0) as worker:
        result = worker.upload_bundle(prefix, files)
        assert len(result["objects"]) == len(payloads)
        objects = worker.bundle_index(prefix, ["images", "depths", "manifest.json"])
        download = lambda paths, destination: worker.download_bundle(prefix, paths, destination)
        cache.sync(prefix, objects, tmp_path / "stage-a", download, ["images", "depths", "manifest.json"])
        downloaded = sum(path.endswith("/bundle.zip") for _, path in requests)
        cache.sync(prefix, worker.bundle_index(prefix, ["images", "depths", "manifest.json"]),
                   tmp_path / "stage-b", download, ["images", "depths", "manifest.json"])
        assert sum(path.endswith("/bundle.zip") for _, path in requests) == downloaded == 1
        for name, payload in payloads.items():
            assert (tmp_path / "stage-a" / name).read_bytes() == (tmp_path / "stage-b" / name).read_bytes() == payload
        changed = files[0][1]
        changed.write_bytes(b"updated" * 90)
        worker.upload_bundle(prefix, [(files[0][0], changed)])
        cache.sync(prefix, worker.bundle_index(prefix, ["images"]), tmp_path / "stage-c", download, ["images"])
        assert (tmp_path / "stage-c" / files[0][0]).read_bytes() == changed.read_bytes()
    assert not any("00007.npy" in path for _, path in requests)


def test_bundle_index_normalizes_legacy_depth_names_without_404_probes(api: TestClient) -> None:
    headers = {"X-Worker-Token": "dev-worker-token"}
    assert api.post("/api/storage/upload-room/v1/upload", headers=headers,
                    files={"file": ("depths/00007.npy", b"legacy depth")}).status_code == 200
    response = api.get("/api/storage/upload-room/v1/bundle.zip?paths=depths", headers=headers)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == ["depths/000007.npy"]
        assert archive.read("depths/000007.npy") == b"legacy depth"
    index = api.get("/api/storage/upload-room/v1/bundle-index?paths=depths", headers=headers).json()
    assert [item["path"] for item in index["objects"]] == ["depths/000007.npy"]


@pytest.mark.parametrize("name", ["../escape.png", "/absolute.png", "masks/../../escape.png", "C:/escape.png", ".uploads/receipt.json"])
def test_bundle_upload_rejects_unsafe_paths_before_writing(api: TestClient, name: str) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("images/valid.jpg", b"must not be published")
        archive.writestr(name, b"bad")
    response = api.post("/api/storage/upload-room/v1/upload-bundle", headers={"X-Worker-Token": "dev-worker-token"},
                        files={"file": ("bundle.zip", stream.getvalue())})
    assert response.status_code == 400
    assert api.get("/api/storage/upload-room/v1/images/valid.jpg").status_code == 404


def test_bundle_permissions_and_empty_optional_directories(api: TestClient) -> None:
    assert api.post("/api/storage/upload-room/v1/upload-bundle", files={"file": ("x.zip", b"bad")}).status_code == 401
    assert api.get("/api/storage/missing/v1/bundle.zip").status_code == 404
    response = api.get("/api/storage/upload-room/v1/bundle.zip?paths=depths")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == []


def test_corrupt_zip_is_rejected_before_any_files_are_published(api: TestClient) -> None:
    stream = io.BytesIO()
    marker = b"payload-that-will-be-corrupted"
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("images/first.png", b"valid file")
        archive.writestr("masks/second.png", marker)
    payload = bytearray(stream.getvalue())
    payload[payload.index(marker)] ^= 1
    response = api.post("/api/storage/upload-room/v1/upload-bundle", headers={"X-Worker-Token": "dev-worker-token"},
                        files={"file": ("bundle.zip", bytes(payload))})
    assert response.status_code == 400
    assert api.get("/api/storage/upload-room/v1/images/first.png").status_code == 404


def test_cache_evicts_oldest_scene_and_bypasses_oversized_assets(tmp_path: Path) -> None:
    import os

    cache = DiskAssetCache(tmp_path / "cache", max_bytes=1000)
    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * 300)
    for version in ("v1", "v2"):
        cache.store(f"p/{version}", [("images/x", source)], [AssetInfo(path="images/x", size=300, etag=version)])
    os.utime(cache.entry("p/v1"), (1, 1))
    assert cache.copy("p/v2/images/x", tmp_path / "keep.bin")
    cache.store("p/v3", [("images/x", source)], [AssetInfo(path="images/x", size=300, etag="v3")])
    assert not cache.entry("p/v1").exists()
    assert cache.entry("p/v2").is_dir() and cache.entry("p/v3").is_dir()
    source.write_bytes(b"y" * 2000)
    cache.store("p/large", [("images/x", source)], [AssetInfo(path="images/x", size=2000, etag="large")])
    assert not cache.entry("p/large").exists()
    assert sum(file.stat().st_size for file in cache.root.rglob("*") if file.is_file()) <= cache.max_bytes


def test_cache_removes_deleted_edit_masks(tmp_path: Path) -> None:
    cache = DiskAssetCache(tmp_path / "cache")
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"mask")
    cache.store("p/edits/e", [("masks/000007.png", mask)], [AssetInfo(path="masks/000007.png", size=4, etag="a")])
    destination = tmp_path / "task"
    assert cache.copy("p/edits/e/masks/000007.png", destination / "masks/000007.png")
    cache.sync("p/edits/e", [], destination, lambda paths, output: pytest.fail("Empty inventory must not download"), ["masks"])
    assert not (destination / "masks/000007.png").exists()
    assert not cache.copy("p/edits/e/masks/000007.png", tmp_path / "stale.png")


def test_900_assets_use_one_zip_and_reuse_cache_across_stages(api: TestClient, tmp_path: Path) -> None:
    requests: list[tuple[str, str, int]] = []

    def bridge(request: httpx.Request) -> httpx.Response:
        response = api.request(request.method, str(request.url), headers=dict(request.headers), content=request.content)
        requests.append((request.method, request.url.path, response.status_code))
        return httpx.Response(response.status_code, content=response.content, headers=dict(response.headers))

    folders = ["images", "depths", "masks", "edited_frames", "thumbnails", "seed_previews"]
    files: list[tuple[str, Path]] = []
    for folder in folders:
        for index in range(150):
            name = f"{folder}/{index:06d}.{'npy' if folder == 'depths' else 'png'}"
            path = tmp_path / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(hashlib.sha256(name.encode()).digest())
            files.append((name, path))
    with WorkerClient("http://testserver", "dev-worker-token", transport=httpx.MockTransport(bridge), retry_delay=0) as client:
        client.upload_bundle("upload-room/v1", files)
        storage = _StorageAccessor(client, None, tmp_path / "cache")
        storage.prepare("upload-room/v1", folders, tmp_path / "track")
        storage.prepare("upload-room/v1", folders, tmp_path / "apply")
    assert len(requests) == 4
    assert sum(path.endswith("/bundle.zip") for _, path, _ in requests) == 1
    assert all(status == 200 for _, _, status in requests)
    for name, source in files:
        assert (tmp_path / "track" / name).read_bytes() == (tmp_path / "apply" / name).read_bytes() == source.read_bytes()


def test_bundle_revision_fence_retries_with_fresh_masks(api: TestClient, tmp_path: Path) -> None:
    headers = {"X-Worker-Token": "dev-worker-token"}
    endpoint = "/api/storage/upload-room/edits/upload"
    key = "e/masks/000007.png"
    assert api.post(endpoint, headers=headers, files={"file": (key, b"old")}).status_code == 200
    statuses: list[int] = []
    changed = False

    def bridge(request: httpx.Request) -> httpx.Response:
        nonlocal changed
        if request.url.path.endswith("bundle.zip") and not changed:
            changed = True
            assert api.post(endpoint, headers=headers, files={"file": (key, b"new")}).status_code == 200
        response = api.request(request.method, str(request.url), headers=dict(request.headers), content=request.content)
        statuses.append(response.status_code)
        return httpx.Response(response.status_code, content=response.content, headers=dict(response.headers))

    with WorkerClient("http://testserver", "dev-worker-token", transport=httpx.MockTransport(bridge)) as worker:
        storage = _StorageAccessor(worker, None, tmp_path / "cache")
        storage.prepare("upload-room/edits/e", ["masks"], tmp_path / "work")
    assert statuses == [200, 412, 200, 200]
    assert (tmp_path / "work/masks/000007.png").read_bytes() == b"new"


def test_oversized_bundle_is_materialized_without_filling_cache(tmp_path: Path) -> None:
    cache = DiskAssetCache(tmp_path / "cache", max_bytes=512)
    content = b"x" * 2000

    def download(paths: list[str], target: Path) -> None:
        (target / "images").mkdir(parents=True)
        (target / "images/000007.png").write_bytes(content)

    cache.sync("p/v", [AssetInfo(path="images/000007.png", size=len(content), etag="a")],
               tmp_path / "work", download, ["images"])
    assert (tmp_path / "work/images/000007.png").read_bytes() == content
    assert sum(file.stat().st_size for file in cache.root.rglob("*") if file.is_file()) <= 512


def test_slow_bundle_does_not_hold_global_cache_lock(tmp_path: Path) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    cache = DiskAssetCache(tmp_path / "cache")
    source = tmp_path / "source.bin"
    source.write_bytes(b"fast")
    cache.store("p/warm", [("images/a", source)], [AssetInfo(path="images/a", size=4, etag="a")])
    started, release = threading.Event(), threading.Event()

    def download(paths: list[str], target: Path) -> None:
        started.set()
        if not release.wait(5):
            raise TimeoutError("Test bundle was never released")
        (target / "images").mkdir()
        (target / "images/a").write_bytes(b"slow")

    with ThreadPoolExecutor(max_workers=2) as pool:
        slow = pool.submit(cache.sync, "p/slow", [AssetInfo(path="images/a", size=4, etag="b")],
                           tmp_path / "slow", download, ["images"])
        assert started.wait(3)
        try:
            warm = pool.submit(cache.copy, "p/warm/images/a", tmp_path / "warm.bin")
            assert warm.result(timeout=1)
        finally:
            release.set()
        slow.result(timeout=3)
    assert (tmp_path / "warm.bin").read_bytes() == b"fast"


def test_mutable_single_file_fetch_does_not_trust_previous_task_cache(api: TestClient, tmp_path: Path) -> None:
    headers = {"X-Worker-Token": "dev-worker-token"}
    endpoint = "/api/storage/upload-room/edits/upload"
    key = "e/masks/000007.png"
    assert api.post(endpoint, headers=headers, files={"file": (key, b"old")}).status_code == 200

    def bridge(request: httpx.Request) -> httpx.Response:
        response = api.request(request.method, str(request.url), headers=dict(request.headers), content=request.content)
        return httpx.Response(response.status_code, content=response.content, headers=dict(response.headers))

    with WorkerClient("http://testserver", "dev-worker-token", transport=httpx.MockTransport(bridge)) as worker:
        storage = _StorageAccessor(worker, None, tmp_path / "cache")
        storage.prepare("upload-room/edits/e", ["masks"], tmp_path / "first")
        assert api.post(endpoint, headers=headers, files={"file": (key, b"new")}).status_code == 200
        storage.validated.clear()
        output = storage.fetch("upload-room/edits/" + key, tmp_path / "latest.png")
        assert output.read_bytes() == b"new"


def test_live_rebuild_progress_is_monotonic(api: TestClient, tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'uploads.db'}")
    with Session(engine) as session:
        session.add(Task(project_id="upload-room", task_type="rebuild", payload_json="{}"))
        session.commit()
    headers = {"X-Worker-Token": "dev-worker-token"}
    claim = api.post("/api/tasks/claim", headers=headers, json={"worker_id": "worker", "capabilities": ["rebuild"]}).json()
    for percent, expected in [(10, 10), (60, 60), (20, 60), (95, 95)]:
        response = api.post(f"/api/tasks/{claim['task_id']}/progress", headers=headers, json={
            "attempt_id": claim["attempt_id"], "stage": "07_rebuild", "progress_pct": percent,
            "message": f"Rebuilding 3DGS step {percent}/100",
        })
        assert response.status_code == 200
        task = next(item for item in api.get("/api/projects/upload-room").json()["tasks"] if item["id"] == claim["task_id"])
        assert task["progress_pct"] == expected
    engine.dispose()


@pytest.mark.parametrize("status", [408, 429, 502, 503, 504, 524])
def test_small_upload_rewinds_body_after_transient_failure(status: int) -> None:
    bodies: list[bytes] = []

    def transient(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        return httpx.Response(status if len(bodies) == 1 else 200, json={"bytes": 6})

    with WorkerClient("http://testserver", "secret", transport=httpx.MockTransport(transient), retry_delay=0) as worker:
        assert worker.upload("room", "v1", "scene.spz", io.BytesIO(b"splats"))["bytes"] == 6
    assert len(bodies) == 2
    assert all(b"splats" in body for body in bodies)


class _Pose:
    def estimate(self, request: PoseRequest) -> BackendOutput:
        points = np.random.default_rng(0).uniform(-0.5, 0.5, (120, 3))
        points[:, 2] += 4
        sparse = write_colmap_binary(request.output_dir / "raw", request.width, request.height,
                                     request.intrinsics, request.frame_records, points)
        count = len(request.frame_records)
        cameras = np.column_stack((np.linspace(-1, 1, count), np.zeros((count, 2))))
        return BackendOutput(count, 0.1, len(points), points, cameras, request.frame_records, sparse)


class _Sam:
    def encode_image(self, image_path: Path) -> None:
        return None

    def predict(self, embedding: object, points: list[PromptPoint]) -> NDArray[np.uint8]:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        return mask


class _VideoSam:
    def propagate(self, image_paths: dict[int, Path], ref_frame_idx: int,
                  seed_mask: NDArray[np.uint8], inference_state_device: str) -> dict[int, NDArray[np.uint8]]:
        return {index: seed_mask.copy() for index in image_paths}


class _Inpaint:
    def inpaint(self, image: NDArray[np.uint8], mask: NDArray[np.uint8], prompt: str,
                seed: int, image_reference: NDArray[np.uint8] | None = None) -> NDArray[np.uint8]:
        edited = image.copy()
        edited[mask > 0] = [210, seed % 255, 70]
        return edited


def _rebuild(request: RebuildRequest, manifest: SceneManifest, base: GaussianState,
             roi: NDArray[np.bool_]) -> GaussianState:
    colors = base.colors.copy()
    colors[roi] = [210, 40, 70]
    return GaussianState(base.points_xyz.copy(), colors, base.opacity.copy(), base.scales.copy(), base.quats)


def test_five_task_flow_over_http_uploads_real_artifacts(api: TestClient, tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    with av.open(str(video_path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            image = np.random.default_rng(index).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    engine = create_engine(f"sqlite:///{tmp_path / 'uploads.db'}")
    with Session(engine) as session:
        session.add(Task(project_id="upload-room", task_type="reconstruct", payload_json=json.dumps({
            "project_id": "upload-room", "version_id": "v1", "source_video_key": "upload-room/input/video.mp4",
            "max_frames": 4, "min_parallax_frac": 0, "iterations": 1,
        })))
        session.commit()
    progress: list[dict[str, object]] = []

    def bridge(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/progress"):
            progress.append(json.loads(request.content))
        response = api.request(request.method, str(request.url), headers=dict(request.headers), content=request.content)
        return httpx.Response(response.status_code, content=response.content, headers=dict(response.headers))

    with WorkerClient("http://testserver", "dev-worker-token", transport=httpx.MockTransport(bridge),
                      upload_chunk_bytes=2048, retry_delay=0) as worker:
        worker.upload_key("upload-room/input/video.mp4", video_path)
        runner = WorkerRunner(worker, work_root=tmp_path / "work", capabilities=[
            "reconstruct", "segment_init", "segment_refine", "segment_track", "diffusion_preview",
            "diffusion_propagate", "rebuild",
        ], backends=WorkerBackends(
            pose_backends={"colmap_global": _Pose()}, sam_manager=Sam2SessionManager(backend=_Sam()),
            video_sam=_VideoSam(), inpaint_backend=_Inpaint(), rebuild_backend=_rebuild,
        ))
        assert runner.run_once()
        assert api.get("/api/projects/upload-room").json()["active_version_id"] == "v1"
        _exercise_edit_flow(api, runner)
    assert {event["stage"] for event in progress} >= {"01_extract", "05_track", "06_diffusion", "07_rebuild"}
    assert any("seed preview" in str(event["message"]) for event in progress)
    assert any("Uploaded" in str(event["message"]) for event in progress)
    engine.dispose()


def _exercise_edit_flow(api: TestClient, runner: WorkerRunner) -> None:
    from app.routers.edits import get_settings
    get_settings().max_upload_bytes = 16 * 1024 * 1024
    response = api.post("/api/projects/upload-room/edits/resolve-click", json={
        "frame_idx": 0, "u_norm": 0.5, "v_norm": 0.5, "base_version_id": "v1",
    })
    assert response.status_code == 201, response.text
    edit_id = response.json()["edit_id"]
    assert runner.run_once()
    assert api.post(f"/api/edits/{edit_id}/track").status_code == 200
    assert runner.run_once()
    edit = api.get(f"/api/edits/{edit_id}").json()
    for idx in edit["masks_summary"]:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        buffer = io.BytesIO()
        Image.fromarray(mask).save(buffer, format="PNG")
        assert api.put(f"/api/edits/{edit_id}/masks/{idx}",
                       files={"file": ("mask.png", buffer.getvalue(), "image/png")}).status_code == 200
    assert api.post(f"/api/edits/{edit_id}/preview-seeds", json={
        "prompt": "red chair", "seeds": [101, 202, 303, 404],
    }).status_code == 200
    assert runner.run_once()
    edit = api.get(f"/api/edits/{edit_id}").json()
    assert len(edit["preview_seeds"]) == 4
    for key in edit["preview_seeds"].values():
        preview = api.get(f"/api/storage/{key}")
        assert preview.content.startswith(b"\x89PNG")
        assert preview.headers["content-type"] == "image/png"
    applied = api.post(f"/api/edits/{edit_id}/apply", json={"chosen_seed_id": "202"})
    assert applied.status_code == 200, applied.text
    assert runner.run_once()
    assert runner.run_once()
    target = applied.json()["target_version_id"]
    assert api.get("/api/projects/upload-room").json()["active_version_id"] == "v1"
    assert api.get(f"/api/storage/upload-room/{target}/scene.spz").content.startswith(b"\x1f\x8b")
    rebuilt = api.get(f"/api/projects/upload-room/versions/{target}/manifest").json()
    assert rebuilt["parent_version_id"] == "v1"
    assert api.get(f"/api/edits/{edit_id}").json()["status"] == "validation_pending"
