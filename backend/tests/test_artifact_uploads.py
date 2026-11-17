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
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db import Base, get_db
from app.main import app
from app.models import Project, Task
from room3d_worker.client import WorkerClient, WorkerClientError
from room3d_worker.common.manifest import SceneManifest
from room3d_worker.pipeline.s02_poses import BackendOutput, PoseRequest, write_colmap_binary
from room3d_worker.pipeline.s05_track import PromptPoint
from room3d_worker.pipeline.s07_rebuild import GaussianState, RebuildRequest
from room3d_worker.runner import Sam2SessionManager, WorkerBackends, WorkerRunner


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    engine = create_engine(f"sqlite:///{tmp_path / 'uploads.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine)
    settings = Settings(local_storage_dir=str(tmp_path / "storage"), max_upload_bytes=1)
    for module in ("app.routers.artifact_uploads", "app.routers.scenes", "app.security",
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
    assert any("Uploaded artifact" in str(event["message"]) for event in progress)
    engine.dispose()


def _exercise_edit_flow(api: TestClient, runner: WorkerRunner) -> None:
    response = api.post("/api/projects/upload-room/edits/resolve-click", json={
        "frame_idx": 0, "u_norm": 0.5, "v_norm": 0.5, "base_version_id": "v1",
    })
    assert response.status_code == 201, response.text
    edit_id = response.json()["edit_id"]
    assert runner.run_once()
    assert api.post(f"/api/edits/{edit_id}/track").status_code == 200
    assert runner.run_once()
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
    assert api.get("/api/projects/upload-room").json()["active_version_id"] == target
    assert api.get(f"/api/storage/upload-room/{target}/scene.spz").content.startswith(b"\x1f\x8b")
    rebuilt = api.get(f"/api/projects/upload-room/versions/{target}/manifest").json()
    assert rebuilt["parent_version_id"] == "v1"
    assert api.get(f"/api/edits/{edit_id}").json()["status"] == "completed"
