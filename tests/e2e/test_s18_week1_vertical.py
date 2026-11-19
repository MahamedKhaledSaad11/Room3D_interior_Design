from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

REPO_ROOT = Path(__file__).resolve().parents[2]
for subdir in ("", "backend", "worker"):
    path_str = str(REPO_ROOT / subdir) if subdir else str(REPO_ROOT)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from app.config import Settings  # noqa: E402
from app.db import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402
from room3d_worker.client import WorkerClient  # noqa: E402
from room3d_worker.common.manifest import load_manifest  # noqa: E402
from room3d_worker.pipeline.s01_extract import extract_keyframes  # noqa: E402
from room3d_worker.pipeline.s02_poses import BackendOutput, PoseRequest, estimate_poses  # noqa: E402
from room3d_worker.pipeline.s03_train import build_train_config, initialize_scales_knn, radial_outlier_mask  # noqa: E402
from room3d_worker.pipeline.s04_export import ExportRequest, export_scene  # noqa: E402
from room3d_worker.runner import WorkerBackends, WorkerRunner  # noqa: E402


def _make_synthetic_video(path: Path, frames: int = 15) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = "yuv420p"
        for idx in range(frames):
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            shift = idx * 14
            cv2.rectangle(img, (20 + shift, 30), (110 + shift, 160), (240, 140, 30), -1)
            cv2.circle(img, (60 + shift, 190), 22, (40, 230, 180), -1)
            cv2.line(img, (0, 230 - idx * 10), (319, idx * 10), (200, 200, 255), 3)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@dataclass
class _SyntheticPoseBackend:
    def estimate(self, request: PoseRequest) -> BackendOutput:
        sparse_dir = request.output_dir / "_raw_sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("cameras.bin", "images.bin", "points3D.bin"):
            (sparse_dir / filename).write_bytes(f"synthetic-{filename}".encode("utf-8"))
        n = len(request.frame_records)
        rng = np.random.default_rng(42)
        points = rng.uniform(-1.5, 1.5, size=(120, 3))
        cameras = np.column_stack((np.linspace(-1.0, 1.0, n), np.zeros(n), np.full(n, 2.5)))
        return BackendOutput(n, 0.35, n, points, cameras, list(request.frame_records), sparse_dir)


def _run_worker_week1(worker: WorkerClient, project_id: str, work_dir: Path, expected_task_id: str) -> str:
    claimed = worker.claim("e2e-worker", ["reconstruct_week1"], interactive_only=False)
    assert claimed is not None
    assert claimed.task_id == expected_task_id

    version_id = f"{project_id}_v0"
    _, storage_version, storage_path = claimed.payload["input_key"].split("/", 2)
    downloaded_video = work_dir / "input.mp4"
    worker.download(project_id, storage_version, storage_path, downloaded_video)

    worker.heartbeat(claimed)
    worker.progress(claimed, "01_extract", 50.0, "Extracting keyframes")
    extracted = extract_keyframes(downloaded_video, work_dir / "frames", max_long_side=640, min_parallax_frac=0.02)
    assert len(extracted.index["frames"]) >= 5

    worker.progress(claimed, "02_poses", 50.0, "Solving poses")
    frame_records = [
        {
            "frame_idx": f["frame_idx"],
            "orig_video_frame": f["orig_video_frame"],
            "timestamp_sec": f["timestamp_sec"],
            "image_relpath": f["image_relpath"],
            "colmap_image_id": f["frame_idx"] + 1,
            "cam_from_world": np.eye(4).tolist(),
        }
        for f in extracted.index["frames"]
    ]
    pose_req = PoseRequest(
        image_dir=work_dir / "frames",
        output_dir=work_dir / "scene",
        project_id=project_id,
        version_id=version_id,
        total_frames=len(frame_records),
        width=320,
        height=240,
        intrinsics={"fx": 280.0, "fy": 280.0, "cx": 160.0, "cy": 120.0},
        backend="colmap_global",
        frame_records=frame_records,  # type: ignore[arg-type]
    )
    pose_manifest, _ = estimate_poses(pose_req, {"colmap_global": _SyntheticPoseBackend()})

    worker.progress(claimed, "03_train", 60.0, "Optimizing Gaussians")
    train_cfg = build_train_config(work_dir / "scene", allow_synthetic_fallback=True) if "allow_synthetic_fallback" in __import__("inspect").signature(build_train_config).parameters else build_train_config(work_dir / "scene")
    assert train_cfg.test_every == 0
    rng = np.random.default_rng(7)
    pts = rng.uniform(-1.0, 1.0, size=(200, 3)).astype(np.float32)
    pts = pts[radial_outlier_mask(pts)]
    scales = initialize_scales_knn(pts).astype(np.float32)
    scale_vec = np.repeat(scales[:, None], 3, axis=1)
    opacity = np.full(len(pts), 0.9, dtype=np.float32)

    worker.progress(claimed, "04_export", 80.0, "Exporting bundles")
    export_scene(
        ExportRequest(
            output_dir=work_dir / "scene",
            project_id=project_id,
            version_id=version_id,
            frames=[f.model_dump() for f in pose_manifest.frames],
            width=pose_manifest.width,
            height=pose_manifest.height,
            intrinsics=pose_manifest.intrinsics.model_dump(),
            T_viewer_from_colmap=pose_manifest.T_viewer_from_colmap,
            pose_backend=pose_manifest.pose_backend,
        ),
        pts,
        opacity,
        scale_vec,
    )
    assert load_manifest(work_dir / "scene" / "manifest.json").project_id == project_id

    for rel in ("manifest.json", "scene.spz", "scene.ply", "sparse_points.json"):
        worker.upload(project_id, version_id, rel, work_dir / "scene" / rel)
    for frame in pose_manifest.frames:
        worker.upload(project_id, version_id, frame.image_relpath, work_dir / "frames" / frame.image_relpath)

    worker.complete(
        claimed,
        result_manifest_key=f"{project_id}/{version_id}/manifest.json",
        result_payload={"version_id": version_id, "label": "Base v0"},
    )
    return version_id


def test_week1_vertical_pipeline_backend_to_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "e2e.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, poolclass=NullPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    settings = Settings(
        database_url=f"sqlite:///{db_path}",
        local_storage_dir=str(tmp_path / "storage"),
        max_upload_bytes=16 * 1024 * 1024,
        auth_required=False,
    )
    for module in ("app.routers.projects", "app.routers.scenes", "app.security", "app.routers.bundles", "app.routers.artifact_uploads", "app.services.queue"):
        monkeypatch.setattr(f"{module}.get_settings", lambda: settings)

    def override_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as client:
            project = client.post("/api/projects", json={"name": "E2E Living Room"}).json()
            project_id = project["id"]
            video_path = tmp_path / "room_15f.mp4"
            _make_synthetic_video(video_path, frames=15)
            upload_res = client.post(
                f"/api/projects/{project_id}/upload",
                files={"file": ("room_15f.mp4", video_path.read_bytes(), "video/mp4")},
            )
            assert upload_res.status_code == 200
            task_id = upload_res.json()["task_id"]

            def _asgi_bridge(request: httpx.Request) -> httpx.Response:
                resp = client.request(
                    method=request.method,
                    url=str(request.url.raw_path.decode("ascii")),
                    headers=dict(request.headers),
                    content=request.content,
                )
                return httpx.Response(resp.status_code, headers=dict(resp.headers), content=resp.content)

            worker = WorkerClient("http://testserver", settings.worker_token, transport=httpx.MockTransport(_asgi_bridge))
            version_id = _run_worker_week1(worker, project_id, tmp_path / "work", task_id)

            detail = client.get(f"/api/projects/{project_id}").json()
            assert detail["active_version_id"] == version_id
            assert detail["tasks"][0]["status"] == "completed"
            assert detail["tasks"][0]["progress_pct"] == 100.0
            assert detail["tasks"][0]["stage"] == "04_export"

            fetched_manifest = client.get(f"/api/projects/{project_id}/versions/{version_id}/manifest")
            assert fetched_manifest.status_code == 200
            assert fetched_manifest.json()["artifacts"]["spz_relpath"] == "scene.spz"
            assert len(json.loads(client.get(f"/api/storage/{project_id}/{version_id}/sparse_points.json").content)) > 0
            assert len(client.get(f"/api/storage/{project_id}/{version_id}/scene.spz").content) > 0
    finally:
        app.dependency_overrides.clear()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_week1_live_browser_renders_reconstructed_room(tmp_path: Path) -> None:
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if not npx:
        pytest.skip("npx not available")
    if not (REPO_ROOT / "frontend" / "node_modules" / "@playwright" / "test").exists():
        pytest.skip("@playwright/test is not installed under frontend/node_modules")

    backend_port = _free_port()
    frontend_port = _free_port()
    db_path = tmp_path / "live.db"
    backend_env = {
        **os.environ,
        "PYTHONPATH": f"{REPO_ROOT / 'backend'}{os.pathsep}{REPO_ROOT / 'worker'}",
        "ROOM3D_DATABASE_URL": f"sqlite:///{db_path}",
        "ROOM3D_LOCAL_STORAGE_DIR": str(tmp_path / "live_storage"),
        "ROOM3D_AUTH_REQUIRED": "false",
        "DATABASE_URL": f"sqlite:///{db_path}",
        "LOCAL_STORAGE_DIR": str(tmp_path / "live_storage"),
        "AUTH_REQUIRED": "false",
        "WORKER_TOKEN": "dev-worker-token",
    }
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(backend_port)],
        cwd=str(REPO_ROOT / "backend"),
        env=backend_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        base_url = f"http://127.0.0.1:{backend_port}"
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{base_url}/api/projects", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.25)
        else:
            raise AssertionError("Live FastAPI backend failed to start")

        project = httpx.post(f"{base_url}/api/projects", json={"name": "E2E Live Room"}, timeout=5.0).json()
        project_id = project["id"]
        video_path = tmp_path / "live_15f.mp4"
        _make_synthetic_video(video_path, frames=15)
        with video_path.open("rb") as stream:
            upload_res = httpx.post(
                f"{base_url}/api/projects/{project_id}/upload",
                files={"file": ("live_15f.mp4", stream, "video/mp4")},
                timeout=10.0,
            )
        upload_res.raise_for_status()
        task_id = upload_res.json()["task_id"]

        worker = WorkerClient(base_url, "dev-worker-token")
        _run_worker_week1(worker, project_id, tmp_path / "live_work", task_id)
        worker.close()

        pw_env = {
            **os.environ,
            "ROOM3D_BACKEND_URL": base_url,
            "ROOM3D_FRONTEND_PORT": str(frontend_port),
            "ROOM3D_LIVE_PROJECT_PATH": f"/projects/{project_id}",
        }
        result = subprocess.run(
            [npx, "playwright", "test", "tests/e2e_live.spec.ts"],
            cwd=str(REPO_ROOT / "frontend"),
            env=pw_env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Playwright live E2E failed:\n{result.stdout}\n{result.stderr}"
    finally:
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_proc.kill()


def test_week1_worker_runner_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "runner_w1.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, poolclass=NullPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    settings = Settings(
        database_url=f"sqlite:///{db_path}",
        local_storage_dir=str(tmp_path / "storage"),
        max_upload_bytes=16 * 1024 * 1024,
        auth_required=False,
    )
    for module in ("app.routers.projects", "app.routers.scenes", "app.security", "app.routers.bundles", "app.routers.artifact_uploads", "app.services.queue"):
        monkeypatch.setattr(f"{module}.get_settings", lambda: settings)

    def override_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as client:
            project = client.post("/api/projects", json={"name": "Runner W1"}).json()
            project_id = project["id"]
            video_path = tmp_path / "room_15f.mp4"
            _make_synthetic_video(video_path, frames=15)
            up = client.post(
                f"/api/projects/{project_id}/upload",
                files={"file": ("room_15f.mp4", video_path.read_bytes(), "video/mp4")},
            )
            assert up.status_code == 200
            version_id = up.json()["version_id"]

            def _asgi_bridge(request: httpx.Request) -> httpx.Response:
                resp = client.request(
                    method=request.method,
                    url=str(request.url.raw_path.decode("ascii")),
                    headers=dict(request.headers),
                    content=request.content,
                )
                return httpx.Response(resp.status_code, headers=dict(resp.headers), content=resp.content)

            worker = WorkerClient("http://testserver", settings.worker_token, transport=httpx.MockTransport(_asgi_bridge))
            runner = WorkerRunner(
                worker,
                worker_id="runner-w1",
                work_root=tmp_path / "work",
                capabilities=["reconstruct_week1"],
                backends=WorkerBackends(pose_backends={"colmap_global": _SyntheticPoseBackend()}),
            )
            assert runner.run_once() is True

            detail = client.get(f"/api/projects/{project_id}").json()
            assert detail["active_version_id"] == version_id
            assert detail["tasks"][0]["status"] == "completed"
            assert detail["tasks"][0]["progress_pct"] == 100.0
            assert len(client.get(f"/api/storage/{project_id}/{version_id}/scene.spz").content) > 0
    finally:
        app.dependency_overrides.clear()


def test_import_kaggle_bundle_zip_serves_project_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zipfile

    from scripts.import_kaggle_bundle import import_kaggle_bundle

    bundle_src = tmp_path / "bundle_src" / "v1"
    (bundle_src / "images").mkdir(parents=True, exist_ok=True)
    (bundle_src / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    manifest_obj = {
        "schema_version": 1,
        "project_id": "playroom-kaggle",
        "version_id": "v1",
        "parent_version_id": None,
        "pose_backend": "precomputed_colmap",
        "width": 960,
        "height": 624,
        "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 480.0, "cy": 312.0},
        "T_viewer_from_colmap": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "artifacts": {"spz_relpath": "scene.spz", "ply_relpath": "scene.ply"},
        "frames": [
            {
                "frame_idx": 0,
                "orig_video_frame": 0,
                "timestamp_sec": 0.0,
                "image_relpath": "images/frame_000000.jpg",
                "colmap_image_id": 1,
                "cam_from_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "excluded": False,
            }
        ],
    }
    (bundle_src / "manifest.json").write_text(json.dumps(manifest_obj), encoding="utf-8")
    (bundle_src / "scene.spz").write_bytes(b"SPZ-KAGGLE-DATA")
    (bundle_src / "scene.ply").write_bytes(b"ply\n")
    (bundle_src / "sparse_points.json").write_text("[[0.0, 0.0, 2.0]]", encoding="utf-8")
    (bundle_src / "images" / "frame_000000.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (bundle_src / "sparse" / "0" / name).write_bytes(b"bin")

    zip_path = tmp_path / "playroom_room3d_bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file_path in (tmp_path / "bundle_src").rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(tmp_path / "bundle_src"))

    db_path = tmp_path / "imported.db"
    storage_dir = tmp_path / "imported_storage"
    db_url = f"sqlite:///{db_path.as_posix()}"
    result = import_kaggle_bundle(
        zip_path,
        project_id="playroom-demo",
        project_name="Playroom Demo",
        storage_dir=storage_dir,
        db_url=db_url,
    )
    assert result["project_id"] == "playroom-demo"
    assert result["active_version_id"] == "v1"

    engine = create_engine(db_url, connect_args={"check_same_thread": False}, poolclass=NullPool)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    settings = Settings(database_url=db_url, local_storage_dir=str(storage_dir), auth_required=False)
    for module in ("app.routers.projects", "app.routers.scenes", "app.security", "app.routers.bundles", "app.routers.artifact_uploads", "app.services.queue"):
        monkeypatch.setattr(f"{module}.get_settings", lambda: settings)

    def override_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as client:
            proj_res = client.get("/api/projects/playroom-demo")
            assert proj_res.status_code == 200
            assert proj_res.json()["active_version_id"] == "v1"
            manifest_res = client.get("/api/projects/playroom-demo/versions/v1/manifest")
            assert manifest_res.status_code == 200
            assert manifest_res.json()["project_id"] == "playroom-demo"
            assert client.get("/api/storage/playroom-demo/v1/scene.spz").content == b"SPZ-KAGGLE-DATA"
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_stage_cli_precomputed_colmap_and_diffusion_config_wiring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from room3d_worker import stage_cli
    from room3d_worker.pipeline import s02_poses

    called: dict[str, Any] = {}

    def _fake_prepare(
        source_scene_dir: Path,
        output_dir: Path,
        *,
        project_id: str,
        version_id: str,
        max_long_side: int = 960,
        manifest_path: Path | None = None,
    ) -> tuple[Any, Any]:
        called.update(
            {
                "source_scene_dir": source_scene_dir,
                "output_dir": output_dir,
                "project_id": project_id,
                "version_id": version_id,
                "max_long_side": max_long_side,
                "manifest_path": manifest_path,
            }
        )
        return (
            SimpleNamespace(version_id=version_id),
            SimpleNamespace(registered_frames=12),
        )

    monkeypatch.setattr(s02_poses, "prepare_precomputed_colmap_scene", _fake_prepare, raising=False)
    out = stage_cli.run_poses_stage(
        {
            "pose_backend": "precomputed_colmap",
            "source_scene_dir": str(tmp_path / "src_colmap"),
            "output_dir": str(tmp_path / "out_scene"),
            "project_id": "proj-db",
            "version_id": "v1",
            "max_long_side": 960,
        }
    )
    assert out["registered_frames"] == 12
    assert out["version_id"] == "v1"
    assert called["source_scene_dir"] == tmp_path / "src_colmap"
    assert called["max_long_side"] == 960

    cfg = stage_cli._diffusion_cfg(
        {"model_variant": "flux2_klein_4b", "quantization": "nf4", "offload_text_encoder": True}
    )
    if hasattr(cfg, "quantization"):
        assert getattr(cfg, "quantization") == "nf4"
    if hasattr(cfg, "offload_text_encoder"):
        assert getattr(cfg, "offload_text_encoder") is True


def test_import_kaggle_bundle_validation_collision_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.models import Project, SceneVersion
    from scripts.import_kaggle_bundle import import_kaggle_bundle

    bundle_dir = tmp_path / "dir_bundle"
    v1_dir = bundle_dir / "v1"
    (v1_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (v1_dir / "meshes").mkdir(parents=True, exist_ok=True)
    manifest_obj = {
        "schema_version": 1,
        "project_id": "drjohnson-kaggle",
        "version_id": "v1",
        "pose_backend": "precomputed_colmap",
        "width": 960,
        "height": 624,
        "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 480.0, "cy": 312.0},
        "T_viewer_from_colmap": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "artifacts": {
            "spz_relpath": "scene.spz",
            "gs_ckpt_relpath": "gs_ckpt.pt",
            "collision_glb_relpath": "meshes/collision.glb",
        },
        "frames": [],
    }
    (v1_dir / "manifest.json").write_text(json.dumps(manifest_obj), encoding="utf-8")
    (v1_dir / "scene.spz").write_bytes(b"SPZ-DRJ")
    (v1_dir / "gs_ckpt.pt").write_bytes(b"CKPT-BYTES")
    (v1_dir / "meshes" / "collision.glb").write_bytes(b"glTF")

    db_path = tmp_path / "collision.db"
    storage_dir = tmp_path / "collision_storage"
    db_url = f"sqlite:///{db_path.as_posix()}"

    # 1) Path traversal project_id rejected before touching storage or DB
    with pytest.raises(ValueError, match="Invalid project_id"):
        import_kaggle_bundle(bundle_dir, project_id="../../outside", storage_dir=storage_dir, db_url=db_url)
    assert not (tmp_path / "outside").exists()

    # 2) Pre-seed another project holding primary-key SceneVersion.id == "v1"
    engine = create_engine(db_url, connect_args={"check_same_thread": False}, poolclass=NullPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    with session_factory() as session:
        session.add(Project(id="other-proj", name="Other", status="completed", active_version_id="v1"))
        session.add(SceneVersion(id="v1", project_id="other-proj", label="V1"))
        session.commit()

    # Importing into a second project allocates a <=36-char UUID version_id and copies gs_ckpt.pt + manifest artifacts
    result = import_kaggle_bundle(bundle_dir, project_id="drj-room", storage_dir=storage_dir, db_url=db_url)
    imported_vid = result["active_version_id"]
    assert imported_vid != "v1" and len(imported_vid) <= 36
    assert (storage_dir / "drj-room" / imported_vid / "gs_ckpt.pt").read_bytes() == b"CKPT-BYTES"
    assert (storage_dir / "drj-room" / imported_vid / "meshes" / "collision.glb").read_bytes() == b"glTF"

    settings = Settings(database_url=db_url, local_storage_dir=str(storage_dir), auth_required=False)
    for module in ("app.routers.projects", "app.routers.scenes", "app.security", "app.routers.bundles", "app.routers.artifact_uploads", "app.services.queue"):
        monkeypatch.setattr(f"{module}.get_settings", lambda: settings)

    def override_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as client:
            res = client.get(f"/api/projects/drj-room/versions/{imported_vid}/manifest")
            assert res.status_code == 200
            assert res.json()["version_id"] == imported_vid
    finally:
        app.dependency_overrides.clear()
        engine.dispose()

    # 3) Missing core artifacts in bundle raises ValueError
    broken_v1 = tmp_path / "broken_bundle" / "v1"
    broken_v1.mkdir(parents=True, exist_ok=True)
    (broken_v1 / "manifest.json").write_text(json.dumps({"project_id": "broken"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required core artifacts"):
        import_kaggle_bundle(tmp_path / "broken_bundle", project_id="broken-proj", storage_dir=storage_dir, db_url=db_url)
