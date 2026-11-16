from __future__ import annotations

import inspect
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from room3d_worker.client import StaleAttemptError, TaskClaim, WorkerClient, WorkerClientError
from room3d_worker.common.errors import StageExecutionError
from room3d_worker.common.manifest import SceneManifest, load_manifest
from room3d_worker.common.subprocess_runner import resolve_physical_gpu, run_stage_on_gpu
from room3d_worker.pipeline.s01_extract import extract_keyframes
from room3d_worker.pipeline.s02_poses import BackendName, PoseBackend, PoseRequest, estimate_poses
from room3d_worker.pipeline.s03_train import build_train_config, train_scene
from room3d_worker.pipeline.s04_export import ExportRequest, export_splat_artifacts
from room3d_worker.pipeline.s05_track import (
    InteractiveSamBackend,
    ManualOverride,
    PromptPoint,
    ReidEncoder,
    VideoSamBackend,
    WarmSamSession,
    load_sam2_backends,
    run_tracking_pipeline,
    save_binary_mask_png,
)
from room3d_worker.pipeline.s06_diffusion import (
    DiffusionConfig,
    InpaintBackend,
    generate_first_frame_seeds,
    load_diffusion_backend,
    propagate_masked_edits,
)
from room3d_worker.pipeline.s07_rebuild import ArrayBool, GaussianState, RebuildRequest, rebuild_scene
from room3d_worker.stage_cli import load_sparse_points


class Sam2SessionManager:
    """Keeps a single warm SAM 2.1 predictor and embedding LRU alive across interactive claims."""

    def __init__(
        self,
        *,
        backend: InteractiveSamBackend | None = None,
        model_cfg: str | None = None,
        checkpoint: str | None = None,
        preferred_gpu: int = 1,
        max_cache_items: int = 16,
    ) -> None:
        self._backend = backend
        self._injected_backend: bool = backend is not None
        self._model_cfg = model_cfg or os.environ.get("ROOM3D_SAM2_CONFIG", "")
        self._checkpoint = checkpoint or os.environ.get("ROOM3D_SAM2_CHECKPOINT", "")
        self._preferred_gpu = preferred_gpu
        self._max_cache_items = max_cache_items
        self._session: WarmSamSession | None = None
        self._video_backend: VideoSamBackend | None = None
        self._lock = threading.Lock()

    @property
    def has_injected_backend(self) -> bool:
        return self._injected_backend

    def _resolved_device(self) -> str:
        physical = resolve_physical_gpu(self._preferred_gpu)
        return f"cuda:{physical}" if physical is not None else "cpu"

    def get_session(self) -> WarmSamSession:
        with self._lock:
            if self._session is not None:
                return self._session
            if self._backend is not None:
                self._session = WarmSamSession(self._backend, max_cache_items=self._max_cache_items)
                return self._session
            if not self._model_cfg or not self._checkpoint:
                raise StageExecutionError(
                    "Interactive SAM 2.1 requires an injected backend or ROOM3D_SAM2_CONFIG + ROOM3D_SAM2_CHECKPOINT"
                )
            interactive, video = load_sam2_backends(self._model_cfg, self._checkpoint, device=self._resolved_device())
            self._backend = interactive
            self._video_backend = video
            self._session = WarmSamSession(interactive, max_cache_items=self._max_cache_items)
            return self._session

    def get_video_backend(self) -> VideoSamBackend:
        self.get_session()
        if self._video_backend is None:
            raise StageExecutionError("Video SAM 2.1 backend is unavailable without live weights or injection")
        return self._video_backend

    def segment(
        self,
        image_path: Path,
        points: Sequence[PromptPoint],
        expected_hw: tuple[int, int],
        mask_out_path: Path,
    ) -> tuple[Path, float]:
        session = self.get_session()
        mask, latency_ms = session.segment_clicks(image_path, points, expected_hw)
        saved = save_binary_mask_png(mask, mask_out_path, expected_hw)
        return saved, latency_ms


@dataclass
class WorkerBackends:
    pose_backends: dict[BackendName, PoseBackend] | None = None
    sam_manager: Sam2SessionManager | None = None
    video_sam: VideoSamBackend | None = None
    reid_encoder: ReidEncoder | None = None
    inpaint_backend: InpaintBackend | None = None
    rebuild_backend: Callable[[RebuildRequest, SceneManifest, GaussianState, ArrayBool], GaussianState] | None = None
    use_isolated_subprocesses: bool = False


class _StorageAccessor:
    def __init__(self, client: WorkerClient, storage_dir: Path | None) -> None:
        self._client = client
        self._storage_dir = storage_dir.resolve() if storage_dir is not None else None

    def _local_path(self, key: str) -> Path | None:
        if self._storage_dir is None:
            return None
        clean = key.lstrip("/")
        candidate = (self._storage_dir / clean).resolve()
        if self._storage_dir not in candidate.parents and candidate != self._storage_dir:
            raise WorkerClientError(f"Storage key escapes root: {key}")
        return candidate

    def fetch(self, key: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        local = self._local_path(key)
        if local is not None:
            if not local.is_file():
                raise FileNotFoundError(f"Missing local storage key: {key}")
            if local != dest.resolve():
                shutil.copy2(local, dest)
            return dest
        self._client.download_key(key, dest)
        return dest

    def try_fetch(self, key: str, dest: Path) -> Path | None:
        try:
            return self.fetch(key, dest)
        except (FileNotFoundError, WorkerClientError):
            return None

    def put(self, key: str, src: Path) -> None:
        local = self._local_path(key)
        if local is not None:
            local.parent.mkdir(parents=True, exist_ok=True)
            if local != src.resolve():
                shutil.copy2(src, local)
            return
        self._client.upload_key(key, src)


def _prompt_points(raw_points: Sequence[dict[str, Any]], width: int, height: int) -> list[PromptPoint]:
    points: list[PromptPoint] = []
    for pt in raw_points:
        if pt.get("x") is not None and pt.get("y") is not None:
            x, y = float(pt["x"]), float(pt["y"])
        else:
            x = float(pt.get("u_norm", 0.5)) * width
            y = float(pt.get("v_norm", 0.5)) * height
        points.append(PromptPoint(x=x, y=y, label=1 if int(pt.get("label", 1)) == 1 else 0))
    return points


def _parse_seed_int(raw_seed: Any) -> int:
    text = str(raw_seed).strip()
    if text.startswith("seed_"):
        text = text[5:]
    try:
        return int(text)
    except ValueError:
        return 0


def _stage_device(preferred_gpu: int) -> str:
    physical = resolve_physical_gpu(preferred_gpu)
    return f"cuda:{physical}" if physical is not None else "cpu"


class WorkerRunner:
    """Pull-based Room3D worker daemon dispatching Week 1 & Week 2 pipeline tasks."""

    def __init__(
        self,
        client: WorkerClient,
        *,
        worker_id: str = "room3d-worker",
        work_root: Path | None = None,
        storage_dir: Path | None = None,
        capabilities: list[str] | None = None,
        interactive_only: bool = False,
        backends: WorkerBackends | None = None,
        poll_interval_sec: float = 0.5,
        heartbeat_interval_sec: float = 15.0,
        stage_timeout_sec: int = 3600,
    ) -> None:
        self.client = client
        self.worker_id = worker_id
        self.work_root = Path(work_root) if work_root is not None else Path(tempfile.mkdtemp(prefix="room3d_work_"))
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.storage = _StorageAccessor(client, storage_dir)
        self.capabilities = list(capabilities or [])
        self.interactive_only = interactive_only
        self.backends = backends or WorkerBackends()
        if self.backends.sam_manager is None:
            self.backends.sam_manager = Sam2SessionManager(preferred_gpu=1)
        self.poll_interval_sec = poll_interval_sec
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.stage_timeout_sec = stage_timeout_sec
        self._frame_cache_dir = self.work_root / "_warm_frames"
        self._frame_cache_dir.mkdir(parents=True, exist_ok=True)

    def _start_heartbeat(self, task: TaskClaim) -> tuple[threading.Event, threading.Thread]:
        stop_event = threading.Event()
        interval = max(0.05, float(self.heartbeat_interval_sec))

        def _loop() -> None:
            while not stop_event.wait(interval):
                try:
                    self.client.heartbeat(task, self.worker_id)
                except (StaleAttemptError, WorkerClientError):
                    break

        thread = threading.Thread(target=_loop, name=f"hb-{task.task_id[:8]}", daemon=True)
        thread.start()
        return stop_event, thread

    def run_once(self) -> bool:
        task = self.client.claim(self.worker_id, self.capabilities, interactive_only=self.interactive_only)
        if task is None:
            return False
        stop_hb: threading.Event | None = None
        hb_thread: threading.Thread | None = None
        try:
            self.client.heartbeat(task, self.worker_id)
            stop_hb, hb_thread = self._start_heartbeat(task)
            result_payload, manifest_key = self.execute_claim(task)
            stop_hb.set()
            hb_thread.join(timeout=2.0)
            self.client.complete(task, result_payload, result_manifest_key=manifest_key)
        except StaleAttemptError:
            return True
        except Exception as exc:
            self.client.fail(task, type(exc).__name__, str(exc), retryable=isinstance(exc, OSError))
        finally:
            if stop_hb is not None:
                stop_hb.set()
            if hb_thread is not None:
                hb_thread.join(timeout=2.0)
        return True

    def run_forever(self, stop_event: threading.Event | None = None, max_tasks: int | None = None) -> int:
        stop_event = stop_event or threading.Event()
        processed = 0
        while not stop_event.is_set():
            if max_tasks is not None and processed >= max_tasks:
                break
            if self.run_once():
                processed += 1
            else:
                stop_event.wait(self.poll_interval_sec)
        return processed

    def execute_claim(self, task: TaskClaim) -> tuple[dict[str, Any], str | None]:
        dispatch = {
            "reconstruct_week1": self._handle_week1,
            "build_week1": self._handle_week1,
            "segment_init": self._handle_segment,
            "segment_refine": self._handle_segment,
            "track_propagate": self._handle_track,
            "diffusion_preview": self._handle_diffusion_preview,
            "diffusion_apply": self._handle_diffusion_apply,
            "rebuild_week2": self._handle_rebuild,
        }
        handler = dispatch.get(task.task_type)
        if handler is None:
            raise WorkerClientError(f"Unsupported task type: {task.task_type}")
        return handler(task)

    def _run_isolated(self, stage: str, spec: dict[str, Any], preferred_gpu: int, work_dir: Path) -> dict[str, Any]:
        spec_path = work_dir / f"{stage}_spec.json"
        out_path = work_dir / f"{stage}_out.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        run_stage_on_gpu(
            [sys.executable, "-m", "room3d_worker.stage_cli", stage, "--spec", str(spec_path), "--out", str(out_path)],
            preferred_gpu,
            None,
            self.stage_timeout_sec,
        )
        return json.loads(out_path.read_text(encoding="utf-8"))

    def _fetch_manifest(self, manifest_key: str, dest_dir: Path) -> SceneManifest:
        manifest_path = self.storage.fetch(manifest_key, dest_dir / "manifest.json")
        return load_manifest(manifest_path)

    def _handle_week1(self, task: TaskClaim) -> tuple[dict[str, Any], str | None]:
        payload = task.payload
        project_id = str(payload.get("project_id") or task.project_id or "")
        version_id = str(payload.get("version_id") or "v1")
        source_key = str(payload.get("source_video_key") or payload.get("input_key") or "")
        if not project_id or not source_key:
            raise WorkerClientError("Week 1 task requires project_id and source_video_key/input_key")

        work_dir = self.work_root / task.task_id
        scene_dir = work_dir / "scene"
        scene_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(source_key).suffix or ".mp4"
        video_path = self.storage.fetch(source_key, work_dir / f"input{suffix}")

        self.client.progress(task, "01_extract", 10.0, "Extracting keyframes")
        extracted = extract_keyframes(video_path, scene_dir, **_extract_kwargs(payload))
        raw_frames = list(extracted.index["frames"])  # type: ignore[arg-type]
        width = int(extracted.index["width"])  # type: ignore[arg-type]
        height = int(extracted.index["height"])  # type: ignore[arg-type]

        self.client.progress(task, "02_poses", 25.0, "Estimating camera poses")
        frame_records = [
            {
                "frame_idx": int(f["frame_idx"]),
                "orig_video_frame": int(f["orig_video_frame"]),
                "timestamp_sec": float(f["timestamp_sec"]),
                "image_relpath": str(f["image_relpath"]),
                "colmap_image_id": int(f["frame_idx"]) + 1,
                "cam_from_world": np.eye(4).tolist(),
            }
            for f in raw_frames
        ]
        focal = float(max(width, height))
        intrinsics = {"fx": focal, "fy": focal, "cx": width / 2.0, "cy": height / 2.0}
        if self.backends.use_isolated_subprocesses and self.backends.pose_backends is None:
            self._run_isolated(
                "poses",
                {
                    "image_dir": str(scene_dir / "images"),
                    "output_dir": str(scene_dir),
                    "total_frames": len(frame_records),
                    "width": width,
                    "height": height,
                    "intrinsics": intrinsics,
                    "project_id": project_id,
                    "version_id": version_id,
                    "pose_backend": payload.get("pose_backend", "colmap_global"),
                    "frame_records": frame_records,
                    "manifest_path": str(scene_dir / "manifest.json"),
                },
                1,
                work_dir,
            )
            pose_manifest = load_manifest(scene_dir / "manifest.json")
        else:
            pose_req = PoseRequest(
                image_dir=scene_dir / "images",
                output_dir=scene_dir,
                total_frames=len(frame_records),
                width=width,
                height=height,
                intrinsics=intrinsics,
                project_id=project_id,
                version_id=version_id,
                backend=payload.get("pose_backend", "colmap_global"),
                frame_records=frame_records,
                manifest_path=scene_dir / "manifest.json",
            )
            pose_manifest, _ = estimate_poses(pose_req, backends=self.backends.pose_backends)

        self.client.progress(task, "03_train", 25.0, "Training 3D Gaussian scene")
        train_device = "cuda:0" if self.backends.use_isolated_subprocesses else _stage_device(1)
        allow_synthetic = self.backends.pose_backends is not None
        if self.backends.use_isolated_subprocesses and self.backends.pose_backends is None:
            self._run_isolated(
                "train",
                {
                    "output_dir": str(scene_dir),
                    "manifest_path": str(scene_dir / "manifest.json"),
                    "device": train_device,
                    "iterations": int(payload.get("iterations", 7000)),
                    "allow_synthetic": allow_synthetic,
                },
                1,
                work_dir,
            )
        else:
            xyz, rgb = _safe_sparse_points(
                scene_dir / "sparse" / "0",
                len(pose_manifest.frames),
                allow_synthetic=allow_synthetic,
            )
            imgs = _load_stack(
                scene_dir,
                [f.model_dump() for f in pose_manifest.frames],
                allow_synthetic=allow_synthetic,
            )
            train_cfg = _build_train_cfg(
                scene_dir,
                device=train_device,
                iterations=int(payload.get("iterations", 7000)),
                allow_synthetic=allow_synthetic,
            )
            train_scene(train_cfg, imgs, xyz, points_rgb=rgb, manifest=pose_manifest)

        self.client.progress(task, "04_export", 50.0, "Exporting .spz and manifest")
        export_splat_artifacts(
            ExportRequest(
                output_dir=scene_dir,
                project_id=project_id,
                version_id=version_id,
                width=pose_manifest.width,
                height=pose_manifest.height,
                intrinsics=pose_manifest.intrinsics.model_dump(),
                frames=[f.model_dump() for f in pose_manifest.frames],
                test_every=pose_manifest.test_every,
                pose_backend=pose_manifest.pose_backend,
                T_viewer_from_colmap=pose_manifest.T_viewer_from_colmap,
            ),
            ckpt_path=scene_dir / "gs_state.npz",
        )

        self._upload_scene_bundle(project_id, version_id, scene_dir, pose_manifest)
        manifest_key = f"{project_id}/{version_id}/manifest.json"
        return {"version_id": version_id, "label": payload.get("label") or version_id, "manifest_key": manifest_key}, manifest_key

    def _upload_scene_bundle(self, project_id: str, version_id: str, scene_dir: Path, manifest: SceneManifest) -> None:
        for rel in ("manifest.json", "scene.spz", "scene.ply", "sparse_points.json", "gs_state.npz", "gs_ckpt.pt"):
            path = scene_dir / rel
            if path.is_file():
                self.storage.put(f"{project_id}/{version_id}/{rel}", path)
        for sparse_file in ("cameras.bin", "images.bin", "points3D.bin"):
            path = scene_dir / "sparse" / "0" / sparse_file
            if path.is_file():
                self.storage.put(f"{project_id}/{version_id}/sparse/0/{sparse_file}", path)
        for frame in manifest.frames:
            img_path = scene_dir / frame.image_relpath
            if img_path.is_file():
                self.storage.put(f"{project_id}/{version_id}/{frame.image_relpath}", img_path)
        depths_dir = scene_dir / "depths"
        if depths_dir.is_dir():
            for npy in depths_dir.glob("*.npy"):
                self.storage.put(f"{project_id}/{version_id}/depths/{npy.name}", npy)

    def _handle_segment(self, task: TaskClaim) -> tuple[dict[str, Any], str | None]:
        payload = task.payload
        project_id = str(payload.get("project_id") or task.project_id or "")
        edit_id = str(payload["edit_id"])
        base_version_id = str(payload.get("base_version_id") or "")
        ref_idx = int(payload.get("ref_frame_idx") or 0)
        manifest_key = payload.get("manifest_key") or f"{project_id}/{base_version_id}/manifest.json"

        cache_dir = self._frame_cache_dir / project_id / (base_version_id or "active")
        manifest_path = cache_dir / "manifest.json"
        manifest = load_manifest(manifest_path) if manifest_path.is_file() else self._fetch_manifest(manifest_key, cache_dir)
        image_rel = payload.get("image_relpath") or next(
            (f.image_relpath for f in manifest.frames if f.frame_idx == ref_idx), None
        )
        image_key = payload.get("image_key") or (
            f"{project_id}/{base_version_id}/{image_rel}" if base_version_id and image_rel else None
        )
        if not image_key or not image_rel:
            raise WorkerClientError(f"Could not locate reference frame {ref_idx} for edit {edit_id}")
        local_img = cache_dir / image_rel
        if not local_img.is_file():
            self.storage.fetch(image_key, local_img)

        points = _prompt_points(payload.get("points") or [], manifest.width, manifest.height)
        mask_out_key = str(payload.get("mask_out_key") or f"{project_id}/edits/{edit_id}/masks/{ref_idx:06d}.png")
        local_mask = self.work_root / task.task_id / f"{ref_idx:06d}.png"
        assert self.backends.sam_manager is not None
        _, latency_ms = self.backends.sam_manager.segment(local_img, points, (manifest.height, manifest.width), local_mask)
        self.storage.put(mask_out_key, local_mask)

        mask_entry = {"frame_idx": ref_idx, "mask_key": mask_out_key, "status": "segmented", "skip_frame": False}
        return {
            "edit_id": edit_id,
            "ref_frame_idx": ref_idx,
            "mask_key": mask_out_key,
            "latency_ms": round(latency_ms, 2),
            "mask_summary": {str(ref_idx): mask_entry},
        }, None

    def _handle_track(self, task: TaskClaim) -> tuple[dict[str, Any], str | None]:
        payload = task.payload
        project_id = str(payload.get("project_id") or task.project_id or "")
        edit_id = str(payload["edit_id"])
        base_version_id = str(payload["base_version_id"])
        ref_idx = int(payload.get("ref_frame_idx") or 0)
        manifest_key = payload.get("manifest_key") or f"{project_id}/{base_version_id}/manifest.json"
        masks_prefix = str(payload.get("masks_prefix") or f"{project_id}/edits/{edit_id}/masks")

        work_dir = self.work_root / task.task_id
        scene_dir = work_dir / "base"
        manifest = self._fetch_manifest(manifest_key, scene_dir)
        image_paths: dict[int, Path] = {}
        for frame in manifest.frames:
            if not frame.excluded:
                image_paths[frame.frame_idx] = self.storage.fetch(
                    f"{project_id}/{base_version_id}/{frame.image_relpath}", scene_dir / frame.image_relpath
                )

        depth_dir = scene_dir / "depths"
        has_depths = False
        for frame in manifest.frames:
            target_depth = depth_dir / f"{frame.frame_idx:06d}.npy"
            fetched = self.storage.try_fetch(
                f"{project_id}/{base_version_id}/depths/{frame.frame_idx:06d}.npy",
                target_depth,
            ) or self.storage.try_fetch(
                f"{project_id}/{base_version_id}/depths/{frame.frame_idx:05d}.npy",
                target_depth,
            )
            if fetched is not None:
                has_depths = True

        corrections: dict[str, Any] = payload.get("keyframe_corrections") or payload.get("keyframe_Corrections") or {}
        overrides: list[ManualOverride] = []
        for idx_str, entry in corrections.items():
            if not isinstance(entry, dict):
                continue
            idx = int(idx_str)
            skip = bool(entry.get("skip_frame", False))
            mkey = entry.get("mask_key")
            mpath = self.storage.try_fetch(str(mkey), work_dir / "manual" / f"{idx:06d}.png") if mkey else None
            overrides.append(ManualOverride(frame_idx=idx, mask_path=mpath, skip_frame=skip))

        self.client.progress(task, "05_track", 40.0, "Tracking object across frames")
        use_isolated_track = (
            self.backends.use_isolated_subprocesses
            and self.backends.video_sam is None
            and (self.backends.sam_manager is None or not self.backends.sam_manager.has_injected_backend)
        )
        if use_isolated_track:
            out = self._run_isolated(
                "track",
                {
                    "manifest_path": str(scene_dir / "manifest.json"),
                    "image_paths": {str(k): str(v) for k, v in image_paths.items()},
                    "ref_frame_idx": ref_idx,
                    "points": list(payload.get("points") or []),
                    "output_dir": str(work_dir / "track"),
                    "depth_dir": str(depth_dir) if has_depths else None,
                    "overrides": [
                        {
                            "frame_idx": ov.frame_idx,
                            "mask_path": str(ov.mask_path) if ov.mask_path else None,
                            "skip_frame": ov.skip_frame,
                        }
                        for ov in overrides
                    ],
                    "sam2_config": self.backends.sam_manager._model_cfg if self.backends.sam_manager else "",
                    "sam2_checkpoint": self.backends.sam_manager._checkpoint if self.backends.sam_manager else "",
                    "device": "cuda:0",
                },
                1,
                work_dir,
            )
            saved_masks = {int(k): Path(v) for k, v in dict(out["saved_masks"]).items()}
            stats = dict(out["stats"])
        else:
            assert self.backends.sam_manager is not None
            sam_session = self.backends.sam_manager.get_session()
            video_backend = self.backends.video_sam or self.backends.sam_manager.get_video_backend()
            points = _prompt_points(payload.get("points") or [], manifest.width, manifest.height)
            saved_masks, stats = run_tracking_pipeline(
                manifest,
                image_paths,
                ref_frame_idx=ref_idx,
                points=points,
                sam_session=sam_session,
                video_backend=video_backend,
                output_dir=work_dir / "track",
                depth_dir=depth_dir if has_depths else None,
                reid_encoder=self.backends.reid_encoder,
                overrides=overrides,
            )
        skipped_set = {int(i) for i in stats.get("skipped_frames", [])}
        masks_summary: dict[str, dict[str, Any]] = {}
        for idx, mpath in sorted(saved_masks.items()):
            mask_key = f"{masks_prefix}/{idx:06d}.png"
            self.storage.put(mask_key, mpath)
            is_manual = str(idx) in corrections and not bool(corrections[str(idx)].get("skip_frame"))
            status = "skipped" if idx in skipped_set else ("manual" if is_manual else "auto")
            masks_summary[str(idx)] = {"status": status, "mask_key": mask_key, "skip_frame": idx in skipped_set}
        return {"edit_id": edit_id, "tracked_frames": len(saved_masks), "masks_summary": masks_summary, "report": stats}, None

    def _handle_diffusion_preview(self, task: TaskClaim) -> tuple[dict[str, Any], str | None]:
        payload = task.payload
        project_id = str(payload.get("project_id") or task.project_id or "")
        edit_id = str(payload["edit_id"])
        base_version_id = str(payload["base_version_id"])
        ref_idx = int(payload.get("ref_frame_idx") or 0)
        manifest_key = payload.get("manifest_key") or f"{project_id}/{base_version_id}/manifest.json"
        previews_prefix = str(payload.get("previews_prefix") or f"{project_id}/edits/{edit_id}/previews").rstrip("/")
        if payload.get("preview_generation") is not None:
            gen_tag = f"g{int(payload['preview_generation'])}"
            if gen_tag not in previews_prefix.split("/"):
                previews_prefix = f"{previews_prefix}/{gen_tag}"
        attempt_tag = str(getattr(task, "attempt_id", None) or task.task_id)
        if attempt_tag and attempt_tag not in previews_prefix.split("/"):
            previews_prefix = f"{previews_prefix}/{attempt_tag}"

        work_dir = self.work_root / task.task_id
        manifest = self._fetch_manifest(manifest_key, work_dir / "base")
        ref_rel = next(f.image_relpath for f in manifest.frames if f.frame_idx == ref_idx)
        ref_img = self.storage.fetch(f"{project_id}/{base_version_id}/{ref_rel}", work_dir / "ref_img.png")
        mask_key = str(payload.get("mask_key") or f"{project_id}/edits/{edit_id}/masks/{ref_idx:06d}.png")
        ref_mask = self.storage.fetch(mask_key, work_dir / "ref_mask.png")

        seeds = [int(s) for s in payload.get("seeds", [101, 202, 303])]
        variant = payload.get("model_variant") or "flux2_klein_4b"
        negative_prompt = payload.get("negative_prompt")
        self.client.progress(task, "06_diffusion", 30.0, "Generating seed previews")

        if self.backends.use_isolated_subprocesses and self.backends.inpaint_backend is None:
            out = self._run_isolated(
                "diffusion_preview",
                {
                    "ref_image_path": str(ref_img),
                    "ref_mask_path": str(ref_mask),
                    "prompt": str(payload["prompt"]),
                    "negative_prompt": negative_prompt,
                    "seeds": seeds,
                    "model_variant": variant,
                    "output_dir": str(work_dir / "seeds"),
                    "device": "cuda:0",
                },
                0,
                work_dir,
            )
            generated = out["seeds"]
        else:
            cfg = _diffusion_cfg(variant, negative_prompt, _stage_device(0), grid_2x2=False)
            backend = self.backends.inpaint_backend or load_diffusion_backend(cfg)
            generated = generate_first_frame_seeds(
                ref_img, ref_mask, str(payload["prompt"]), seeds, work_dir / "seeds", backend, config=cfg
            )

        preview_seeds: dict[str, str] = {}
        seeds_out: list[dict[str, Any]] = []
        for item in generated:
            seed_val = int(item["seed"])
            seed_id = str(item["seed_id"])
            key = f"{previews_prefix}/{seed_id}.png"
            self.storage.put(key, Path(item["path"]))
            preview_seeds[str(seed_val)] = key
            seeds_out.append({"seed_id": seed_id, "seed": seed_val, "preview_key": key})
        return {"edit_id": edit_id, "preview_seeds": preview_seeds, "seeds": seeds_out}, None

    def _handle_diffusion_apply(self, task: TaskClaim) -> tuple[dict[str, Any], str | None]:
        payload = task.payload
        project_id = str(payload.get("project_id") or task.project_id or "")
        edit_id = str(payload["edit_id"])
        base_version_id = str(payload["base_version_id"])
        ref_idx = int(payload.get("ref_frame_idx") or 0)
        manifest_key = payload.get("manifest_key") or f"{project_id}/{base_version_id}/manifest.json"
        masks_prefix = str(payload.get("masks_prefix") or f"{project_id}/edits/{edit_id}/masks")
        edited_prefix = str(payload.get("edited_frames_prefix") or f"{project_id}/edits/{edit_id}/edited_frames")

        work_dir = self.work_root / task.task_id
        manifest = self._fetch_manifest(manifest_key, work_dir / "base")
        frame_paths: dict[int, Path] = {}
        mask_paths: dict[int, Path] = {}
        masks_summary: dict[str, Any] = dict(payload.get("masks_summary") or {})
        for frame in manifest.frames:
            if frame.excluded:
                continue
            idx = frame.frame_idx
            frame_paths[idx] = self.storage.fetch(
                f"{project_id}/{base_version_id}/{frame.image_relpath}", work_dir / "frames" / f"{idx:06d}.png"
            )
            mkey = (masks_summary.get(str(idx)) or {}).get("mask_key") or f"{masks_prefix}/{idx:06d}.png"
            fetched_mask = self.storage.try_fetch(str(mkey), work_dir / "masks" / f"{idx:06d}.png")
            if fetched_mask is not None:
                mask_paths[idx] = fetched_mask

        preview_key = str(payload["preview_key"])
        ref_edited = self.storage.fetch(preview_key, work_dir / "ref_edited.png")
        seed_int = _parse_seed_int(payload.get("selected_seed") or payload.get("chosen_seed_id") or 0)
        skip_frames = {int(i) for i in payload.get("exclude_frames", [])}
        variant = payload.get("model_variant") or "flux2_klein_4b"
        negative_prompt = payload.get("negative_prompt")

        self.client.progress(task, "06_diffusion", 50.0, "Applying reference-conditioned edits")
        if self.backends.use_isolated_subprocesses and self.backends.inpaint_backend is None:
            metrics = self._run_isolated(
                "diffusion_apply",
                {
                    "frame_paths": {str(k): str(v) for k, v in frame_paths.items()},
                    "mask_paths": {str(k): str(v) for k, v in mask_paths.items()},
                    "ref_frame_idx": ref_idx,
                    "ref_edited_path": str(ref_edited),
                    "prompt": str(payload["prompt"]),
                    "negative_prompt": negative_prompt,
                    "seed": seed_int,
                    "skip_frames": sorted(skip_frames),
                    "model_variant": variant,
                    "output_dir": str(work_dir / "diff_out"),
                    "device": "cuda:0",
                },
                0,
                work_dir,
            )
        else:
            cfg = _diffusion_cfg(variant, negative_prompt, _stage_device(0), grid_2x2=False)
            backend = self.backends.inpaint_backend or load_diffusion_backend(cfg)
            metrics = propagate_masked_edits(
                frame_paths,
                mask_paths,
                ref_idx,
                ref_edited,
                str(payload["prompt"]),
                seed_int,
                work_dir / "diff_out",
                backend,
                skip_frames=skip_frames,
                config=cfg,
            )

        edited_dir = work_dir / "diff_out" / "edited_frames"
        for out_file in sorted(edited_dir.glob("*.png")):
            self.storage.put(f"{edited_prefix}/{out_file.name}", out_file)
        return {"edit_id": edit_id, "edited_frames_prefix": edited_prefix, **metrics}, None

    def _handle_rebuild(self, task: TaskClaim) -> tuple[dict[str, Any], str | None]:
        payload = task.payload
        project_id = str(payload.get("project_id") or task.project_id or "")
        edit_id = str(payload["edit_id"])
        base_version_id = str(payload["base_version_id"])
        new_version_id = str(payload.get("new_version_id") or payload.get("target_version_id") or f"edit_{edit_id[:8]}")
        manifest_key = payload.get("manifest_key") or f"{project_id}/{base_version_id}/manifest.json"
        masks_prefix = str(payload.get("masks_prefix") or f"{project_id}/edits/{edit_id}/masks")
        edited_prefix = str(payload.get("edited_frames_prefix") or f"{project_id}/edits/{edit_id}/edited_frames")

        work_dir = self.work_root / task.task_id
        base_dir = work_dir / "base"
        edited_dir = work_dir / "edited_frames"
        masks_dir = work_dir / "masks"
        out_dir = work_dir / "rebuilt"
        edited_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        manifest = self._fetch_manifest(manifest_key, base_dir)
        for rel in ("gs_state.npz", "gs_ckpt.pt", "scene.ply", "sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin"):
            self.storage.try_fetch(f"{project_id}/{base_version_id}/{rel}", base_dir / rel)
        masks_summary: dict[str, Any] = dict(payload.get("masks_summary") or {})
        for frame in manifest.frames:
            idx = frame.frame_idx
            self.storage.try_fetch(f"{project_id}/{base_version_id}/{frame.image_relpath}", base_dir / frame.image_relpath)
            self.storage.try_fetch(f"{edited_prefix}/{idx:06d}.png", edited_dir / f"{idx:06d}.png")
            mkey = (masks_summary.get(str(idx)) or {}).get("mask_key") or f"{masks_prefix}/{idx:06d}.png"
            self.storage.try_fetch(str(mkey), masks_dir / f"{idx:06d}.png")

        self.client.progress(task, "07_rebuild", 50.0, "Rebuilding 3D Gaussian scene")
        exclude_frames = tuple(int(i) for i in payload.get("exclude_frames", []))
        mode = payload.get("rebuild_mode", "masked_finetune")
        if self.backends.use_isolated_subprocesses and self.backends.rebuild_backend is None:
            metrics = self._run_isolated(
                "rebuild",
                {
                    "base_dir": str(base_dir),
                    "edited_frames_dir": str(edited_dir),
                    "masks_dir": str(masks_dir),
                    "output_dir": str(out_dir),
                    "project_id": project_id,
                    "version_id": new_version_id,
                    "parent_version_id": base_version_id,
                    "mode": mode,
                    "exclude_frames": list(exclude_frames),
                    "device": "cuda:0",
                },
                1,
                work_dir,
            )
            rebuilt_manifest = load_manifest(out_dir / "manifest.json")
        else:
            req = RebuildRequest(
                base_dir=base_dir,
                edited_frames_dir=edited_dir,
                masks_dir=masks_dir,
                output_dir=out_dir,
                project_id=project_id,
                version_id=new_version_id,
                parent_version_id=base_version_id,
                mode=mode,
                exclude_frames=exclude_frames,
                device="cpu" if self.backends.rebuild_backend is not None else _stage_device(1),
            )
            metrics = rebuild_scene(req, backend=self.backends.rebuild_backend)
            rebuilt_manifest = load_manifest(out_dir / "manifest.json")

        self._upload_scene_bundle(project_id, new_version_id, out_dir, rebuilt_manifest)
        for frame in rebuilt_manifest.frames:
            edited_file = edited_dir / f"{frame.frame_idx:06d}.png"
            src = edited_file if edited_file.is_file() else (base_dir / frame.image_relpath)
            if src.is_file():
                self.storage.put(f"{project_id}/{new_version_id}/{frame.image_relpath}", src)

        out_manifest_key = f"{project_id}/{new_version_id}/manifest.json"
        return {
            "version_id": new_version_id,
            "parent_version_id": base_version_id,
            "label": f"Edit {edit_id[:8]}",
            **metrics,
        }, out_manifest_key


def _build_train_cfg(out_dir: Path, device: str, iterations: int, *, allow_synthetic: bool) -> Any:
    sig = inspect.signature(build_train_config)
    kwargs: dict[str, Any] = {"device": device, "iterations": iterations}
    if "allow_synthetic_fallback" in sig.parameters:
        kwargs["allow_synthetic_fallback"] = allow_synthetic
    return build_train_config(out_dir, **kwargs)


def _extract_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(extract_keyframes)
    optional = ("max_long_side", "min_parallax_frac", "sharpness_window", "target_fps", "max_frames", "trim_start_sec", "trim_end_sec")
    kwargs: dict[str, Any] = {}
    for key in optional:
        if key in sig.parameters and payload.get(key) is not None:
            kwargs[key] = payload[key]
    return kwargs


def _diffusion_cfg(
    variant: Any,
    negative_prompt: Any,
    device: str,
    *,
    grid_2x2: bool = False,
) -> DiffusionConfig:
    sig = inspect.signature(DiffusionConfig)
    kwargs: dict[str, Any] = {
        "model_variant": variant or "flux2_klein_4b",
        "device": device,
        "grid_2x2": grid_2x2,
    }
    if "negative_prompt" in sig.parameters and negative_prompt:
        kwargs["negative_prompt"] = str(negative_prompt)
    return DiffusionConfig(**kwargs)


def _safe_sparse_points(
    sparse_dir: Path, num_frames: int, *, allow_synthetic: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        return load_sparse_points(sparse_dir)
    except (StageExecutionError, struct.error, OSError) as exc:
        if not allow_synthetic:
            raise StageExecutionError(f"Missing sparse point cloud under {sparse_dir}") from exc
        rng = np.random.default_rng(max(1, num_frames))
        pts = rng.uniform(-1.0, 1.0, size=(64, 3)).astype(np.float32)
        return pts, None


def _load_stack(scene_dir: Path, frames: list[dict[str, Any]], *, allow_synthetic: bool = False) -> np.ndarray:
    imgs: list[np.ndarray] = []
    for frame in frames:
        path = scene_dir / str(frame["image_relpath"])
        if path.is_file():
            with Image.open(path) as img:
                imgs.append(np.asarray(img.convert("RGB"), dtype=np.uint8))
    if not imgs:
        if allow_synthetic:
            return np.zeros((1, 64, 64, 3), dtype=np.uint8)
        raise StageExecutionError(f"No extracted RGB frame images found under {scene_dir}")
    return np.stack(imgs, axis=0)
