from __future__ import annotations

import importlib
import inspect
import json
import os
import shutil
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from room3d_worker.common.coords import (
    align_gravity_up,
    mod16_crop_or_pad_params,
    transform_cam_from_world_sim3,
    umeyama_sim3,
)
from room3d_worker.common.errors import GeometryError, StageExecutionError
from room3d_worker.common.manifest import SceneManifest
from room3d_worker.common.subprocess_runner import run_isolated_stage

BackendName = Literal[
    "colmap_global",
    "colmap_incremental",
    "vggt_chunked",
    "map_anything",
    "mapanything",
    "precomputed_colmap",
]
FloatMatrix = NDArray[np.float64]


class ColmapFailureError(StageExecutionError):
    """COLMAP reconstruction failed or produced insufficient registered geometry."""


@dataclass(frozen=True)
class PoseRequest:
    image_dir: Path
    output_dir: Path
    total_frames: int
    width: int
    height: int
    intrinsics: dict[str, float]
    project_id: str = "project"
    version_id: str = "version"
    database_path: Path | None = None
    backend: BackendName = "colmap_global"
    frame_records: list[dict[str, Any]] = field(default_factory=list)
    manifest_path: Path | None = None
    depth_dir: Path | None = None
    vocab_tree_path: Path | None = None


@dataclass(frozen=True)
class PoseTelemetry:
    registered_frames: int
    total_frames: int
    mean_reproj_px: float
    num_points3d: int
    runtime_sec: float
    peak_vram_mb: float


@dataclass
class BackendOutput:
    registered_frames: int
    mean_reproj_px: float
    num_points3d: int
    points_xyz: NDArray[np.float64]
    cam_positions_xyz: NDArray[np.float64]
    frames: list[dict[str, Any]]
    sparse_dir: Path | None = None
    depth_dir: Path | None = None
    width: int | None = None
    height: int | None = None
    intrinsics: dict[str, float] | None = None
    depths: Any = None


class PoseBackend(Protocol):
    def estimate(self, request: PoseRequest) -> BackendOutput: ...


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise StageExecutionError(f"Pose backend dependency is unavailable: {name}") from exc


def _call_optional(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if not kwargs:
        return fn(*args)
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return fn(*args, **kwargs)
    params = sig.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return fn(*args, **kwargs)
    accepted = {
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return fn(*args, **filtered)


def align_sim3(source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Return scale, rotation, translation mapping source points onto target points."""
    sim = umeyama_sim3(source_xyz, target_xyz)
    return sim.scale, sim.rotation, sim.translation


def _peak_vram_mb() -> float:
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except ImportError:
        return 0.0
    return 0.0


def _rotmat_to_qvec(rot: NDArray[np.float64]) -> tuple[float, float, float, float]:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rot.flat
    k = np.array([
        [rxx - ryy - rzz, 0.0, 0.0, 0.0],
        [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
        [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
        [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], int(np.argmax(eigvals))]
    if qvec[0] < 0.0:
        qvec = -qvec
    qvec = qvec / np.linalg.norm(qvec)
    return float(qvec[0]), float(qvec[1]), float(qvec[2]), float(qvec[3])


def write_colmap_binary(
    sparse_dir: Path,
    width: int,
    height: int,
    intrinsics: dict[str, float],
    frames: Sequence[dict[str, Any]],
    points_xyz: NDArray[np.float64],
    points_rgb: NDArray[np.uint8] | None = None,
) -> Path:
    """Write standard COLMAP PINHOLE binary files (cameras.bin, images.bin, points3D.bin)."""
    sparse_dir.mkdir(parents=True, exist_ok=True)
    fx, fy, cx, cy = (float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))

    with (sparse_dir / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<IiQQ4d", 1, 1, int(width), int(height), fx, fy, cx, cy))

    registered = [fr for fr in frames if fr.get("cam_from_world") is not None]
    with (sparse_dir / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(registered)))
        for idx, fr in enumerate(registered, start=1):
            image_id = int(fr.get("colmap_image_id") or idx)
            pose = np.asarray(fr["cam_from_world"], dtype=np.float64)
            qw, qx, qy, qz = _rotmat_to_qvec(pose[:3, :3])
            tx, ty, tz = (float(v) for v in pose[:3, 3])
            name = Path(str(fr.get("image_relpath") or f"{idx:06d}.png")).name
            handle.write(struct.pack("<I4d3dI", image_id, qw, qx, qy, qz, tx, ty, tz, 1))
            handle.write(name.encode("utf-8") + b"\x00")
            handle.write(struct.pack("<Q", 0))

    pts = np.asarray(points_xyz, dtype=np.float64)
    if len(pts) == 0:
        pts = np.empty((0, 3), dtype=np.float64)
    rgb = np.asarray(points_rgb, dtype=np.uint8) if points_rgb is not None else np.full((len(pts), 3), 200, dtype=np.uint8)
    with (sparse_dir / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(pts)))
        for pid, (xyz, color) in enumerate(zip(pts, rgb), start=1):
            handle.write(struct.pack("<Q3d3BdQ", pid, float(xyz[0]), float(xyz[1]), float(xyz[2]),
                                     int(color[0]), int(color[1]), int(color[2]), 0.0, 0))
    return sparse_dir


def _extract_cam_from_world_4x4(image: Any) -> list[list[float]]:
    raw = image.cam_from_world() if callable(getattr(image, "cam_from_world", None)) else getattr(image, "cam_from_world", None)
    if raw is None:
        raise ColmapFailureError(f"Registered image {getattr(image, 'name', '')} is missing cam_from_world")
    mat = np.asarray(raw.matrix() if hasattr(raw, "matrix") else raw, dtype=np.float64)
    if mat.shape == (3, 4):
        full = np.eye(4, dtype=np.float64)
        full[:3, :] = mat
        mat = full
    elif mat.shape != (4, 4):
        raise ColmapFailureError(f"Unsupported cam_from_world shape {mat.shape}")
    if not np.isfinite(mat).all():
        raise ColmapFailureError("cam_from_world contains non-finite values")
    return mat.tolist()


def _extract_camera_params(camera: Any) -> tuple[int, int, float, float, float, float]:
    width = int(camera.width)
    height = int(camera.height)
    model_obj = getattr(camera, "model", None)
    if model_obj is not None:
        model_name = getattr(model_obj, "name", str(model_obj)).split(".")[-1].upper()
        if model_name not in {"PINHOLE", "SIMPLE_PINHOLE"}:
            raise ColmapFailureError(f"Expected undistorted PINHOLE camera model, got {model_name}")
    else:
        model_name = "PINHOLE"

    if hasattr(camera, "params"):
        params = [float(v) for v in camera.params]
        if model_name == "SIMPLE_PINHOLE" or len(params) == 3:
            f, cx, cy = params[:3]
            return width, height, f, f, cx, cy
        if len(params) >= 4:
            fx, fy, cx, cy = params[:4]
            return width, height, fx, fy, cx, cy
    return width, height, float(camera.fx), float(camera.fy), float(camera.cx), float(camera.cy)


def _update_camera_params(camera: Any, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> None:
    camera.width = width
    camera.height = height
    if hasattr(camera, "params"):
        if len(camera.params) == 3:
            camera.params = np.array([fx, cx, cy], dtype=np.float64) if isinstance(camera.params, np.ndarray) else [fx, cx, cy]
        else:
            camera.params = np.array([fx, fy, cx, cy], dtype=np.float64) if isinstance(camera.params, np.ndarray) else [fx, fy, cx, cy]
    else:
        camera.fx, camera.fy, camera.cx, camera.cy = fx, fy, cx, cy


def _find_source_image(candidate_dirs: Sequence[Path], relpath: str) -> Path | None:
    target_name = Path(relpath).name
    target_stem = Path(relpath).stem
    for img_dir in candidate_dirs:
        if not img_dir.is_dir():
            continue
        exact = img_dir / target_name
        if exact.is_file():
            return exact
        for ext in (".png", ".jpg", ".jpeg"):
            alt = img_dir / f"{target_stem}{ext}"
            if alt.is_file():
                return alt
    return None


def _publish_mod16_images(
    candidate_dirs: Sequence[Path],
    output_dir: Path,
    frames: Sequence[dict[str, Any]],
    orig_width: int,
    orig_height: int,
    crop_box: tuple[int, int, int, int],
) -> None:
    needs_crop = crop_box != (0, 0, orig_width, orig_height)
    for fr in frames:
        relpath = str(fr.get("image_relpath") or "")
        if not relpath:
            continue
        src = _find_source_image(candidate_dirs, relpath)
        if src is None:
            continue
        dst = output_dir / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not needs_crop:
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            continue
        with Image.open(src) as handle:
            if handle.size == (orig_width, orig_height):
                handle.crop(crop_box).save(dst)
            elif src.resolve() != dst.resolve():
                shutil.copy2(src, dst)


def _crop_images_to_mod16(
    candidate_dirs: Sequence[Path],
    orig_width: int,
    orig_height: int,
    crop_box: tuple[int, int, int, int],
) -> None:
    for img_dir in candidate_dirs:
        if not img_dir.is_dir():
            continue
        cropped_any = False
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            with Image.open(img_path) as handle:
                if handle.size != (orig_width, orig_height):
                    continue
                cropped = handle.crop(crop_box)
                cropped.save(img_path)
                cropped_any = True
        if cropped_any:
            break


class PyColmapBackend:
    def __init__(self, mode: Literal["global", "incremental"]) -> None:
        self.mode = mode

    def estimate(self, request: PoseRequest) -> BackendOutput:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        pycolmap = _module("pycolmap")
        database = request.database_path or request.output_dir / "database.db"
        if database.resolve().parent == request.image_dir.resolve() or request.image_dir.resolve() in database.resolve().parents:
            raise StageExecutionError("database.db must be outside image_dir")
        database.parent.mkdir(parents=True, exist_ok=True)
        extraction = self._extract(pycolmap, database, request)
        raw_sparse = request.output_dir / "raw_sparse"
        if self.mode == "global":
            result = self._global(pycolmap, database, request, raw_sparse, extraction)
        else:
            result = self._incremental(pycolmap, database, request, raw_sparse, extraction)
        return self._coerce_result(result, request, raw_sparse)

    def _configure_options(
        cls: Any,
        updates: dict[str, Any],
        fallback: dict[str, Any],
        nested: tuple[str, str, Any] | None = None,
    ) -> Any:
        if cls is None:
            return dict(fallback)
        opts = cls()
        for attr, value in updates.items():
            if value is not None and hasattr(opts, attr):
                setattr(opts, attr, value)
        if nested is not None and hasattr(opts, nested[0]):
            sub = getattr(opts, nested[0])
            if hasattr(sub, nested[1]):
                setattr(sub, nested[1], nested[2])
        return opts

    def _extract(self, pycolmap: Any, database: Path, request: PoseRequest) -> Any:
        enums = (getattr(pycolmap, "FeatureExtractorType", None), getattr(pycolmap, "FeatureMatcherType", None))
        onnx_ready = all(enum is not None and hasattr(enum, name) for enum, name in zip(enums, ("ALIKED_N16ROT", "ALIKED_LIGHTGLUE")))
        if onnx_ready:
            extractor, matcher = enums[0].ALIKED_N16ROT, enums[1].ALIKED_LIGHTGLUE
        else:
            extractor, matcher = "DSP-SIFT", "sequential"

        camera_mode_enum = getattr(pycolmap, "CameraMode", None)
        single_camera_mode = getattr(camera_mode_enum, "SINGLE", None)

        reader_options = PyColmapBackend._configure_options(
            getattr(pycolmap, "ImageReaderOptions", None),
            {"camera_mode": single_camera_mode, "camera_model": "PINHOLE",
             "camera_params": ",".join(str(request.intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))},
            {"camera_mode": single_camera_mode or "SINGLE"},
        )
        extraction_options = PyColmapBackend._configure_options(
            getattr(pycolmap, "FeatureExtractionOptions", None),
            {"max_num_features": 4096, "num_threads": 4},
            {"feature_type": extractor, "max_num_features": 4096, "num_threads": 4},
            nested=("sift", "max_num_features", 4096),
        )
        extract_kwargs: dict[str, Any] = {
            "extraction_options": extraction_options,
            "reader_options": reader_options,
        }
        if single_camera_mode is not None:
            extract_kwargs["camera_mode"] = single_camera_mode
        _call_optional(pycolmap.extract_features, str(database), str(request.image_dir), **extract_kwargs)

        loop_det = bool(request.vocab_tree_path)
        matching_options = PyColmapBackend._configure_options(
            getattr(pycolmap, "FeatureMatchingOptions", None),
            {"num_threads": 4},
            {"matcher_type": matcher, "num_threads": 4},
        )
        pair_updates: dict[str, Any] = {"overlap": 12, "loop_detection": loop_det}
        if request.vocab_tree_path is not None:
            pair_updates["vocab_tree_path"] = str(request.vocab_tree_path)
        pairing_options = PyColmapBackend._configure_options(
            getattr(pycolmap, "SequentialPairingOptions", None),
            pair_updates,
            {"overlap": 12, "loop_detection": loop_det},
        )

        match_kwargs: dict[str, Any] = {
            "matching_options": matching_options,
            "pairing_options": pairing_options,
        }
        try:
            sig = inspect.signature(pycolmap.match_sequential)
            if "match_options" in sig.parameters:
                match_kwargs["match_options"] = {"overlap": 12, "loop_detection": loop_det, "matcher_type": matcher}
        except (ValueError, TypeError):
            pass
        return _call_optional(pycolmap.match_sequential, str(database), **match_kwargs)

    def _global(self, pycolmap: Any, database: Path, request: PoseRequest, sparse: Path, _: Any) -> Any:
        sparse.mkdir(parents=True, exist_ok=True)
        options = {"num_threads": 4}
        result = _call_optional(pycolmap.global_mapping, str(database), str(request.image_dir), str(sparse), options=options)
        return self._maybe_undistort(pycolmap, request, sparse, result)

    def _incremental(self, pycolmap: Any, database: Path, request: PoseRequest, sparse: Path, _: Any) -> Any:
        sparse.mkdir(parents=True, exist_ok=True)
        options = {"ba_use_gpu": True, "ba_global_frames_ratio": 1.25, "ba_global_max_num_iterations": 30}
        result = _call_optional(pycolmap.incremental_mapping, str(database), str(request.image_dir), str(sparse), options=options)
        return self._maybe_undistort(pycolmap, request, sparse, result)

    def _maybe_undistort(self, pycolmap: Any, request: PoseRequest, sparse: Path, result: Any) -> Any:
        undistort = getattr(pycolmap, "undistort_images", None)
        if undistort is None:
            return result
        input_sparse = sparse / "0" if (sparse / "0").is_dir() else sparse
        undistort_dir = request.output_dir / "undistorted"
        undistort_dir.mkdir(parents=True, exist_ok=True)
        _call_optional(undistort, str(undistort_dir), str(input_sparse), str(request.image_dir))
        undistorted_sparse = undistort_dir / "sparse" / "0" if (undistort_dir / "sparse" / "0").is_dir() else undistort_dir / "sparse"
        recon_cls = getattr(pycolmap, "Reconstruction", None)
        if recon_cls is not None and (undistorted_sparse / "cameras.bin").exists():
            return {0: recon_cls(str(undistorted_sparse))}
        return result

    def _coerce_result(self, value: Any, request: PoseRequest, sparse_dir: Path) -> BackendOutput:
        if isinstance(value, BackendOutput):
            return value
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            return BackendOutput(**value)
        if isinstance(value, dict) and value:
            best = max(value.values(), key=lambda rec: len(getattr(rec, "images", {})))
            return self._from_reconstruction(best, request, sparse_dir)
        raise StageExecutionError(f"Pose backend returned unsupported result: {type(value).__name__}")

    def _from_reconstruction(self, rec: Any, request: PoseRequest, sparse_dir: Path) -> BackendOutput:
        cameras = getattr(rec, "cameras", None) or {}
        if not cameras:
            raise ColmapFailureError("COLMAP reconstruction contains no cameras")
        images = getattr(rec, "images", None) or {}
        primary_cam = next(iter(cameras.values()))
        if images:
            first_img = next(iter(images.values()))
            cam_id = getattr(first_img, "camera_id", None)
            if cam_id in cameras:
                primary_cam = cameras[cam_id]

        orig_w, orig_h, fx, fy, cx, cy = _extract_camera_params(primary_cam)
        new_w, new_h, new_fx, new_fy, new_cx, new_cy, crop_box = mod16_crop_or_pad_params(
            orig_w, orig_h, cx, cy, fx, fy
        )
        if (new_w, new_h) != (orig_w, orig_h):
            _crop_images_to_mod16(
                [request.output_dir / "undistorted" / "images", sparse_dir.parent / "images", request.image_dir],
                orig_w,
                orig_h,
                crop_box,
            )
            for cam in cameras.values():
                _update_camera_params(cam, new_w, new_h, new_fx, new_fy, new_cx, new_cy)

        by_name = {Path(str(img.name)).name: img for img in images.values()}
        frames: list[dict[str, Any]] = []
        cam_positions: list[NDArray[np.float64]] = []
        registered_count = 0

        if request.frame_records:
            for fr in request.frame_records:
                record = dict(fr)
                fname = Path(str(fr["image_relpath"])).name
                matched = by_name.get(fname)
                if matched is not None:
                    pose_4x4 = _extract_cam_from_world_4x4(matched)
                    record["colmap_image_id"] = int(matched.image_id)
                    record["cam_from_world"] = pose_4x4
                    registered_count += 1
                    pose_arr = np.asarray(pose_4x4, dtype=np.float64)
                    cam_positions.append(-(pose_arr[:3, :3].T @ pose_arr[:3, 3]))
                else:
                    record["colmap_image_id"] = None
                    record["cam_from_world"] = None
                frames.append(record)
        else:
            for idx, img in enumerate(sorted(images.values(), key=lambda item: int(item.image_id))):
                pose_4x4 = _extract_cam_from_world_4x4(img)
                pose_arr = np.asarray(pose_4x4, dtype=np.float64)
                cam_positions.append(-(pose_arr[:3, :3].T @ pose_arr[:3, 3]))
                frames.append({
                    "frame_idx": idx,
                    "orig_video_frame": idx,
                    "timestamp_sec": float(idx),
                    "image_relpath": f"images/{Path(str(img.name)).name}",
                    "colmap_image_id": int(img.image_id),
                    "cam_from_world": pose_4x4,
                })
            registered_count = len(frames)

        if registered_count < 2:
            raise ColmapFailureError(f"COLMAP registered only {registered_count} frame(s); at least 2 are required")

        points_dict = getattr(rec, "points3D", None) or {}
        if not points_dict:
            raise ColmapFailureError("COLMAP reconstruction produced empty points3D")
        points = np.asarray([p.xyz for p in points_dict.values()], dtype=np.float64)
        points_rgb = None
        first_pt = next(iter(points_dict.values()))
        if hasattr(first_pt, "color"):
            points_rgb = np.asarray([p.color for p in points_dict.values()], dtype=np.uint8)

        calibrated = {"fx": new_fx, "fy": new_fy, "cx": new_cx, "cy": new_cy}
        _publish_mod16_images(
            [request.output_dir / "undistorted" / "images", sparse_dir.parent / "images", request.image_dir],
            request.output_dir,
            frames,
            orig_w,
            orig_h,
            crop_box,
        )
        sparse_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(rec, "write_binary"):
            rec.write_binary(str(sparse_dir))
        elif hasattr(rec, "write"):
            rec.write(str(sparse_dir))
        else:
            write_colmap_binary(sparse_dir, new_w, new_h, calibrated, frames, points, points_rgb)

        reproj = (
            float(rec.compute_mean_reprojection_error())
            if hasattr(rec, "compute_mean_reprojection_error") and points_dict
            else 0.0
        )
        return BackendOutput(
            registered_frames=registered_count,
            mean_reproj_px=reproj,
            num_points3d=len(points),
            points_xyz=points,
            cam_positions_xyz=np.asarray(cam_positions, dtype=np.float64),
            frames=frames,
            sparse_dir=sparse_dir,
            width=new_w,
            height=new_h,
            intrinsics=calibrated,
        )


class ColmapCliBackend(PyColmapBackend):
    """Use installed Linux COLMAP/GLOMAP tools and normalize through PyCOLMAP."""

    def __init__(self, mode: Literal["global", "incremental"], colmap: str) -> None:
        super().__init__(mode)
        self.colmap = colmap

    def _feature_prefix(self, command: str, modern: str, legacy: str) -> str:
        result = run_isolated_stage([self.colmap, command, "--help"], None, None, 30)
        return modern if f"--{modern}.num_threads" in result.stdout + result.stderr else legacy

    def _extract(self, pycolmap: Any, database: Path, request: PoseRequest) -> None:
        extraction = self._feature_prefix("feature_extractor", "FeatureExtraction", "SiftExtraction")
        run_isolated_stage([
            self.colmap, "feature_extractor", "--database_path", str(database),
            "--image_path", str(request.image_dir), "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "PINHOLE", "--ImageReader.camera_params",
            ",".join(str(request.intrinsics[k]) for k in ("fx", "fy", "cx", "cy")),
            f"--{extraction}.num_threads", "4", f"--{extraction}.use_gpu", "0",
            "--SiftExtraction.max_num_features", "4096",
        ], None, None, 3600)
        matching = self._feature_prefix("sequential_matcher", "FeatureMatching", "SiftMatching")
        command = [
            self.colmap, "sequential_matcher", "--database_path", str(database),
            f"--{matching}.num_threads", "4", f"--{matching}.use_gpu", "0",
            "--SequentialMatching.overlap", "12", "--SequentialMatching.loop_detection",
            "1" if request.vocab_tree_path else "0",
        ]
        if request.vocab_tree_path:
            command.extend(["--SequentialMatching.vocab_tree_path", str(request.vocab_tree_path)])
        run_isolated_stage(command, None, None, 3600)

    def _map(self, pycolmap: Any, database: Path, request: PoseRequest, sparse: Path) -> Any:
        sparse.mkdir(parents=True, exist_ok=True)
        glomap = shutil.which("glomap") if self.mode == "global" else None
        run_isolated_stage([
            glomap or self.colmap, "mapper", "--database_path", str(database),
            "--image_path", str(request.image_dir), "--output_path", str(sparse),
        ], None, None, 3600)
        models = [path.parent for path in sparse.glob("*/cameras.bin")]
        if not models:
            raise ColmapFailureError(f"Mapper produced no binary reconstruction under {sparse}")
        reconstructions = {path: pycolmap.Reconstruction(str(path)) for path in models}
        best = max(reconstructions, key=lambda path: len(reconstructions[path].images))
        return self._maybe_undistort(pycolmap, request, best, {0: reconstructions[best]})

    def _global(self, pycolmap: Any, database: Path, request: PoseRequest, sparse: Path, _: Any) -> Any:
        return self._map(pycolmap, database, request, sparse)

    def _incremental(self, pycolmap: Any, database: Path, request: PoseRequest, sparse: Path, _: Any) -> Any:
        return self._map(pycolmap, database, request, sparse)


class VGGTRunner(Protocol):
    def infer(self, images: Sequence[Path], *, dtype: str) -> BackendOutput: ...


def _frame_match_key(frame: dict[str, Any], fallback_idx: int) -> str | int:
    relpath = frame.get("image_relpath")
    if relpath:
        return Path(str(relpath)).name
    if "frame_idx" in frame:
        return int(frame["frame_idx"])
    return fallback_idx


def _ordered_request_images(request: PoseRequest) -> list[Path]:
    if request.frame_records:
        mapped: list[Path] = []
        for fr in request.frame_records:
            candidate = request.image_dir / Path(str(fr["image_relpath"])).name
            if not candidate.exists():
                break
            mapped.append(candidate)
        if len(mapped) == len(request.frame_records):
            return mapped
    return sorted(path for path in request.image_dir.iterdir() if path.is_file())


def _reconcile_chunk_frames(
    raw_frames: list[dict[str, Any]],
    chunk_images: Sequence[Path],
    records_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    reconciled: list[dict[str, Any]] = []
    for idx, fr in enumerate(raw_frames):
        fr_copy = dict(fr)
        rel = fr_copy.get("image_relpath")
        name = Path(str(rel)).name if rel else (chunk_images[idx].name if idx < len(chunk_images) else "")
        if name not in records_by_name and idx < len(chunk_images) and chunk_images[idx].name in records_by_name:
            name = chunk_images[idx].name
        matched = records_by_name.get(name)
        if matched is not None:
            merged = dict(matched)
            merged.update(fr_copy)
            merged["frame_idx"] = int(matched["frame_idx"])
            merged["image_relpath"] = str(matched["image_relpath"])
            if "orig_video_frame" in matched:
                merged["orig_video_frame"] = int(matched["orig_video_frame"])
            if "timestamp_sec" in matched:
                merged["timestamp_sec"] = float(matched["timestamp_sec"])
            fr_copy = merged
        elif idx < len(chunk_images) and not rel:
            fr_copy["image_relpath"] = f"images/{chunk_images[idx].name}"
        reconciled.append(fr_copy)
    return reconciled


class VGGTChunkedBackend:
    def __init__(self, runner: VGGTRunner, chunk_size: int = 32, overlap: int = 6) -> None:
        if chunk_size < 2:
            raise ValueError(f"chunk_size must be >= 2, got {chunk_size}")
        if not (0 < overlap < chunk_size):
            raise ValueError(f"overlap must satisfy 0 < overlap < chunk_size, got overlap={overlap}, chunk_size={chunk_size}")
        self.runner = runner
        self.chunk_size = chunk_size
        self.overlap = overlap

    def estimate(self, request: PoseRequest) -> BackendOutput:
        images = _ordered_request_images(request)
        if not images:
            raise StageExecutionError("VGGT received no images")
        records_by_name = {
            Path(str(fr["image_relpath"])).name: fr
            for fr in request.frame_records
            if fr.get("image_relpath")
        }
        chunks: list[BackendOutput] = []
        step = self.chunk_size - self.overlap
        for start in range(0, len(images), step):
            chunk_images = images[start : start + self.chunk_size]
            raw = self.runner.infer(chunk_images, dtype="fp16")
            raw.frames = _reconcile_chunk_frames([dict(fr) for fr in raw.frames], chunk_images, records_by_name)
            raw.points_xyz = np.asarray(raw.points_xyz, dtype=np.float64).copy()
            raw.cam_positions_xyz = np.asarray(raw.cam_positions_xyz, dtype=np.float64).copy()
            if chunks:
                self._align_chunk(chunks[-1], raw)
            chunks.append(raw)
            if start + self.chunk_size >= len(images):
                break

        seen_keys: set[str | int] = set()
        merged_frames: list[dict[str, Any]] = []
        merged_cam_positions: list[NDArray[np.float64]] = []
        merged_points_parts: list[NDArray[np.float64]] = []
        depth_by_frame: dict[int, NDArray[np.float32]] = {}
        reproj_weighted = 0.0

        for chunk in chunks:
            if len(chunk.points_xyz):
                merged_points_parts.append(chunk.points_xyz)
            reproj_weighted += float(chunk.mean_reproj_px)
            chunk_depth_dict = chunk.depths if isinstance(chunk.depths, dict) else None
            for idx, fr in enumerate(chunk.frames):
                key = _frame_match_key(fr, idx)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                fr_copy = dict(fr)
                frame_idx = int(fr_copy.get("frame_idx", len(merged_frames)))
                if fr_copy.get("colmap_image_id") is None:
                    fr_copy["colmap_image_id"] = len(merged_frames) + 1
                merged_frames.append(fr_copy)
                if idx < len(chunk.cam_positions_xyz):
                    merged_cam_positions.append(chunk.cam_positions_xyz[idx])
                if chunk_depth_dict is not None:
                    d_key = str(key) if str(key) in chunk_depth_dict else str(frame_idx)
                    if d_key in chunk_depth_dict:
                        depth_by_frame[frame_idx] = np.asarray(chunk_depth_dict[d_key], dtype=np.float32)
                    elif frame_idx in chunk_depth_dict:
                        depth_by_frame[frame_idx] = np.asarray(chunk_depth_dict[frame_idx], dtype=np.float32)
                elif chunk.depths is not None and idx < len(chunk.depths):
                    depth_by_frame[frame_idx] = np.asarray(chunk.depths[idx], dtype=np.float32)

        merged_points = np.vstack(merged_points_parts) if merged_points_parts else np.empty((0, 3), dtype=np.float64)
        merged_cams = np.asarray(merged_cam_positions, dtype=np.float64) if merged_cam_positions else np.empty((0, 3), dtype=np.float64)
        first = chunks[0]
        width = first.width or request.width
        height = first.height or request.height
        intr = first.intrinsics or request.intrinsics
        new_w, new_h, new_fx, new_fy, new_cx, new_cy, crop_box = mod16_crop_or_pad_params(
            width, height, intr["cx"], intr["cy"], intr["fx"], intr["fy"]
        )
        calibrated = {"fx": new_fx, "fy": new_fy, "cx": new_cx, "cy": new_cy}
        _publish_mod16_images([request.image_dir], request.output_dir, merged_frames, width, height, crop_box)

        out_depth_dir: Path | None = request.depth_dir
        final_depths: Any = None
        if depth_by_frame:
            out_depth_dir = request.depth_dir or (request.output_dir / "depths")
            out_depth_dir.mkdir(parents=True, exist_ok=True)
            x0, y0, x1, y1 = crop_box
            cropped_list: list[NDArray[np.float32]] = []
            for fr in merged_frames:
                f_idx = int(fr.get("frame_idx", 0))
                if f_idx not in depth_by_frame:
                    continue
                depth_arr = depth_by_frame[f_idx]
                if (new_w, new_h) != (width, height) and depth_arr.shape == (height, width):
                    depth_arr = depth_arr[y0:y1, x0:x1]
                depth_arr = np.asarray(depth_arr, dtype=np.float32)
                np.save(out_depth_dir / f"{f_idx:06d}.npy", depth_arr, allow_pickle=False)
                cropped_list.append(depth_arr)
            if cropped_list:
                final_depths = np.stack(cropped_list, axis=0)

        sparse_dir = request.output_dir / "sparse" / "0"
        write_colmap_binary(sparse_dir, new_w, new_h, calibrated, merged_frames, merged_points)

        return BackendOutput(
            registered_frames=len(merged_frames),
            mean_reproj_px=reproj_weighted / len(chunks),
            num_points3d=len(merged_points),
            points_xyz=merged_points,
            cam_positions_xyz=merged_cams,
            frames=merged_frames,
            sparse_dir=sparse_dir,
            depth_dir=out_depth_dir,
            width=new_w,
            height=new_h,
            intrinsics=calibrated,
            depths=final_depths,
        )

    def _align_chunk(self, reference: BackendOutput, chunk: BackendOutput) -> None:
        prev_index_by_key = {_frame_match_key(fr, idx): idx for idx, fr in enumerate(reference.frames)}
        matched_pairs: list[tuple[int, int]] = []
        for cur_idx, fr in enumerate(chunk.frames):
            key = _frame_match_key(fr, cur_idx)
            if key in prev_index_by_key:
                matched_pairs.append((prev_index_by_key[key], cur_idx))

        if len(matched_pairs) < 3:
            raise GeometryError(
                f"VGGT chunk overlap needs at least three matched frames for Sim(3), got {len(matched_pairs)}"
            )

        dst_centers = reference.cam_positions_xyz[[prev_i for prev_i, _ in matched_pairs]]
        src_centers = chunk.cam_positions_xyz[[cur_i for _, cur_i in matched_pairs]]
        sim = umeyama_sim3(src_centers, dst_centers)

        if len(chunk.points_xyz):
            chunk.points_xyz = sim.apply(chunk.points_xyz)
        if len(chunk.cam_positions_xyz):
            chunk.cam_positions_xyz = sim.apply(chunk.cam_positions_xyz)
        for fr in chunk.frames:
            if fr.get("cam_from_world") is not None:
                transformed = transform_cam_from_world_sim3(np.asarray(fr["cam_from_world"], dtype=np.float64), sim)
                fr["cam_from_world"] = transformed.tolist()
        if chunk.depths is not None:
            if isinstance(chunk.depths, dict):
                chunk.depths = {k: np.asarray(v, dtype=np.float32) * np.float32(sim.scale) for k, v in chunk.depths.items()}
            else:
                chunk.depths = np.asarray(chunk.depths, dtype=np.float32) * np.float32(sim.scale)
        if chunk.depth_dir is not None and Path(chunk.depth_dir).is_dir():
            for npy_path in Path(chunk.depth_dir).glob("*.npy"):
                scaled = np.load(npy_path, allow_pickle=False).astype(np.float32) * np.float32(sim.scale)
                np.save(npy_path, scaled, allow_pickle=False)


class MapAnythingBackend:
    def __init__(self, runner: VGGTRunner) -> None:
        self.runner = runner

    def estimate(self, request: PoseRequest) -> BackendOutput:
        return self.runner.infer(sorted(request.image_dir.glob("*")), dtype="fp16")


def _coerce_backend_output(value: Any, request: PoseRequest) -> BackendOutput:
    return PyColmapBackend("global")._coerce_result(value, request, request.output_dir / "raw_sparse")


def _copy_sparse(source: Path | None, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if source is None:
        return
    source = source / "0" if (source / "0").is_dir() else source
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        candidate = source / name
        destination = target / name
        if candidate.exists() and candidate.resolve() != destination.resolve():
            shutil.copy2(candidate, destination)


def _as_cam_from_world_4x4(mat: Any) -> NDArray[np.float64]:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = arr
        return out
    raise StageExecutionError(f"Expected 3x4 or 4x4 extrinsic matrix from VGGT, got shape {arr.shape}")


def _to_numpy(value: Any) -> NDArray[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_4d_depth(d_arr: NDArray[np.float32], expected_s: int) -> NDArray[np.float32]:
    if expected_s > 0:
        if d_arr.shape[-1] == 1 and d_arr.shape[0] == expected_s and d_arr.shape[2] > 1:
            return d_arr[..., 0]
        if d_arr.shape[0] == 1 and d_arr.shape[1] == expected_s:
            return d_arr[0]
    if d_arr.shape[-1] == 1:
        return d_arr[..., 0]
    if d_arr.shape[0] == 1:
        return d_arr[0]
    raise StageExecutionError(f"Expected 4D VGGT depth shape [1, S, H, W] or [S, H, W, 1], got {d_arr.shape}")


def _normalize_depth_stack(d_arr: NDArray[np.float32], expected_s: int) -> NDArray[np.float32]:
    if d_arr.ndim == 5:
        if d_arr.shape[0] != 1 or d_arr.shape[-1] != 1:
            raise StageExecutionError(f"Expected 5D VGGT depth shape [1, S, H, W, 1], got {d_arr.shape}")
        d_arr = d_arr[0, ..., 0]
    elif d_arr.ndim == 4:
        d_arr = _normalize_4d_depth(d_arr, expected_s)
    elif d_arr.ndim == 2:
        d_arr = d_arr[None, ...]
    if d_arr.ndim != 3 or (expected_s > 0 and d_arr.shape[0] != expected_s):
        raise StageExecutionError(f"Expected VGGT depth stack [S={expected_s}, H, W], got {d_arr.shape}")
    return np.ascontiguousarray(d_arr, dtype=np.float32)


def _vggt_preprocess_geometry(
    orig_w: int,
    orig_h: int,
    in_w: int,
    in_h: int,
    raw: Mapping[str, Any] | None = None,
) -> tuple[int, int, int, int]:
    if (in_w, in_h) == (orig_w, orig_h) and not (
        raw and any(k in raw for k in ("resized_width", "resized_height", "crop_x", "crop_y"))
    ):
        return orig_w, orig_h, 0, 0
    patch = 14 if (in_w % 14 == 0 and in_h % 14 == 0) else 1
    cand_h = max(patch, int(round((orig_h * (in_w / float(orig_w))) / patch)) * patch)
    if cand_h >= in_h:
        resized_w, resized_h = in_w, cand_h
        crop_x, crop_y = 0, (cand_h - in_h) // 2
    else:
        cand_w = max(patch, int(round((orig_w * (in_h / float(orig_h))) / patch)) * patch)
        resized_w, resized_h = max(in_w, cand_w), in_h
        crop_x, crop_y = (resized_w - in_w) // 2, 0
    if raw is not None:
        if "resized_width" in raw:
            resized_w = int(raw["resized_width"])
        if "resized_height" in raw:
            resized_h = int(raw["resized_height"])
        crop_x = int(raw.get("crop_x", max(0, (resized_w - in_w) // 2)))
        crop_y = int(raw.get("crop_y", max(0, (resized_h - in_h) // 2)))
    return resized_w, resized_h, crop_x, crop_y


def _resize_depths(depths: NDArray[np.float32], target_w: int, target_h: int) -> NDArray[np.float32]:
    if depths.ndim == 2:
        depths = depths[None, ...]
    if depths.shape[-2:] == (target_h, target_w):
        return depths
    resized = [
        np.asarray(Image.fromarray(slice_2d, mode="F").resize((target_w, target_h), Image.Resampling.BILINEAR), dtype=np.float32)
        for slice_2d in depths
    ]
    return np.stack(resized, axis=0)


def _unpreprocess_depths(
    depths: NDArray[np.float32],
    orig_w: int,
    orig_h: int,
    resized_w: int,
    resized_h: int,
    crop_x: int,
    crop_y: int,
) -> NDArray[np.float32]:
    if crop_x == 0 and crop_y == 0:
        return _resize_depths(depths, orig_w, orig_h)
    in_h, in_w = int(depths.shape[-2]), int(depths.shape[-1])
    scale_x = orig_w / float(resized_w)
    scale_y = orig_h / float(resized_h)
    x0 = max(0, min(orig_w, int(round(crop_x * scale_x))))
    x1 = max(x0 + 1, min(orig_w, int(round((crop_x + in_w) * scale_x))))
    y0 = max(0, min(orig_h, int(round(crop_y * scale_y))))
    y1 = max(y0 + 1, min(orig_h, int(round((crop_y + in_h) * scale_y))))
    out = np.zeros((depths.shape[0], orig_h, orig_w), dtype=np.float32)
    out[:, y0:y1, x0:x1] = _resize_depths(depths, x1 - x0, y1 - y0)
    return out


def _coerce_vggt_output(
    raw: Any,
    images: Sequence[Path],
    *,
    input_size: tuple[int, int] | None = None,
    frame_records: Sequence[dict[str, Any]] | None = None,
) -> BackendOutput:
    if isinstance(raw, BackendOutput):
        return raw
    if not isinstance(raw, dict):
        raise StageExecutionError(f"Unsupported VGGT output type: {type(raw).__name__}")

    with Image.open(images[0]) as first_im:
        orig_w, orig_h = first_im.size

    depths_raw = raw.get("depths", raw.get("depth"))
    depths: NDArray[np.float32] | None = None
    if depths_raw is not None:
        depths = _normalize_depth_stack(_to_numpy(depths_raw).astype(np.float32), expected_s=len(images))

    if input_size is not None:
        in_w, in_h = int(input_size[0]), int(input_size[1])
    elif "width" in raw and "height" in raw:
        in_w, in_h = int(raw["width"]), int(raw["height"])
    elif depths is not None:
        in_h, in_w = int(depths.shape[-2]), int(depths.shape[-1])
    else:
        in_w, in_h = orig_w, orig_h

    extrinsics_raw = raw.get("extrinsics")
    raw_frames = raw.get("frames")
    frames: list[dict[str, Any]] = []
    cam_positions: list[NDArray[np.float64]] = []
    if raw_frames is not None:
        for idx, fr in enumerate(raw_frames):
            fr_dict = dict(fr)
            fr_dict.setdefault("frame_idx", idx)
            fr_dict.setdefault("image_relpath", f"images/{images[idx].name}" if idx < len(images) else f"images/{idx:06d}.png")
            c2w_or_w2c = fr_dict.get("cam_from_world")
            if c2w_or_w2c is not None:
                mat = _as_cam_from_world_4x4(c2w_or_w2c)
                fr_dict["cam_from_world"] = mat.tolist()
                cam_positions.append((-mat[:3, :3].T @ mat[:3, 3]).astype(np.float64))
            frames.append(fr_dict)
    elif extrinsics_raw is not None:
        ext_arr = _to_numpy(extrinsics_raw)
        if ext_arr.ndim == 4:
            ext_arr = ext_arr[0]
        for idx in range(len(images)):
            mat = _as_cam_from_world_4x4(ext_arr[idx])
            cam_positions.append((-mat[:3, :3].T @ mat[:3, 3]).astype(np.float64))
            frames.append({
                "frame_idx": idx,
                "image_relpath": f"images/{images[idx].name}",
                "cam_from_world": mat.tolist(),
            })
    else:
        raise StageExecutionError("VGGT output missing both 'frames' and 'extrinsics'")

    if frame_records:
        records_by_name = {
            Path(str(fr["image_relpath"])).name: fr
            for fr in frame_records
            if fr.get("image_relpath")
        }
        frames = _reconcile_chunk_frames(frames, images, records_by_name)

    intr_raw = raw.get("intrinsics")
    intrinsics: dict[str, float]
    if isinstance(intr_raw, dict):
        intrinsics = {k: float(v) for k, v in intr_raw.items()}
    elif intr_raw is not None:
        k_arr = _to_numpy(intr_raw)
        while k_arr.ndim > 2:
            k_arr = k_arr[0]
        intrinsics = {
            "fx": float(k_arr[0, 0]),
            "fy": float(k_arr[1, 1]),
            "cx": float(k_arr[0, 2]),
            "cy": float(k_arr[1, 2]),
        }
    else:
        focal = float(max(in_w, in_h))
        intrinsics = {"fx": focal, "fy": focal, "cx": in_w / 2.0, "cy": in_h / 2.0}

    resized_w, resized_h, crop_x, crop_y = _vggt_preprocess_geometry(orig_w, orig_h, in_w, in_h, raw)
    if (resized_w, resized_h) != (orig_w, orig_h) or crop_x != 0 or crop_y != 0:
        scale_x = orig_w / float(resized_w)
        scale_y = orig_h / float(resized_h)
        intrinsics = {
            "fx": intrinsics["fx"] * scale_x,
            "fy": intrinsics["fy"] * scale_y,
            "cx": (intrinsics["cx"] + crop_x) * scale_x,
            "cy": (intrinsics["cy"] + crop_y) * scale_y,
        }
        if depths is not None:
            depths = _unpreprocess_depths(depths, orig_w, orig_h, resized_w, resized_h, crop_x, crop_y)

    pts_raw = raw.get("points_xyz", raw.get("world_points"))
    points_xyz = np.empty((0, 3), dtype=np.float64) if pts_raw is None else _to_numpy(pts_raw).reshape(-1, 3).astype(np.float64)
    cams_xyz = (
        np.asarray(raw["cam_positions_xyz"], dtype=np.float64)
        if raw.get("cam_positions_xyz") is not None
        else (np.asarray(cam_positions, dtype=np.float64) if cam_positions else np.empty((0, 3), dtype=np.float64))
    )
    return BackendOutput(
        registered_frames=len(frames),
        mean_reproj_px=float(raw.get("mean_reproj_px", 0.5)),
        num_points3d=int(raw.get("num_points3d", len(points_xyz))),
        points_xyz=points_xyz,
        cam_positions_xyz=cams_xyz,
        frames=frames,
        width=orig_w,
        height=orig_h,
        intrinsics=intrinsics,
        depths=depths,
    )


def _resolve_vggt_cls(vggt_mod: Any) -> Any:
    cls = getattr(vggt_mod, "VGGT", None)
    if cls is not None:
        return cls
    try:
        sub = importlib.import_module("vggt.models.vggt")
        return getattr(sub, "VGGT", None)
    except ImportError:
        return None


def _run_vggt_pretrained(vggt_mod: Any, vggt_cls: Any, model_id: str, images: Sequence[Path], dtype: str) -> BackendOutput:
    torch = importlib.import_module("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if (dtype == "fp16" and device == "cuda") else torch.float32
    model = vggt_cls.from_pretrained(model_id).to(device)
    model.eval()
    load_fn = getattr(vggt_mod, "load_and_preprocess_images", None)
    if load_fn is None:
        try:
            load_mod = importlib.import_module("vggt.utils.load_fn")
            load_fn = getattr(load_mod, "load_and_preprocess_images", None)
        except ImportError:
            load_fn = None
    if load_fn is None:
        arrs = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 for p in images]
        inputs = torch.from_numpy(np.stack(arrs, axis=0)).permute(0, 3, 1, 2).to(device)
    else:
        inputs = load_fn([str(p) for p in images]).to(device)
    in_h, in_w = int(inputs.shape[-2]), int(inputs.shape[-1])
    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast("cuda", dtype=torch_dtype):
                predictions = model(inputs)
        else:
            predictions = model(inputs)
    if isinstance(predictions, dict) and "pose_enc" in predictions and "extrinsics" not in predictions:
        pose_mod = importlib.import_module("vggt.utils.pose_enc")
        pose_to_ext = getattr(pose_mod, "pose_encoding_to_extri_intri")
        extri, intri = pose_to_ext(predictions["pose_enc"], (in_h, in_w))
        predictions = dict(predictions)
        predictions["extrinsics"] = extri
        predictions["intrinsics"] = intri
    return _coerce_vggt_output(predictions, images, input_size=(in_w, in_h))


class _DefaultVGGTRunner:
    def infer(self, images: Sequence[Path], *, dtype: str) -> BackendOutput:
        vggt_mod = _module("vggt")
        infer_chunk = getattr(vggt_mod, "infer_chunk", None)
        if callable(infer_chunk):
            return _coerce_vggt_output(infer_chunk(list(images), dtype=dtype), images)

        model_id = os.getenv("ROOM3D_VGGT_MODEL")
        vggt_cls = _resolve_vggt_cls(vggt_mod) if model_id else None
        if model_id and vggt_cls is not None and callable(getattr(vggt_cls, "from_pretrained", None)):
            return _run_vggt_pretrained(vggt_mod, vggt_cls, model_id, images, dtype)

        raise StageExecutionError(
            "VGGT model weights are not configured for standalone vggt_chunked inference; "
            "set ROOM3D_VGGT_MODEL, provide vggt.infer_chunk, or inject a VGGTRunner."
        )


_COLMAP_CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}


def _qvec_to_rotmat(qvec: tuple[float, float, float, float]) -> NDArray[np.float64]:
    q = np.asarray(qvec, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ColmapFailureError("COLMAP quaternion has zero or non-finite norm")
    w, x, y, z = (q / norm).tolist()
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _read_colmap_cameras_bin(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise ColmapFailureError(f"Missing COLMAP cameras.bin at {path}")
    data = path.read_bytes()
    if len(data) < 8:
        raise ColmapFailureError(f"Truncated COLMAP cameras.bin at {path}")
    header_size = struct.calcsize("<IiQQ")
    num_cameras = struct.unpack_from("<Q", data, 0)[0]
    offset = 8
    cameras: dict[int, dict[str, Any]] = {}
    for _ in range(num_cameras):
        if offset + header_size > len(data):
            raise ColmapFailureError(f"Truncated camera header in {path}")
        cam_id, model_id, width, height = struct.unpack_from("<IiQQ", data, offset)
        offset += header_size
        spec = _COLMAP_CAMERA_MODELS.get(model_id)
        if spec is None:
            raise ColmapFailureError(f"Unsupported COLMAP camera model_id={model_id} in {path}")
        model_name, num_params = spec
        param_bytes = num_params * 8
        if offset + param_bytes > len(data):
            raise ColmapFailureError(f"Truncated camera params in {path}")
        params = list(struct.unpack_from(f"<{num_params}d", data, offset))
        offset += param_bytes
        if model_name in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            fx = fy = float(params[0])
            cx, cy = float(params[1]), float(params[2])
            dist_coeffs = [float(v) for v in params[3:]]
        else:
            fx, fy, cx, cy = (float(params[i]) for i in range(4))
            dist_coeffs = [float(v) for v in params[4:]]
        if width <= 0 or height <= 0 or not all(np.isfinite(val) for val in params) or min(fx, fy) <= 0.0:
            raise ColmapFailureError(f"Camera {cam_id} has non-finite or non-positive parameters in {path}")
        if any((not np.isfinite(coeff)) or abs(coeff) > 1e-8 for coeff in dist_coeffs):
            raise ColmapFailureError(
                f"Camera {cam_id} ({model_name}) has non-zero distortion {dist_coeffs}; undistorted PINHOLE input is required"
            )
        cameras[int(cam_id)] = {
            "camera_id": int(cam_id),
            "model_id": int(model_id),
            "model": model_name,
            "width": int(width),
            "height": int(height),
            "params": params,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }
    return cameras


def _read_colmap_images_bin(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise ColmapFailureError(f"Missing COLMAP images.bin at {path}")
    data = path.read_bytes()
    if len(data) < 8:
        raise ColmapFailureError(f"Truncated COLMAP images.bin at {path}")
    header_size = struct.calcsize("<I4d3dI")
    num_images = struct.unpack_from("<Q", data, 0)[0]
    offset = 8
    images: dict[int, dict[str, Any]] = {}
    for _ in range(num_images):
        if offset + header_size > len(data):
            raise ColmapFailureError(f"Truncated image header in {path}")
        image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack_from("<I4d3dI", data, offset)
        offset += header_size
        try:
            nul = data.index(0, offset)
        except ValueError as exc:
            raise ColmapFailureError(f"Unterminated image name in {path}") from exc
        name = data[offset:nul].decode("utf-8")
        offset = nul + 1
        if offset + 8 > len(data):
            raise ColmapFailureError(f"Truncated point2D count in {path}")
        num_pts2d = struct.unpack_from("<Q", data, offset)[0]
        offset += 8 + num_pts2d * 24
        if offset > len(data):
            raise ColmapFailureError(f"Truncated point2D entries in {path}")
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _qvec_to_rotmat((qw, qx, qy, qz))
        pose[:3, 3] = [tx, ty, tz]
        images[int(image_id)] = {
            "image_id": int(image_id),
            "camera_id": int(camera_id),
            "name": name,
            "cam_from_world": pose.tolist(),
        }
    return images


def _read_colmap_points3d_bin(path: Path) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    if not path.is_file():
        raise ColmapFailureError(f"Missing COLMAP points3D.bin at {path}")
    data = path.read_bytes()
    if len(data) < 8:
        raise ColmapFailureError(f"Truncated COLMAP points3D.bin at {path}")
    header_size = struct.calcsize("<Q3d3BdQ")
    num_pts = struct.unpack_from("<Q", data, 0)[0]
    offset = 8
    xyz_list: list[tuple[float, float, float]] = []
    rgb_list: list[tuple[int, int, int]] = []
    for _ in range(num_pts):
        if offset + header_size > len(data):
            raise ColmapFailureError(f"Truncated point3D header in {path}")
        _, x, y, z, r, g, b, _, track_len = struct.unpack_from("<Q3d3BdQ", data, offset)
        offset += header_size + track_len * 8
        if offset > len(data):
            raise ColmapFailureError(f"Truncated point3D track in {path}")
        xyz_list.append((x, y, z))
        rgb_list.append((r, g, b))
    if not xyz_list:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    return np.asarray(xyz_list, dtype=np.float64), np.asarray(rgb_list, dtype=np.uint8)


def _write_sparse_points_json(path: Path, points_xyz: NDArray[np.float64], points_rgb: NDArray[np.uint8]) -> None:
    step = max(1, len(points_xyz) // 100_000)
    payload = [
        {
            "position": [float(v) for v in pt],
            "x": float(pt[0]),
            "y": float(pt[1]),
            "z": float(pt[2]),
            "color": [int(c) for c in clr],
        }
        for pt, clr in zip(points_xyz[::step], points_rgb[::step])
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def _scale_and_crop_intrinsics(
    old_w: int,
    old_h: int,
    fx_old: float,
    fy_old: float,
    cx_old: float,
    cy_old: float,
    target_w: int,
    target_h: int,
) -> tuple[int, int, float, float, float, float, tuple[int, int, int, int]]:
    scale_x = target_w / float(old_w)
    scale_y = target_h / float(old_h)
    return mod16_crop_or_pad_params(
        target_w,
        target_h,
        cx_old * scale_x,
        cy_old * scale_y,
        fx_old * scale_x,
        fy_old * scale_y,
    )


def _resolve_single_calibration_camera(
    cameras: dict[int, dict[str, Any]],
    sorted_imgs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    referenced_ids: list[int] = []
    seen: set[int] = set()
    for img in sorted_imgs:
        cid = int(img["camera_id"])
        if cid not in cameras:
            raise ColmapFailureError(f"Image {img.get('name', img.get('image_id'))} references missing camera_id={cid}")
        if cid not in seen:
            seen.add(cid)
            referenced_ids.append(cid)
    if not referenced_ids:
        return next(iter(cameras.values()))
    ref = cameras[referenced_ids[0]]
    for cid in referenced_ids[1:]:
        other = cameras[cid]
        if int(other["width"]) != int(ref["width"]) or int(other["height"]) != int(ref["height"]):
            raise ColmapFailureError(
                "Precomputed COLMAP scene contains multiple cameras with incompatible intrinsics or dimensions"
            )
        if any(abs(float(other[k]) - float(ref[k])) > 1e-4 for k in ("fx", "fy", "cx", "cy")):
            raise ColmapFailureError(
                "Precomputed COLMAP scene contains multiple cameras with incompatible intrinsics or dimensions"
            )
    return ref


def _resolve_precomputed_image(image_root: Path, colmap_name: str) -> Path:
    candidates = [image_root / colmap_name, image_root / Path(colmap_name).name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ColmapFailureError(f"Missing precomputed COLMAP source image {colmap_name} under {image_root}")


class PrecomputedColmapBackend:
    """Adapter for datasets that already provide sparse/0/{cameras,images,points3D}.bin."""

    def _locate_sparse_dir(self, request: PoseRequest) -> Path:
        candidates = (
            request.image_dir / "sparse" / "0",
            request.image_dir.parent / "sparse" / "0",
            request.output_dir / "sparse" / "0",
        )
        for cand in candidates:
            if (cand / "cameras.bin").is_file() and (cand / "images.bin").is_file() and (cand / "points3D.bin").is_file():
                return cand
        raise ColmapFailureError(f"Could not find sparse/0 COLMAP binary files near {request.image_dir}")

    def _detect_image_size(self, request: PoseRequest, fallback: tuple[int, int]) -> tuple[int, int]:
        candidates = _ordered_request_images(request) if request.image_dir.is_dir() else []
        for path in candidates:
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                with Image.open(path) as handle:
                    return handle.size
        return fallback

    def estimate(self, request: PoseRequest) -> BackendOutput:
        src_sparse = self._locate_sparse_dir(request)
        cameras = _read_colmap_cameras_bin(src_sparse / "cameras.bin")
        images_bin = _read_colmap_images_bin(src_sparse / "images.bin")
        points_xyz, points_rgb = _read_colmap_points3d_bin(src_sparse / "points3D.bin")
        if not cameras or len(images_bin) < 2 or len(points_xyz) == 0:
            raise ColmapFailureError("Precomputed COLMAP reconstruction requires >=2 images and non-empty points3D")

        sorted_imgs = sorted(images_bin.values(), key=lambda item: int(item["image_id"]))
        cam = _resolve_single_calibration_camera(cameras, sorted_imgs)
        actual_w, actual_h = self._detect_image_size(request, (request.width, request.height))
        final_w, final_h, final_fx, final_fy, final_cx, final_cy, crop_box = _scale_and_crop_intrinsics(
            int(cam["width"]),
            int(cam["height"]),
            float(cam["fx"]),
            float(cam["fy"]),
            float(cam["cx"]),
            float(cam["cy"]),
            actual_w,
            actual_h,
        )
        calibrated = {"fx": final_fx, "fy": final_fy, "cx": final_cx, "cy": final_cy}

        by_name = {Path(str(img["name"])).name: img for img in sorted_imgs}
        frames: list[dict[str, Any]] = []
        cam_positions: list[NDArray[np.float64]] = []
        registered_count = 0
        if request.frame_records:
            for fr in request.frame_records:
                rec = dict(fr)
                matched = by_name.get(Path(str(fr["image_relpath"])).name)
                if matched is not None:
                    pose_arr = np.asarray(matched["cam_from_world"], dtype=np.float64)
                    cam_positions.append(-(pose_arr[:3, :3].T @ pose_arr[:3, 3]))
                    rec["colmap_image_id"] = int(matched["image_id"])
                    rec["cam_from_world"] = matched["cam_from_world"]
                    registered_count += 1
                else:
                    rec["colmap_image_id"] = None
                    rec["cam_from_world"] = None
                frames.append(rec)
        else:
            for idx, img in enumerate(sorted_imgs):
                pose_arr = np.asarray(img["cam_from_world"], dtype=np.float64)
                cam_positions.append(-(pose_arr[:3, :3].T @ pose_arr[:3, 3]))
                frames.append({
                    "frame_idx": idx,
                    "orig_video_frame": idx,
                    "timestamp_sec": float(idx),
                    "image_relpath": f"images/{Path(str(img['name'])).name}",
                    "colmap_image_id": int(img["image_id"]),
                    "cam_from_world": img["cam_from_world"],
                })
            registered_count = len(frames)

        if registered_count < 2:
            raise ColmapFailureError(f"Precomputed COLMAP matched only {registered_count} frame(s); >=2 required")

        _publish_mod16_images([request.image_dir], request.output_dir, frames, actual_w, actual_h, crop_box)
        sparse_dir = request.output_dir / "sparse" / "0"
        write_colmap_binary(sparse_dir, final_w, final_h, calibrated, frames, points_xyz, points_rgb)
        _write_sparse_points_json(request.output_dir / "sparse_points.json", points_xyz, points_rgb)

        return BackendOutput(
            registered_frames=registered_count,
            mean_reproj_px=0.5,
            num_points3d=len(points_xyz),
            points_xyz=points_xyz,
            cam_positions_xyz=np.asarray(cam_positions, dtype=np.float64),
            frames=frames,
            sparse_dir=sparse_dir,
            width=final_w,
            height=final_h,
            intrinsics=calibrated,
        )


def prepare_precomputed_colmap_scene(
    source_scene_dir: Path,
    output_dir: Path,
    *,
    project_id: str,
    version_id: str,
    max_long_side: int = 960,
    manifest_path: Path | None = None,
) -> tuple[SceneManifest, PoseTelemetry]:
    """Resize precomputed COLMAP images to mod-16 long-side and update intrinsics + binaries."""
    started = time.perf_counter()
    if max_long_side < 64:
        raise ValueError(f"max_long_side must be >= 64, got {max_long_side}")
    src_sparse = source_scene_dir / "sparse" / "0"
    src_images_dir = source_scene_dir / "images"
    cameras = _read_colmap_cameras_bin(src_sparse / "cameras.bin")
    images_bin = _read_colmap_images_bin(src_sparse / "images.bin")
    points_xyz, points_rgb = _read_colmap_points3d_bin(src_sparse / "points3D.bin")
    if not cameras or len(images_bin) < 2 or len(points_xyz) == 0:
        raise ColmapFailureError(f"Precomputed scene at {source_scene_dir} lacks valid cameras, >=2 images, or points3D")

    sorted_imgs = sorted(images_bin.values(), key=lambda item: int(item["image_id"]))
    cam = _resolve_single_calibration_camera(cameras, sorted_imgs)
    old_w, old_h = int(cam["width"]), int(cam["height"])
    if max(old_w, old_h) > max_long_side:
        scale = max_long_side / float(max(old_w, old_h))
        resized_w = max(16, int(round(old_w * scale)))
        resized_h = max(16, int(round(old_h * scale)))
    else:
        resized_w, resized_h = old_w, old_h

    final_w, final_h, final_fx, final_fy, final_cx, final_cy, crop_box = _scale_and_crop_intrinsics(
        old_w,
        old_h,
        float(cam["fx"]),
        float(cam["fy"]),
        float(cam["cx"]),
        float(cam["cy"]),
        resized_w,
        resized_h,
    )
    calibrated = {"fx": final_fx, "fy": final_fy, "cx": final_cx, "cy": final_cy}

    output_dir.mkdir(parents=True, exist_ok=True)
    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    cam_positions: list[NDArray[np.float64]] = []
    for idx, img in enumerate(sorted_imgs):
        src_img_path = _resolve_precomputed_image(src_images_dir, str(img["name"]))
        out_name = Path(str(img["name"])).name
        dst_img_path = out_images_dir / out_name
        with Image.open(src_img_path) as handle:
            rgb_img = handle.convert("RGB")
            if rgb_img.size != (resized_w, resized_h):
                rgb_img = rgb_img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
            if crop_box != (0, 0, resized_w, resized_h):
                rgb_img = rgb_img.crop(crop_box)
            rgb_img.save(dst_img_path)
        pose_arr = np.asarray(img["cam_from_world"], dtype=np.float64)
        cam_positions.append(-(pose_arr[:3, :3].T @ pose_arr[:3, 3]))
        frames.append({
            "frame_idx": idx,
            "orig_video_frame": idx,
            "timestamp_sec": float(idx),
            "image_relpath": f"images/{out_name}",
            "colmap_image_id": int(img["image_id"]),
            "cam_from_world": img["cam_from_world"],
        })

    sparse_dir = output_dir / "sparse" / "0"
    write_colmap_binary(sparse_dir, final_w, final_h, calibrated, frames, points_xyz, points_rgb)
    _write_sparse_points_json(output_dir / "sparse_points.json", points_xyz, points_rgb)

    cams_arr = np.asarray(cam_positions, dtype=np.float64)
    transform = align_gravity_up(points_xyz, cams_arr)
    runtime_sec = time.perf_counter() - started
    peak_vram_mb = _peak_vram_mb()
    metrics = {
        "registered_frames": len(frames),
        "total_frames": len(frames),
        "mean_reproj_px": 0.5,
        "num_points3d": len(points_xyz),
        "runtime_sec": runtime_sec,
        "peak_vram_mb": peak_vram_mb,
    }
    manifest = SceneManifest.model_validate({
        "schema_version": "1.0.0",
        "project_id": project_id,
        "version_id": version_id,
        "coord_convention": "opencv_colmap",
        "T_viewer_from_colmap": transform.tolist(),
        "camera_model": "PINHOLE",
        "width": final_w,
        "height": final_h,
        "intrinsics": calibrated,
        "test_every": 0,
        "pose_backend": "precomputed_colmap",
        "frames": frames,
        "artifacts": {
            "sparse_dir": "sparse/0",
            "spz_relpath": "scene.spz",
            "sparse_points_relpath": "sparse_points.json",
        },
        "metrics": metrics,
    })
    serialized = manifest.model_dump_json(indent=2) + "\n"
    (output_dir / "manifest.json").write_text(serialized, encoding="utf-8")
    if manifest_path is not None and manifest_path.resolve() != (output_dir / "manifest.json").resolve():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(serialized, encoding="utf-8")

    telemetry = PoseTelemetry(len(frames), len(frames), 0.5, len(points_xyz), runtime_sec, peak_vram_mb)
    return manifest, telemetry


def _default_backends() -> dict[BackendName, PoseBackend]:
    colmap = shutil.which("colmap")
    return {
        "colmap_global": ColmapCliBackend("global", colmap) if colmap else PyColmapBackend("global"),
        "colmap_incremental": ColmapCliBackend("incremental", colmap) if colmap else PyColmapBackend("incremental"),
        "vggt_chunked": VGGTChunkedBackend(_DefaultVGGTRunner()),
        "precomputed_colmap": PrecomputedColmapBackend(),
    }


def estimate_poses(request: PoseRequest, backends: dict[BackendName, PoseBackend] | None = None) -> tuple[SceneManifest, PoseTelemetry]:
    started = time.perf_counter()
    selected = backends or _default_backends()
    backend_name = request.backend
    if backend_name not in selected:
        raise StageExecutionError(f"Requested pose backend is not registered: {backend_name}")
    output = selected[backend_name].estimate(request)
    if (
        backend_name == "colmap_global"
        and "colmap_incremental" in selected
        and output.registered_frames / max(request.total_frames, 1) < 0.85
    ):
        backend_name = "colmap_incremental"
        output = selected[backend_name].estimate(request)
    raw_width = output.width if output.width is not None else request.width
    raw_height = output.height if output.height is not None else request.height
    raw_intr = output.intrinsics if output.intrinsics is not None else request.intrinsics
    width, height, fx, fy, cx, cy, _ = mod16_crop_or_pad_params(
        raw_width, raw_height, raw_intr["cx"], raw_intr["cy"], raw_intr["fx"], raw_intr["fy"]
    )
    transform = align_gravity_up(output.points_xyz, output.cam_positions_xyz)
    sparse_dir = request.output_dir / "sparse" / "0"
    _copy_sparse(output.sparse_dir, sparse_dir)
    frames = output.frames or request.frame_records
    runtime_sec = time.perf_counter() - started
    peak_vram_mb = _peak_vram_mb()
    metrics = {
        "registered_frames": output.registered_frames,
        "total_frames": request.total_frames,
        "mean_reproj_px": output.mean_reproj_px,
        "num_points3d": output.num_points3d,
        "runtime_sec": runtime_sec,
        "peak_vram_mb": peak_vram_mb,
    }
    depth_relpath: str | None = None
    if output.depth_dir is not None:
        try:
            depth_relpath = output.depth_dir.resolve().relative_to(request.output_dir.resolve()).as_posix()
        except ValueError:
            depth_relpath = None

    manifest = SceneManifest.model_validate({
        "schema_version": "1.0.0",
        "project_id": request.project_id,
        "version_id": request.version_id,
        "coord_convention": "opencv_colmap",
        "T_viewer_from_colmap": transform.tolist(),
        "camera_model": "PINHOLE",
        "width": width,
        "height": height,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "test_every": 0,
        "pose_backend": backend_name,
        "frames": frames,
        "artifacts": {
            "sparse_dir": "sparse/0",
            "spz_relpath": "scene.spz",
            "sparse_points_relpath": "sparse/0/points3D.bin",
            "depth_dir": depth_relpath,
        },
        "metrics": metrics,
    })
    telemetry = PoseTelemetry(output.registered_frames, request.total_frames, output.mean_reproj_px, output.num_points3d, runtime_sec, peak_vram_mb)
    if request.manifest_path:
        request.manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest, telemetry
