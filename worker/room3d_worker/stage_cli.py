from __future__ import annotations

import argparse
import inspect
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from room3d_worker.common.errors import StageExecutionError
from room3d_worker.common.manifest import load_manifest
from room3d_worker.pipeline.s01_extract import extract_keyframes
from room3d_worker.pipeline.s02_poses import PoseRequest, estimate_poses
from room3d_worker.pipeline.s03_train import build_train_config, train_scene
from room3d_worker.pipeline.s04_export import ExportRequest, export_splat_artifacts
from room3d_worker.pipeline.s05_track import (
    ManualOverride,
    PromptPoint,
    WarmSamSession,
    load_sam2_backends,
    run_tracking_pipeline,
)
from room3d_worker.pipeline.s06_diffusion import (
    DiffusionConfig,
    generate_first_frame_seeds,
    load_diffusion_backend,
    propagate_masked_edits,
)
from room3d_worker.pipeline.s07_rebuild import RebuildRequest, rebuild_scene


def load_sparse_points(sparse_dir: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load sparse XYZ and optional uint8 RGB from COLMAP points3D.bin or numpy files."""
    xyz_npy = sparse_dir / "points3d_xyz.npy"
    rgb_npy = sparse_dir / "points3d_rgb.npy"
    if xyz_npy.exists():
        xyz = np.asarray(np.load(xyz_npy, allow_pickle=False), dtype=np.float32)
        rgb = np.asarray(np.load(rgb_npy, allow_pickle=False), dtype=np.uint8) if rgb_npy.exists() else None
        return xyz, rgb
    bin_path = sparse_dir / "points3D.bin"
    if not bin_path.exists() or bin_path.stat().st_size < 8:
        raise StageExecutionError(f"Missing sparse point cloud under {sparse_dir}")
    with bin_path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        if count == 0 or count > 20_000_000:
            raise StageExecutionError(f"Invalid point count ({count}) in {bin_path}")
        xyz_rows: list[tuple[float, float, float]] = []
        rgb_rows: list[tuple[int, int, int]] = []
        for _ in range(count):
            _, x, y, z, r, g, b, _ = struct.unpack("<Q3d3Bd", handle.read(43))
            track_len = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(8 * track_len, 1)
            xyz_rows.append((x, y, z))
            rgb_rows.append((r, g, b))
    return np.asarray(xyz_rows, dtype=np.float32), np.asarray(rgb_rows, dtype=np.uint8)


def _load_rgb_stack(scene_dir: Path, frames: list[dict[str, Any]], *, allow_synthetic: bool = False) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for frame in frames:
        img_path = scene_dir / str(frame["image_relpath"])
        if img_path.exists():
            with Image.open(img_path) as img:
                arrays.append(np.asarray(img.convert("RGB"), dtype=np.uint8))
    if not arrays:
        if allow_synthetic:
            return np.zeros((1, 64, 64, 3), dtype=np.uint8)
        raise StageExecutionError(f"No extracted RGB frame images found under {scene_dir}")
    return np.stack(arrays, axis=0)


def _extract_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(extract_keyframes)
    optional = ("max_long_side", "min_parallax_frac", "sharpness_window", "target_fps", "max_frames", "trim_start_sec", "trim_end_sec")
    kwargs: dict[str, Any] = {}
    for key in optional:
        if key in sig.parameters and spec.get(key) is not None:
            kwargs[key] = spec[key]
    return kwargs


def _diffusion_cfg(spec: dict[str, Any], *, grid_2x2: bool = True) -> DiffusionConfig:
    sig = inspect.signature(DiffusionConfig)
    kwargs: dict[str, Any] = {
        "model_variant": spec.get("model_variant") or "flux2_klein_4b",
        "device": str(spec.get("device", "cuda:0")),
        "grid_2x2": grid_2x2,
    }
    if "negative_prompt" in sig.parameters and spec.get("negative_prompt"):
        kwargs["negative_prompt"] = str(spec["negative_prompt"])
    if "quantization" in sig.parameters:
        kwargs["quantization"] = str(spec.get("quantization") or "nf4")
    if "offload_text_encoder" in sig.parameters:
        kwargs["offload_text_encoder"] = bool(spec.get("offload_text_encoder", True))
    return DiffusionConfig(**kwargs)


def run_extract_stage(spec: dict[str, Any]) -> dict[str, Any]:
    res = extract_keyframes(spec["video_path"], spec["output_dir"], **_extract_kwargs(spec))
    return {"index_path": str(res.index_path), "frame_count": len(res.index.get("frames", []))}  # type: ignore[arg-type]


def run_poses_stage(spec: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(spec["output_dir"])
    manifest_path = Path(spec.get("manifest_path") or (out_dir / "manifest.json"))
    if spec.get("pose_backend") == "precomputed_colmap" and spec.get("source_scene_dir"):
        from room3d_worker.pipeline import s02_poses

        prepare_fn = getattr(s02_poses, "prepare_precomputed_colmap_scene", None)
        if prepare_fn is None:
            raise StageExecutionError("prepare_precomputed_colmap_scene is not available in s02_poses")
        manifest, telemetry = prepare_fn(
            Path(spec["source_scene_dir"]),
            out_dir,
            project_id=str(spec["project_id"]),
            version_id=str(spec["version_id"]),
            max_long_side=int(spec.get("max_long_side", 960)),
            manifest_path=manifest_path,
        )
        return {
            "manifest_path": str(manifest_path),
            "registered_frames": telemetry.registered_frames,
            "version_id": manifest.version_id,
        }
    req = PoseRequest(
        image_dir=Path(spec["image_dir"]),
        output_dir=out_dir,
        total_frames=int(spec["total_frames"]),
        width=int(spec["width"]),
        height=int(spec["height"]),
        intrinsics=dict(spec["intrinsics"]),
        project_id=str(spec["project_id"]),
        version_id=str(spec["version_id"]),
        backend=spec.get("pose_backend", "colmap_global"),
        frame_records=list(spec.get("frame_records", [])),
        manifest_path=manifest_path,
    )
    manifest, telemetry = estimate_poses(req)
    return {"manifest_path": str(manifest_path), "registered_frames": telemetry.registered_frames, "version_id": manifest.version_id}


def _build_train_cfg(out_dir: Path, device: str, iterations: int, *, allow_synthetic: bool) -> Any:
    sig = inspect.signature(build_train_config)
    kwargs: dict[str, Any] = {"device": device, "iterations": iterations}
    if "allow_synthetic_fallback" in sig.parameters:
        kwargs["allow_synthetic_fallback"] = allow_synthetic
    return build_train_config(out_dir, **kwargs)


def run_train_stage(spec: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(spec["output_dir"])
    manifest = load_manifest(spec["manifest_path"])
    allow_synthetic = bool(spec.get("allow_synthetic", False))
    xyz, rgb = load_sparse_points(out_dir / "sparse" / "0")
    images = _load_rgb_stack(out_dir, [f.model_dump() for f in manifest.frames], allow_synthetic=allow_synthetic)
    cfg = _build_train_cfg(
        out_dir,
        device=str(spec.get("device", "cuda:0")),
        iterations=int(spec.get("iterations", 30000)),
        allow_synthetic=allow_synthetic,
    )
    telemetry = train_scene(cfg, images, xyz, points_rgb=rgb, manifest=manifest)
    return {"psnr": telemetry.psnr, "ssim": telemetry.ssim, "gaussian_count": telemetry.gaussian_count}


def run_export_stage(spec: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(spec["output_dir"])
    manifest = load_manifest(spec["manifest_path"])
    ckpt_path = Path(spec.get("ckpt_path") or (out_dir / "gs_state.npz"))
    exported = export_splat_artifacts(
        ExportRequest(
            output_dir=out_dir,
            project_id=manifest.project_id,
            version_id=manifest.version_id,
            parent_version_id=manifest.parent_version_id,
            width=manifest.width,
            height=manifest.height,
            intrinsics=manifest.intrinsics.model_dump(),
            frames=[f.model_dump() for f in manifest.frames],
            test_every=manifest.test_every,
            pose_backend=manifest.pose_backend,
            T_viewer_from_colmap=manifest.T_viewer_from_colmap,
        ),
        ckpt_path=ckpt_path,
    )
    return {"manifest_path": str(out_dir / "manifest.json"), "metrics": exported.metrics}


def run_track_stage(spec: dict[str, Any]) -> dict[str, Any]:
    manifest = load_manifest(spec["manifest_path"])
    image_paths = {int(k): Path(v) for k, v in dict(spec["image_paths"]).items()}
    points = [
        PromptPoint(
            x=int(round(float(p["x"]))) if p.get("x") is not None else int(round(float(p["u_norm"]) * manifest.width)),
            y=int(round(float(p["y"]))) if p.get("y") is not None else int(round(float(p["v_norm"]) * manifest.height)),
            label=int(p.get("label", 1)),
        )
        for p in spec.get("points", [])
    ]
    overrides = [
        ManualOverride(
            frame_idx=int(o["frame_idx"]),
            mask_path=Path(o["mask_path"]) if o.get("mask_path") else None,
            skip_frame=bool(o.get("skip_frame", False)),
        )
        for o in spec.get("overrides", [])
    ]
    model_cfg = str(spec.get("sam2_config") or os.environ.get("ROOM3D_SAM2_CONFIG", ""))
    checkpoint = str(spec.get("sam2_checkpoint") or os.environ.get("ROOM3D_SAM2_CHECKPOINT", ""))
    interactive, video_backend = load_sam2_backends(model_cfg, checkpoint, device=str(spec.get("device", "cuda:0")))
    saved_masks, stats = run_tracking_pipeline(
        manifest,
        image_paths,
        ref_frame_idx=int(spec["ref_frame_idx"]),
        points=points,
        sam_session=WarmSamSession(interactive),
        video_backend=video_backend,
        output_dir=Path(spec["output_dir"]),
        depth_dir=Path(spec["depth_dir"]) if spec.get("depth_dir") else None,
        overrides=overrides,
    )
    return {"saved_masks": {str(k): str(v) for k, v in saved_masks.items()}, "stats": stats}


def run_diffusion_preview_stage(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = _diffusion_cfg(spec, grid_2x2=bool(spec.get("grid_2x2", True)))
    backend = load_diffusion_backend(cfg)
    seeds = generate_first_frame_seeds(
        Path(spec["ref_image_path"]),
        Path(spec["ref_mask_path"]),
        str(spec["prompt"]),
        [int(s) for s in spec["seeds"]],
        Path(spec["output_dir"]),
        backend,
        config=cfg,
    )
    return {"seeds": seeds}


def run_diffusion_apply_stage(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = _diffusion_cfg(spec, grid_2x2=False)
    backend = load_diffusion_backend(cfg)
    frame_paths = {int(k): Path(v) for k, v in dict(spec["frame_paths"]).items()}
    mask_paths = {int(k): Path(v) for k, v in dict(spec["mask_paths"]).items()}
    return propagate_masked_edits(
        frame_paths,
        mask_paths,
        int(spec["ref_frame_idx"]),
        Path(spec["ref_edited_path"]),
        str(spec["prompt"]),
        int(spec["seed"]),
        Path(spec["output_dir"]),
        backend,
        skip_frames={int(i) for i in spec.get("skip_frames", [])},
        config=cfg,
    )


def run_rebuild_stage(spec: dict[str, Any]) -> dict[str, Any]:
    req = RebuildRequest(
        base_dir=Path(spec["base_dir"]),
        edited_frames_dir=Path(spec["edited_frames_dir"]),
        masks_dir=Path(spec["masks_dir"]),
        output_dir=Path(spec["output_dir"]),
        project_id=str(spec["project_id"]),
        version_id=str(spec["version_id"]),
        parent_version_id=str(spec["parent_version_id"]),
        mode=spec.get("mode", "masked_finetune"),
        exclude_frames=tuple(int(i) for i in spec.get("exclude_frames", [])),
        device=str(spec.get("device", "cuda:0")),
        rebuild_safeguards=bool(spec.get("rebuild_safeguards", False)),
        unmasked_weight=float(spec.get("unmasked_weight", 1.0)),
    )
    return rebuild_scene(req)


_STAGE_DISPATCH = {
    "extract": run_extract_stage,
    "poses": run_poses_stage,
    "train": run_train_stage,
    "export": run_export_stage,
    "track": run_track_stage,
    "diffusion_preview": run_diffusion_preview_stage,
    "diffusion_apply": run_diffusion_apply_stage,
    "rebuild": run_rebuild_stage,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute an isolated Room3D pipeline stage")
    parser.add_argument("stage", choices=sorted(_STAGE_DISPATCH))
    parser.add_argument("--spec", required=True, help="Path to JSON stage specification file")
    parser.add_argument("--out", required=False, help="Optional path to write JSON stage output")
    args = parser.parse_args(argv)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    result = _STAGE_DISPATCH[args.stage](spec)
    rendered = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
