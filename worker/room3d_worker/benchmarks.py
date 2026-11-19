from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from room3d_worker.common.manifest import SceneManifest
from room3d_worker.pipeline.s02_poses import BackendOutput, PoseRequest, VGGTChunkedBackend, estimate_poses
from room3d_worker.pipeline.s04_export import ExportRequest, export_scene, read_spz_header
from room3d_worker.pipeline.s05_track import (
    PromptPoint,
    WarmSamSession,
    lift_and_reproject_masks,
    mask_iou,
    run_tracking_pipeline,
)
from room3d_worker.pipeline.s06_diffusion import (
    DiffusionConfig,
    generate_first_frame_seeds,
    propagate_masked_edits,
    resolve_t4_dtype,
)


def _probe_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return info
    info["torch_version"] = getattr(torch, "__version__", "unknown")
    if not torch.cuda.is_available():
        return info
    info["cuda_available"] = True
    info["device_count"] = int(torch.cuda.device_count())
    for idx in range(info["device_count"]):
        cap = torch.cuda.get_device_capability(idx)
        _, total_b = torch.cuda.mem_get_info(idx)
        info["devices"].append({
            "index": idx,
            "name": torch.cuda.get_device_name(idx),
            "compute_capability": f"{cap[0]}.{cap[1]}",
            "vram_total_mb": round(total_b / (1024 * 1024), 1),
        })
    return info


def _write_synthetic_frames(image_dir: Path, num_frames: int, hw: tuple[int, int]) -> dict[int, Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    h, w = hw
    paths: dict[int, Path] = {}
    for idx in range(num_frames):
        arr = np.full((h, w, 3), 45, dtype=np.uint8)
        cx = int(w * 0.35 + (idx / max(num_frames - 1, 1)) * w * 0.3)
        cy = h // 2
        arr[max(0, cy - 16) : min(h, cy + 16), max(0, cx - 16) : min(w, cx + 16)] = (210, 75, 45)
        path = image_dir / f"{idx:06d}.png"
        Image.fromarray(arr).save(path, format="PNG")
        paths[idx] = path
    return paths


def _benchmark_poses(root: Path, num_frames: int, hw: tuple[int, int]) -> dict[str, Any]:
    h, w = hw
    img_dir = root / "pose_images"
    paths = _write_synthetic_frames(img_dir, num_frames, hw)
    frames = [
        {
            "frame_idx": idx,
            "orig_video_frame": idx,
            "timestamp_sec": float(idx) * 0.5,
            "image_relpath": f"images/{paths[idx].name}",
            "colmap_image_id": idx + 1,
            "cam_from_world": np.array(
                [[1.0, 0.0, 0.0, -0.1 * idx], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.5], [0.0, 0.0, 0.0, 1.0]],
                dtype=float,
            ).tolist(),
        }
        for idx in range(num_frames)
    ]
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    cams = np.array(
        [[np.cos(i * 0.4), 1.5 + 0.25 * (i % 2), np.sin(i * 0.4) - 2.5] for i in range(num_frames)],
        dtype=np.float64,
    )
    req = PoseRequest(
        image_dir=img_dir,
        output_dir=root / "pose_scene",
        total_frames=num_frames,
        width=w,
        height=h,
        intrinsics={"fx": float(w), "fy": float(w), "cx": w / 2.0, "cy": h / 2.0},
        backend="colmap_global",
        frame_records=frames,
    )

    class _FastColmap:
        def estimate(self, request: PoseRequest) -> BackendOutput:
            return BackendOutput(num_frames, 0.35, len(pts), pts.copy(), cams.copy(), list(request.frame_records))

    t0 = time.perf_counter()
    manifest, telemetry = estimate_poses(req, {"colmap_global": _FastColmap()})
    colmap_ms = (time.perf_counter() - t0) * 1000.0

    class _ChunkRunner:
        def __init__(self) -> None:
            self.chunk_calls = 0

        def infer(self, images: list[Path], *, dtype: str) -> BackendOutput:
            _ = dtype
            scale = 1.0 if self.chunk_calls == 0 else 1.8
            shift = np.zeros(3) if self.chunk_calls == 0 else np.array([2.0, -1.0, 0.5])
            self.chunk_calls += 1
            sub_frames = [
                frames[int(p.stem)]
                for p in images
            ]
            sub_cams = np.array([cams[int(p.stem)] for p in images], dtype=np.float64) * scale + shift
            return BackendOutput(len(images), 0.4, len(pts), pts.copy() * scale + shift, sub_cams, sub_frames)

    overlap = min(3, max(1, num_frames - 1))
    chunk_size = max(overlap + 1, min(num_frames, (num_frames // 2) + overlap))
    t1 = time.perf_counter()
    vggt_out = VGGTChunkedBackend(_ChunkRunner(), chunk_size=chunk_size, overlap=overlap).estimate(
        PoseRequest(
            image_dir=img_dir,
            output_dir=root / "vggt_scene",
            total_frames=num_frames,
            width=w,
            height=h,
            intrinsics={"fx": float(w), "fy": float(w), "cx": w / 2.0, "cy": h / 2.0},
            backend="vggt_chunked",
            frame_records=frames,
        )
    )
    vggt_ms = (time.perf_counter() - t1) * 1000.0
    return {
        "frames": num_frames,
        "resolution": f"{w}x{h}",
        "colmap_synth_ms": round(colmap_ms, 3),
        "vggt_chunked_synth_ms": round(vggt_ms, 3),
        "registered_frames": telemetry.registered_frames,
        "vggt_registered_frames": vggt_out.registered_frames,
        "manifest_mod16_ok": manifest.width % 16 == 0 and manifest.height % 16 == 0,
    }


def _benchmark_tracking(root: Path, num_frames: int, hw: tuple[int, int]) -> dict[str, Any]:
    h, w = hw
    paths = _write_synthetic_frames(root / "track_images", num_frames, hw)
    depth_dir = root / "track_depths"
    depth_dir.mkdir(parents=True, exist_ok=True)
    gt_masks: dict[int, NDArray[np.uint8]] = {}
    frames = []
    for idx in range(num_frames):
        cx = int(w * 0.35 + (idx / max(num_frames - 1, 1)) * w * 0.3)
        cy = h // 2
        mask = np.zeros(hw, dtype=np.uint8)
        mask[max(0, cy - 16) : min(h, cy + 16), max(0, cx - 16) : min(w, cx + 16)] = 255
        gt_masks[idx] = mask
        np.save(depth_dir / f"{idx:06d}.npy", np.full(hw, 2.5, dtype=np.float32))
        frames.append({
            "frame_idx": idx,
            "orig_video_frame": idx,
            "timestamp_sec": float(idx) * 0.5,
            "image_relpath": f"images/{idx:06d}.png",
            "colmap_image_id": idx + 1,
            "cam_from_world": np.eye(4).tolist(),
        })
    manifest = SceneManifest.model_validate({
        "schema_version": "1.0.0",
        "project_id": "bench",
        "version_id": "v1",
        "coord_convention": "opencv_colmap",
        "T_viewer_from_colmap": np.eye(4).tolist(),
        "camera_model": "PINHOLE",
        "width": w,
        "height": h,
        "intrinsics": {"fx": float(w), "fy": float(w), "cx": w / 2.0, "cy": h / 2.0},
        "test_every": 0,
        "frames": frames,
        "artifacts": {"sparse_dir": "sparse/0", "spz_relpath": "scene.spz", "sparse_points_relpath": "sparse_points.json"},
    })

    class _BenchInteractiveSam:
        def encode_image(self, image_path: Path) -> NDArray[np.float32]:
            _ = image_path
            return np.zeros((4, 4), dtype=np.float32)

        def predict(self, embedding: Any, points: list[PromptPoint]) -> NDArray[np.uint8]:
            _ = embedding
            mask = np.zeros(hw, dtype=np.uint8)
            px, py = int(points[0].x), int(points[0].y)
            mask[max(0, py - 16) : min(h, py + 16), max(0, px - 16) : min(w, px + 16)] = 255
            return mask

    class _BenchVideoSam:
        def propagate(
            self,
            image_paths: dict[int, Path],
            ref_frame_idx: int,
            seed_mask: NDArray[np.uint8],
            inference_state_device: str,
        ) -> dict[int, NDArray[np.uint8]]:
            _ = (ref_frame_idx, seed_mask, inference_state_device)
            return {idx: gt_masks[idx].copy() for idx in image_paths}

    session = WarmSamSession(_BenchInteractiveSam(), max_cache_items=4)
    ref_cx = int(w * 0.35)
    _, miss_ms = session.segment_clicks(paths[0], [PromptPoint(ref_cx, h // 2, 1)], hw)
    _, hit_ms = session.segment_clicks(paths[0], [PromptPoint(ref_cx + 1, h // 2, 1)], hw)

    t0 = time.perf_counter()
    saved, _ = run_tracking_pipeline(
        manifest=manifest,
        image_paths=paths,
        ref_frame_idx=0,
        points=[PromptPoint(ref_cx, h // 2, 1)],
        sam_session=session,
        video_backend=_BenchVideoSam(),
        output_dir=root / "track_out",
        depth_dir=depth_dir,
    )
    track_ms = (time.perf_counter() - t0) * 1000.0
    ious = [
        mask_iou(np.asarray(Image.open(saved[idx])), gt_masks[idx])
        for idx in sorted(saved)
    ]
    no_depth_masks, no_depth_vis, _ = lift_and_reproject_masks(gt_masks, manifest, paths, None, 0, session)
    vos_preserved = all(np.array_equal(no_depth_masks[i], gt_masks[i]) and no_depth_vis[i] == "no_depth" for i in gt_masks)
    return {
        "frames": num_frames,
        "warm_click_miss_ms": round(miss_ms, 3),
        "warm_click_hit_ms": round(hit_ms, 3),
        "pipeline_ms": round(track_ms, 3),
        "ms_per_frame": round(track_ms / max(num_frames, 1), 3),
        "mean_iou": round(float(np.mean(ious)), 4),
        "vos_preserved_without_depth": vos_preserved,
    }


def _benchmark_diffusion(root: Path, num_frames: int, hw: tuple[int, int]) -> dict[str, Any]:
    h, w = hw
    frame_paths = _write_synthetic_frames(root / "diff_images", num_frames, hw)
    mask_dir = root / "diff_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_paths: dict[int, Path] = {}
    for idx in range(num_frames):
        mask = np.zeros(hw, dtype=np.uint8)
        if idx != num_frames - 1:
            mask[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4] = 255
        mp = mask_dir / f"{idx:06d}.png"
        Image.fromarray(mask).save(mp, format="PNG")
        mask_paths[idx] = mp

    class _FastInpaintBackend:
        def __init__(self) -> None:
            self.calls = 0

        def inpaint(
            self,
            image: NDArray[np.uint8],
            mask: NDArray[np.uint8],
            prompt: str,
            seed: int,
            image_reference: NDArray[np.uint8] | None = None,
        ) -> NDArray[np.uint8]:
            _ = (mask, prompt, seed, image_reference)
            self.calls += 1
            return np.full_like(image, (40, 180, 90), dtype=np.uint8)

    backend = _FastInpaintBackend()
    t0 = time.perf_counter()
    seeds = generate_first_frame_seeds(frame_paths[0], mask_paths[0], "green velvet chair", [11, 22], root / "seeds", backend)
    seed_ms = (time.perf_counter() - t0) * 1000.0
    ref_seed_path = Path(seeds[0]["path"])

    t1 = time.perf_counter()
    grid_metrics = propagate_masked_edits(
        frame_paths, mask_paths, 0, ref_seed_path, "green velvet chair", 11, root / "diff_grid", backend,
        config=DiffusionConfig(grid_2x2=True),
    )
    grid_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    seq_metrics = propagate_masked_edits(
        frame_paths, mask_paths, 0, ref_seed_path, "green velvet chair", 11, root / "diff_seq", backend,
        config=DiffusionConfig(grid_2x2=False),
    )
    seq_ms = (time.perf_counter() - t2) * 1000.0

    return {
        "frames": num_frames,
        "seed_preview_2_ms": round(seed_ms, 3),
        "grid_2x2_ms": round(grid_ms, 3),
        "sequential_ms": round(seq_ms, 3),
        "ms_per_frame_grid": round(grid_ms / max(num_frames, 1), 3),
        "ms_per_frame_seq": round(seq_ms / max(num_frames, 1), 3),
        "max_unmasked_diff": max(int(grid_metrics["max_unmasked_diff"]), int(seq_metrics["max_unmasked_diff"])),
        "ref_method": grid_metrics["frames"][0]["method"],
        "empty_mask_skipped": bool(grid_metrics["frames"][-1]["skipped"]),
        "t4_sm75_dtype": resolve_t4_dtype((7, 5)),
    }


def _benchmark_export(root: Path, num_gaussians: int, hw: tuple[int, int]) -> dict[str, Any]:
    h, w = hw
    rng = np.random.default_rng(42)
    points = rng.normal(0.0, 1.5, size=(num_gaussians, 3)).astype(np.float32)
    opacity = rng.uniform(0.2, 0.98, size=(num_gaussians,)).astype(np.float32)
    scales = rng.uniform(0.005, 0.05, size=(num_gaussians, 3)).astype(np.float32)
    colors = rng.integers(0, 256, size=(num_gaussians, 3), dtype=np.uint8)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_gaussians, 1))
    out_dir = root / "export_out"

    t0 = time.perf_counter()
    manifest = export_scene(
        ExportRequest(output_dir=out_dir, project_id="bench", version_id="v1", width=w, height=h),
        points_xyz=points,
        opacity=opacity,
        scales=scales,
        colors=colors,
        quats=quats,
    )
    export_ms = (time.perf_counter() - t0) * 1000.0
    ply_bytes = (out_dir / "scene.ply").stat().st_size
    spz_bytes = (out_dir / "scene.spz").stat().st_size
    header = read_spz_header(out_dir / "scene.spz")
    return {
        "gaussians_in": num_gaussians,
        "gaussians_exported": int((manifest.metrics or {}).get("gaussian_count", 0)),
        "export_ms": round(export_ms, 3),
        "ply_bytes": ply_bytes,
        "spz_bytes": spz_bytes,
        "compression_ratio": round(ply_bytes / max(spz_bytes, 1), 2),
        "spz_magic_hex": hex(header["magic"]),
        "spz_version": header["version"],
    }


def run_benchmarks(
    work_dir: Path | None = None,
    *,
    num_frames: int = 8,
    num_gaussians: int = 10_000,
    image_hw: tuple[int, int] = (128, 160),
) -> dict[str, Any]:
    """Run reproducible local CPU/GPU benchmarks across pose, tracking, diffusion compositing, and export."""
    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="room3d_bench_") as tmp:
            return run_benchmarks(Path(tmp), num_frames=num_frames, num_gaussians=num_gaussians, image_hw=image_hw)
    work_dir.mkdir(parents=True, exist_ok=True)
    return {
        "hardware": _probe_hardware(),
        "poses": _benchmark_poses(work_dir, num_frames, image_hw),
        "tracking": _benchmark_tracking(work_dir, num_frames, image_hw),
        "diffusion": _benchmark_diffusion(work_dir, num_frames, image_hw),
        "export": _benchmark_export(work_dir, num_gaussians, image_hw),
    }


def format_markdown_report(results: dict[str, Any]) -> str:
    hw = results["hardware"]
    gpu_desc = ", ".join(f"{d['name']} (sm_{d['compute_capability']}, {d['vram_total_mb']} MB)" for d in hw.get("devices", [])) or "CPU only"
    p = results["poses"]
    t = results["tracking"]
    d = results["diffusion"]
    e = results["export"]
    return "\n".join([
        "# Room3D Local Benchmark Report",
        "",
        f"- **OS / Python:** `{hw['os']}` / Python `{hw['python']}`",
        f"- **Compute Device(s):** `{gpu_desc}`",
        "",
        "| Stage | Key Metric | Measured Value |",
        "| :--- | :--- | :--- |",
        f"| `s02_poses` | Synthetic COLMAP + VGGT-3chunk (`{p['frames']}` frames) | `{p['colmap_synth_ms']} ms` / `{p['vggt_chunked_synth_ms']} ms` |",
        f"| `s05_track` | Warm click hit / full 5-tier (`{t['frames']}` frames, IoU `{t['mean_iou']}`) | `{t['warm_click_hit_ms']} ms` / `{t['pipeline_ms']} ms` (`{t['ms_per_frame']} ms/frame`) |",
        f"| `s06_diffusion` | 2x2 grid vs sequential compositing (`max_unmasked_diff={d['max_unmasked_diff']}`) | `{d['grid_2x2_ms']} ms` / `{d['sequential_ms']} ms` (`ref={d['ref_method']}`) |",
        f"| `s04_export` | PLY -> SPZ v{e['spz_version']} (`{e['gaussians_exported']}` Gaussians) | `{e['export_ms']} ms` (`{e['compression_ratio']}x` compression, `{e['spz_bytes']}` B) |",
        "",
    ])


def write_benchmark_outputs(
    results: dict[str, Any],
    json_path: Path | None = None,
    markdown_path: Path | None = None,
) -> tuple[Path | None, Path | None]:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(format_markdown_report(results), encoding="utf-8")
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Room3D local timing and quality benchmarks.")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--num-gaussians", type=int, default=10_000)
    args = parser.parse_args(argv)
    results = run_benchmarks(args.work_dir, num_frames=args.num_frames, num_gaussians=args.num_gaussians)
    write_benchmark_outputs(results, args.json_out, args.md_out)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
