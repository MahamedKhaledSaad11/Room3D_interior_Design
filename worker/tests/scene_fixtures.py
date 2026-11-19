from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from PIL import Image

from room3d_worker.pipeline.s04_export import ExportRequest, export_scene


def _write_points3d_bin(path: Path, points: list[tuple[float, float, float, int, int, int]]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        for idx, (x, y, z, r, g, b) in enumerate(points, start=1):
            handle.write(struct.pack("<Q3d3BdQ", idx, x, y, z, r, g, b, 0.1, 0))


def _write_images_bin(path: Path, records: list[tuple[int, str, float]]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(records)))
        for img_id, name, tx in records:
            handle.write(struct.pack("<I4d3dI", img_id, 1.0, 0.0, 0.0, 0.0, tx, 0.0, 0.0, 1))
            handle.write(name.encode("utf-8") + b"\x00")
            handle.write(struct.pack("<Q", 0))


def _setup_base_scene(tmp_path: Path, with_sparse: bool = True, with_points_bin: bool = False) -> tuple[Path, Path, Path]:
    base_dir = tmp_path / "v1"
    edited_dir = tmp_path / "edited"
    masks_dir = tmp_path / "masks"
    edited_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    pts = np.array([[0.0, 0.0, 4.0], [3.5, 0.0, 4.0], [-3.5, 0.0, 4.0], [0.2, 0.2, 4.0]], dtype=np.float32)
    op = np.full(4, 0.85, dtype=np.float32)
    scales = np.array([[0.02, 0.04, 0.06], [0.03, 0.02, 0.05], [0.02, 0.02, 0.02], [0.05, 0.03, 0.02]], dtype=np.float32)
    quats = np.array([[1.0, 0.0, 0.0, 0.0], [0.707, 0.707, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.707, 0.0, 0.707, 0.0]], dtype=np.float32)
    colors = np.full((4, 3), 120, dtype=np.uint8)
    frames = [
        {"frame_idx": 0, "orig_video_frame": 0, "timestamp_sec": 0.0, "image_relpath": "images/frame_000000.jpg",
         "colmap_image_id": 1, "cam_from_world": np.eye(4).tolist()},
        {"frame_idx": 1, "orig_video_frame": 1, "timestamp_sec": 0.5, "image_relpath": "images/frame_000001.jpg",
         "colmap_image_id": 2, "cam_from_world": [[1, 0, 0, -0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
    ]
    export_scene(
        ExportRequest(output_dir=base_dir, project_id="proj-1", version_id="v1", width=160, height=128,
                      intrinsics={"fx": 160.0, "fy": 160.0, "cx": 80.0, "cy": 64.0}, frames=frames),
        points_xyz=pts, opacity=op, scales=scales, colors=colors, quats=quats,
    )
    if with_sparse:
        sparse_dir = base_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        (sparse_dir / "cameras.bin").write_bytes(b"cam")
        _write_images_bin(sparse_dir / "images.bin", [(1, "frame_000000.jpg", 0.0), (2, "frame_000001.jpg", -0.1)])
        if with_points_bin:
            _write_points3d_bin(sparse_dir / "points3D.bin", [(0.0, 0.0, 3.0, 10, 200, 30), (0.5, 0.1, 3.5, 220, 40, 50)])
    mask = np.zeros((128, 160), dtype=np.uint8)
    mask[56:80, 72:104] = 255
    (base_dir / "images").mkdir(parents=True, exist_ok=True)
    (base_dir / "depths").mkdir(parents=True, exist_ok=True)
    for idx in (0, 1):
        Image.fromarray(mask).save(masks_dir / f"{idx:05d}.png", format="PNG")
        edited = np.full((128, 160, 3), 240, dtype=np.uint8)
        edited[0, 0] = [13, 77, 211]
        Image.fromarray(edited).save(edited_dir / f"{idx:05d}.png", format="PNG")
        Image.fromarray(np.full((128, 160, 3), 55, dtype=np.uint8)).save(base_dir / "images" / f"frame_{idx:06d}.jpg", format="JPEG")
        np.save(base_dir / "depths" / f"{idx:05d}.npy", np.full((128, 160), 2.0, dtype=np.float32))
    return base_dir, edited_dir, masks_dir
