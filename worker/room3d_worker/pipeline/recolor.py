from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from room3d_worker.common.manifest import FrameRecord


def recolor_rgb(rgb: NDArray[np.uint8], color_hex: str, strength: float = 1.0) -> NDArray[np.uint8]:
    """Apply shared paint chromaticity with bounded luminance detail in linear light."""
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color_hex) or not 0.0 < strength <= 1.0:
        raise ValueError("Recolor requires #RRGGBB and strength in (0, 1]")
    source = rgb.astype(np.float32) / 255.0
    color = np.array([int(color_hex[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255.0
    linear = np.where(source <= 0.04045, source / 12.92, ((source + 0.055) / 1.055) ** 2.4)
    paint = np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    # Fixed exposure across all views avoids per-frame normalization and its color flicker.
    shading = np.clip(luminance / 0.5, 0.0, 1.5)[..., None]
    result = np.clip((1 - strength) * linear + strength * paint * shading, 0, 1)
    srgb = np.where(result <= 0.0031308, result * 12.92, 1.055 * result ** (1 / 2.4) - 0.055)
    return np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8)


def diverse_frames(frames: Sequence[FrameRecord], reference: int, count: int = 4) -> list[FrameRecord]:
    candidates = [frame for frame in frames if frame.cam_from_world is not None and not frame.excluded]
    if not candidates:
        return []
    features = []
    for frame in candidates:
        pose = np.asarray(frame.cam_from_world, dtype=np.float64)
        features.append(np.concatenate((-pose[:3, :3].T @ pose[:3, 3], pose[2, :3])))
    values = np.stack(features)
    values[:, :3] /= max(float(np.linalg.norm(np.ptp(values[:, :3], axis=0))), 1e-6)
    chosen = [next((i for i, frame in enumerate(candidates) if frame.frame_idx == reference), 0)]
    while len(chosen) < min(count, len(candidates)):
        distances = np.min(np.linalg.norm(values[:, None] - values[chosen], axis=2), axis=1)
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return [candidates[i] for i in chosen]
