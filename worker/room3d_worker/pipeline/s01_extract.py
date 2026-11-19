from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import av
import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


@dataclass(frozen=True)
class ExtractionResult:
    output_dir: Path
    index_path: Path
    index: dict[str, object]


def _frame_timestamp(frame: av.VideoFrame, index: int, rate: float) -> float:
    if frame.pts is not None and frame.time_base is not None:
        return float(frame.pts * frame.time_base)
    return index / rate if rate > 0 else 0.0


def _sharpness(gray: Image) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _parallax(previous: Image, current: Image) -> float:
    points = cv2.goodFeaturesToTrack(previous, maxCorners=160, qualityLevel=0.01, minDistance=8)
    if points is None or len(points) < 4:
        return 0.0
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if next_points is None or status is None:
        return 0.0
    valid = status.reshape(-1).astype(bool)
    if valid.sum() < 4:
        return 0.0
    displacement = np.linalg.norm(next_points[valid] - points[valid], axis=1)
    return float(np.median(displacement) / min(previous.shape[:2]))


def _prepare(image: Image, max_long_side: int) -> tuple[Image, int, int]:
    height, width = image.shape[:2]
    scale = min(1.0, max_long_side / max(width, height))
    if scale < 1.0:
        width, height = round(width * scale), round(height * scale)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    width -= width % 16
    height -= height % 16
    if width < 16 or height < 16:
        raise ValueError("Video frames must be at least 16 pixels in both dimensions")
    x0, y0 = (image.shape[1] - width) // 2, (image.shape[0] - height) // 2
    return image[y0:y0 + height, x0:x0 + width], width, height


def _write_jpeg(path: Path, image: Image) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise OSError(f"Could not encode JPEG {path}")
    path.write_bytes(encoded.tobytes())


def _subsample_records(destination: Path, records: list[dict[str, object]], max_frames: int) -> list[dict[str, object]]:
    if len(records) <= max_frames:
        return records
    chosen = [0] if max_frames == 1 else np.round(np.linspace(0, len(records) - 1, max_frames)).astype(int).tolist()
    chosen_set = set(chosen)
    for idx, rec in enumerate(records):
        if idx not in chosen_set:
            (destination / str(rec["image_relpath"])).unlink(missing_ok=True)
    reindexed: list[dict[str, object]] = []
    for new_idx, old_idx in enumerate(chosen):
        rec = dict(records[old_idx])
        old_rel = Path(str(rec["image_relpath"]))
        new_rel = Path("images") / f"frame_{new_idx:06d}.jpg"
        if old_rel != new_rel:
            (destination / old_rel).replace(destination / new_rel)
        rec["frame_idx"] = new_idx
        rec["image_relpath"] = new_rel.as_posix()
        reindexed.append(rec)
    return reindexed


def extract_clean_frames(
    video: str | os.PathLike[str] | BinaryIO,
    output_dir: str | os.PathLike[str],
    *,
    max_long_side: int = 960,
    min_parallax_frac: float = 0.04,
    sharpness_window: int = 7,
    target_fps: float | None = 4.0,
    max_frames: int | None = 300,
    trim_start_sec: float | None = None,
    trim_end_sec: float | None = None,
) -> ExtractionResult:
    if not 640 <= max_long_side <= 960:
        raise ValueError("max_long_side must be between 640 and 960")
    if min_parallax_frac < 0 or sharpness_window < 1:
        raise ValueError("min_parallax_frac and sharpness_window must be non-negative/positive")
    if target_fps is not None and target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if trim_start_sec is not None and trim_start_sec < 0:
        raise ValueError("trim_start_sec must be non-negative")
    if trim_end_sec is not None and trim_end_sec <= 0:
        raise ValueError("trim_end_sec must be positive")
    if trim_start_sec is not None and trim_end_sec is not None and trim_end_sec <= trim_start_sec:
        raise ValueError("trim_end_sec must be greater than trim_start_sec")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    sharpnesses: list[float] = []
    previous_gray: Image | None = None
    last_kept_ts: float | None = None
    min_interval = (1.0 / target_fps) - 1e-6 if target_fps is not None else 0.0
    width = 0
    height = 0
    with av.open(video) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or 0.0)
        for original_index, frame in enumerate(container.decode(video=0)):
            timestamp = _frame_timestamp(frame, original_index, rate)
            if trim_start_sec is not None and timestamp < trim_start_sec:
                continue
            if trim_end_sec is not None and timestamp > trim_end_sec:
                continue
            image = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            score = _sharpness(gray)
            sharpnesses.append(score)
            baseline = float(np.median(sharpnesses[-sharpness_window:]))
            if score < max(12.0, baseline * 0.35):
                continue
            if last_kept_ts is not None and target_fps is not None and (timestamp - last_kept_ts) < min_interval:
                continue
            motion = 1.0 if previous_gray is None else _parallax(previous_gray, gray)
            if previous_gray is not None and motion < min_parallax_frac:
                continue
            prepared, width, height = _prepare(image, max_long_side)
            frame_index = len(records)
            relative = Path("images") / f"frame_{frame_index:06d}.jpg"
            target = destination / relative
            target.parent.mkdir(exist_ok=True)
            _write_jpeg(target, prepared)
            records.append({
                "frame_idx": frame_index,
                "orig_video_frame": original_index,
                "timestamp_sec": timestamp,
                "image_relpath": relative.as_posix(),
            })
            previous_gray = gray
            last_kept_ts = timestamp
    if not records:
        raise ValueError("No sharp, sufficiently parallax-separated keyframes were found")
    if max_frames is not None:
        records = _subsample_records(destination, records, max_frames)
    index: dict[str, object] = {"frames": records, "width": width, "height": height}
    fd, temporary = tempfile.mkstemp(prefix="extract_index.", suffix=".json", dir=destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        index_path = destination / "extract_index.json"
        os.replace(temporary, index_path)
    except OSError:
        Path(temporary).unlink(missing_ok=True)
        raise
    return ExtractionResult(destination, index_path, index)


extract_keyframes = extract_clean_frames
