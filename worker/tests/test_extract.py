from __future__ import annotations

import json
from pathlib import Path

import av
import cv2
import numpy as np
import pytest
from PIL import Image

from room3d_worker.client import StaleAttemptError, WorkerClient
from room3d_worker.pipeline.s01_extract import extract_clean_frames, extract_keyframes


def _video(path: Path) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 322, 242, "yuv420p"
        for index in range(12):
            if index < 3:
                image = np.full((242, 322, 3), 80, np.uint8)
            elif index < 6:
                image = np.full((242, 322, 3), 80 + index, np.uint8)
            else:
                image = np.zeros((242, 322, 3), np.uint8)
                shift = (index - 6) * 24
                cv2.rectangle(image, (30 + shift, 30), (150 + shift, 180), (255, 120, 20), -1)
                cv2.line(image, (0, 240 - index * 8), (321, index * 8), (20, 240, 255), 4)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _sharp_moving_video(path: Path, count: int = 24, fps: int = 8) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for index in range(count):
            image = np.zeros((240, 320, 3), np.uint8)
            shift = index * 8
            cv2.rectangle(image, (20 + shift, 30), (100 + shift, 170), (255, 120, 20), -1)
            cv2.line(image, (0, 239 - index * 6), (319, index * 6), (20, 240, 255), 4)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_extract_keyframes_drops_blurry_duplicates_and_writes_clean_index(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "frames"
    _video(source)

    result = extract_keyframes(source, output, max_long_side=640, min_parallax_frac=0.04)

    index = json.loads(result.index_path.read_text(encoding="utf-8"))
    assert index == result.index
    assert len(index["frames"]) >= 2
    assert all(item["orig_video_frame"] >= 6 for item in index["frames"])
    assert [item["frame_idx"] for item in index["frames"]] == list(range(len(index["frames"])))
    for item in index["frames"]:
        image_path = output / item["image_relpath"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        assert image.shape[1] % 16 == 0 and image.shape[0] % 16 == 0
        with av.open(str(image_path)) as container:
            stream = container.streams.video[0]
            assert not container.streams.audio
            assert not stream.metadata.get("exif")
            assert not stream.metadata.get("gps")
        assert not Image.open(image_path).getexif()


def test_worker_client_claim_heartbeat_and_stale_completion() -> None:
    responses = {
        "/api/tasks/claim": (200, {
            "task_id": "t1", "attempt_id": "a1", "task_type": "segment_refine", "payload": {},
        }),
        "/api/tasks/t1/heartbeat": (200, {"ok": True}),
        "/api/tasks/t1/complete": (409, {"detail": "stale attempt"}),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        status, payload = responses[request.url.path]
        return httpx.Response(status, json=payload)

    import httpx

    client = WorkerClient("https://backend.test", "secret", transport=httpx.MockTransport(handler))
    task = client.claim("worker-1", ["segment_refine"], interactive_only=True)
    assert task.task_id == "t1"
    assert client.heartbeat(task) == {"ok": True}
    with pytest.raises(StaleAttemptError):
        client.complete(task, result_payload={})


def test_extract_clean_frames_subsamples_target_fps_and_max_frames(tmp_path: Path) -> None:
    source = tmp_path / "moving.mp4"
    _sharp_moving_video(source, count=24, fps=8)

    fps_dir = tmp_path / "fps2"
    fps_result = extract_clean_frames(source, fps_dir, max_long_side=640, min_parallax_frac=0.0, target_fps=2.0)
    fps_frames = fps_result.index["frames"]
    assert isinstance(fps_frames, list)
    assert len(fps_frames) == 6
    assert [item["frame_idx"] for item in fps_frames] == list(range(6))
    timestamps = [float(item["timestamp_sec"]) for item in fps_frames]
    assert all(b - a >= 0.5 - 1e-3 for a, b in zip(timestamps, timestamps[1:]))

    max_dir = tmp_path / "max5"
    max_result = extract_clean_frames(
        source,
        max_dir,
        max_long_side=640,
        min_parallax_frac=0.0,
        target_fps=None,
        max_frames=5,
    )
    max_frames = max_result.index["frames"]
    assert isinstance(max_frames, list)
    assert len(max_frames) == 5
    assert [item["frame_idx"] for item in max_frames] == [0, 1, 2, 3, 4]
    assert max_frames[0]["orig_video_frame"] == 0
    assert max_frames[-1]["orig_video_frame"] == 23
    saved_jpegs = sorted((max_dir / "images").glob("*.jpg"))
    assert [p.name for p in saved_jpegs] == [f"frame_{i:06d}.jpg" for i in range(5)]

    with pytest.raises(ValueError, match="target_fps"):
        extract_clean_frames(source, tmp_path / "bad_fps", target_fps=0.0)
    with pytest.raises(ValueError, match="max_frames"):
        extract_clean_frames(source, tmp_path / "bad_max", max_frames=0)

    # Default target_fps=4.0 enforces <=4 FPS on high-FPS video out of the box
    high_fps_video = tmp_path / "high_fps.mp4"
    _sharp_moving_video(high_fps_video, count=24, fps=24)
    default_result = extract_clean_frames(high_fps_video, tmp_path / "default_cap", max_long_side=640, min_parallax_frac=0.0)
    default_frames = default_result.index["frames"]
    assert isinstance(default_frames, list)
    assert 3 <= len(default_frames) <= 5


def test_extract_clean_frames_respects_trim_window(tmp_path: Path) -> None:
    source = tmp_path / "trim.mp4"
    _sharp_moving_video(source, count=24, fps=8)

    trimmed = extract_clean_frames(
        source,
        tmp_path / "trimmed",
        max_long_side=640,
        min_parallax_frac=0.0,
        target_fps=None,
        trim_start_sec=0.5,
        trim_end_sec=1.5,
    )
    frames = trimmed.index["frames"]
    assert isinstance(frames, list)
    assert len(frames) == 9
    assert [int(f["frame_idx"]) for f in frames] == list(range(9))
    assert [int(f["orig_video_frame"]) for f in frames] == list(range(4, 13))
    assert all(0.5 <= float(f["timestamp_sec"]) <= 1.5 for f in frames)

    with pytest.raises(ValueError, match="trim_start_sec"):
        extract_clean_frames(source, tmp_path / "bad_trim_start", trim_start_sec=-0.1)
    with pytest.raises(ValueError, match="trim_end_sec"):
        extract_clean_frames(source, tmp_path / "bad_trim_order", trim_start_sec=1.0, trim_end_sec=1.0)
