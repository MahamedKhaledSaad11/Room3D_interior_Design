from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import httpx
from numpy.typing import NDArray
from PIL import Image

from room3d_worker.common.progress import ProgressEvent, progress_sink
from room3d_worker.common.stage_budget import stage_budget
from room3d_worker.pipeline.s06_diffusion import (
    DiffusionConfig, _DiffusersFluxKleinBackend, generate_first_frame_seeds, propagate_masked_edits,
)


class RecordingBackend:
    def __init__(self, fail_at: int | None = None) -> None:
        self.calls = 0
        self.fail_at = fail_at

    def inpaint(self, image: NDArray[np.uint8], mask: NDArray[np.uint8], prompt: str,
                seed: int, image_reference: NDArray[np.uint8] | None = None) -> NDArray[np.uint8]:
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("inference interrupted")
        assert image.shape[:2] == mask.shape
        return np.full_like(image, 210)


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[dict[int, Path], dict[int, Path], Path]:
    frames: dict[int, Path] = {}
    masks: dict[int, Path] = {}
    for idx in range(9):
        frames[idx], masks[idx] = tmp_path / f"frame{idx}.png", tmp_path / f"mask{idx}.png"
        Image.fromarray(np.full((32, 48, 3), idx + 20, dtype=np.uint8)).save(frames[idx])
        mask = np.zeros((32, 48), dtype=np.uint8)
        if idx < 4:
            mask[8:24, 8:32] = 255
        Image.fromarray(mask).save(masks[idx])
    reference = tmp_path / "reference.png"
    Image.fromarray(np.full((32, 48, 3), 200, dtype=np.uint8)).save(reference)
    return frames, masks, reference


def run_edit(inputs: tuple[dict[int, Path], dict[int, Path], Path], output: Path,
             checkpoints: Path, backend: RecordingBackend, **overrides: Any) -> dict[str, Any]:
    frames, masks, reference = inputs
    options = {"prompt": "blue leather", "seed": 5, "config": DiffusionConfig(device="cpu", grid_2x2=False)}
    options.update(overrides)
    return propagate_masked_edits(frames, masks, 0, reference, output_dir=output, backend=backend,
                                 checkpoint_root=checkpoints, **options)


def test_partial_failure_resumes_in_new_work_directory(inputs: tuple[dict[int, Path], dict[int, Path], Path], tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    with pytest.raises(RuntimeError, match="interrupted"):
        run_edit(inputs, tmp_path / "attempt1", root, RecordingBackend(fail_at=2))
    assert len(list(root.glob("*/*.json"))) == 1
    backend = RecordingBackend()
    events: list[ProgressEvent] = []
    with progress_sink(events.append):
        metrics = run_edit(inputs, tmp_path / "attempt2", root, backend)
    assert backend.calls == 2
    assert metrics["frames_resumed"] == 1
    assert metrics["frames_inpainted"] == 3
    assert metrics["frames_skipped"] == 5
    assert metrics["max_unmasked_diff"] == 0
    assert events[0].percent < 50, "Copied originals must not inflate inference progress"
    assert events[0].telemetry["operations_total"] == 3
    for idx, frame in inputs[0].items():
        original = np.array(Image.open(frame))
        mask = np.array(Image.open(inputs[1][idx]))
        edited = np.array(Image.open(tmp_path / "attempt2" / "edited_frames" / f"{idx:06d}.png"))
        assert np.array_equal(edited[mask == 0], original[mask == 0])


@pytest.mark.parametrize("change", ["prompt", "seed", "mask", "source", "reference", "preset", "revision", "excluded"])
def test_changed_inputs_invalidate_checkpoint(inputs: tuple[dict[int, Path], dict[int, Path], Path], tmp_path: Path, change: str) -> None:
    root = tmp_path / "checkpoints"
    run_edit(inputs, tmp_path / "first", root, RecordingBackend(), checkpoint_context={"mask_revision": 1})
    unchanged = RecordingBackend()
    run_edit(inputs, tmp_path / "same", root, unchanged, checkpoint_context={"mask_revision": 1})
    assert unchanged.calls == 0
    kwargs: dict[str, Any] = {"checkpoint_context": {"mask_revision": 1}}
    if change in {"mask", "source", "reference"}:
        path = inputs[1][1] if change == "mask" else inputs[0][1] if change == "source" else inputs[2]
        pixels = np.array(Image.open(path))
        pixels[0, 0] = 255
        Image.fromarray(pixels).save(path)
    elif change == "preset":
        kwargs["config"] = DiffusionConfig(device="cpu", grid_2x2=False, quality_preset="quality")
    elif change == "revision":
        kwargs["checkpoint_context"] = {"mask_revision": 2}
    elif change == "excluded":
        kwargs["skip_frames"] = {3}
    else:
        kwargs[change] = "green" if change == "prompt" else 9
    changed = RecordingBackend()
    run_edit(inputs, tmp_path / "changed", root, changed, **kwargs)
    assert changed.calls == (2 if change == "excluded" else 3)


def test_corrupt_checkpoint_is_regenerated(inputs: tuple[dict[int, Path], dict[int, Path], Path], tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    run_edit(inputs, tmp_path / "first", root, RecordingBackend())
    next(root.glob("*/*.png")).write_bytes(b"partial image")
    backend = RecordingBackend()
    metrics = run_edit(inputs, tmp_path / "retry", root, backend)
    assert backend.calls == 1
    assert metrics["frames_resumed"] == 2


def test_grid_resume_requires_every_tile_receipt(inputs: tuple[dict[int, Path], dict[int, Path], Path], tmp_path: Path) -> None:
    Image.fromarray(np.full((32, 48), 255, dtype=np.uint8)).save(inputs[1][4])
    root = tmp_path / "checkpoints"
    config = DiffusionConfig(device="cpu", grid_2x2=True)
    initial = run_edit(inputs, tmp_path / "first", root, RecordingBackend(), config=config)
    assert initial["inference_operations"] == 1
    assert initial["frames_inpainted"] == 4
    next(root.glob("*/*.json")).unlink()
    backend = RecordingBackend()
    retry = run_edit(inputs, tmp_path / "partial_group", root, backend, config=config)
    assert retry["frames_resumed"] == 0
    assert backend.calls == 1
    final = RecordingBackend()
    run_edit(inputs, tmp_path / "full_group", root, final, config=config)
    assert final.calls == 0


def test_seed_preview_retry_reuses_completed_seeds(inputs: tuple[dict[int, Path], dict[int, Path], Path], tmp_path: Path) -> None:
    frame, mask = inputs[0][0], inputs[1][0]
    config = DiffusionConfig(device="cpu")
    root = tmp_path / "checkpoints"
    with pytest.raises(RuntimeError):
        generate_first_frame_seeds(frame, mask, "blue", [11, 22, 33], tmp_path / "seeds1",
                                   RecordingBackend(fail_at=2), config=config, checkpoint_root=root)
    backend = RecordingBackend()
    previews = generate_first_frame_seeds(frame, mask, "blue", [11, 22, 33], tmp_path / "seeds2",
                                          backend, config=config, checkpoint_root=root)
    assert len(previews) == 3
    assert backend.calls == 2


def test_budget_counts_nonempty_unexcluded_inferences(inputs: tuple[dict[int, Path], dict[int, Path], Path]) -> None:
    frames, masks, _ = inputs
    spec = {"frame_paths": {str(k): str(v) for k, v in frames.items()},
            "mask_paths": {str(k): str(v) for k, v in masks.items()}, "ref_frame_idx": 0,
            "skip_frames": [3], "quality_preset": "quality"}
    budget = stage_budget("diffusion_apply", spec, 3600, 900)
    assert budget.operations == 2
    assert budget.absolute_sec == 1800 + 2 * 50 * 30
    assert stage_budget("diffusion_apply", {**spec, "quality_preset": "fast"}, 3600, 900).absolute_sec < budget.absolute_sec
    assert stage_budget("train", {}, 3600, 900).absolute_sec == 3600


def test_official_klein_receives_mask_reference_and_real_step_callbacks() -> None:
    calls: dict[str, Any] = {}

    class Pipe:
        num_timesteps = 4

        def __call__(self, *, image: list[Image.Image], prompt: str, height: int, width: int,
                     num_inference_steps: int, guidance_scale: float, generator: object,
                     callback_on_step_end: Any, callback_on_step_end_tensor_inputs: list[str]) -> object:
            calls.update(image=image, prompt=prompt)
            for step in range(4):
                assert callback_on_step_end(self, step, None, {}) == {}
            return SimpleNamespace(images=[image[0]])

    torch = SimpleNamespace(Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: seed))
    backend = _DiffusersFluxKleinBackend(Pipe(), torch, DiffusionConfig(device="cpu"), "cpu")
    original = np.full((32, 48, 3), 20, dtype=np.uint8)
    mask = np.full((32, 48), 255, dtype=np.uint8)
    reference = np.full_like(original, 200)
    from room3d_worker.common.generation_progress import GenerationProgress, generation_scope

    events: list[ProgressEvent] = []
    with progress_sink(events.append), generation_scope(GenerationProgress(3, 3)):
        backend.inpaint(original, mask, "blue leather", 5, reference)
    assert len(calls["image"]) == 3
    assert np.array_equal(np.array(calls["image"][1])[:, :, 0], mask)
    assert np.array_equal(np.array(calls["image"][2]), reference)
    assert "blue leather" in calls["prompt"]
    assert [event.telemetry["denoising_step"] for event in events] == [1, 2, 3, 4]
    assert all("eta_sec" in event.telemetry for event in events)


def test_worker_restart_restores_uploaded_checkpoint_without_publishing_partial_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scene_fixtures import _setup_base_scene
    from room3d_worker.client import TaskClaim, WorkerClient
    from room3d_worker.runner import WorkerBackends, WorkerRunner

    monkeypatch.setattr("room3d_worker.runner._stage_device", lambda ordinal: "cpu")
    storage = tmp_path / "storage"
    scene, edited, masks = _setup_base_scene(tmp_path / "fixture")
    shutil.copytree(scene, storage / "proj-1/v1")
    edit_root = storage / "proj-1/edits/edit1"
    (edit_root / "masks").mkdir(parents=True)
    for idx in (0, 1):
        shutil.copyfile(masks / f"{idx:05d}.png", edit_root / "masks" / f"{idx:06d}.png")
    shutil.copyfile(edited / "00000.png", edit_root / "preview.png")
    payload = {"edit_id": "edit1", "base_version_id": "v1", "ref_frame_idx": 0,
               "preview_key": "proj-1/edits/edit1/preview.png", "prompt": "blue", "selected_seed": 5}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    with WorkerClient("http://testserver", "secret", transport=transport) as client:
        first = WorkerRunner(client, work_root=tmp_path / "worker1", storage_dir=storage,
                             backends=WorkerBackends(inpaint_backend=RecordingBackend()))
        publish = first.storage.put_many

        def interrupted(files: list[tuple[str, Path]]) -> None:
            publish(files)
            raise RuntimeError("session ended after checkpoint upload")

        monkeypatch.setattr(first.storage, "put_many", interrupted)
        with pytest.raises(RuntimeError, match="session ended"):
            first._handle_diffusion_apply(TaskClaim("task1", "a1", "proj-1", None, "diffusion_apply", payload))
        assert not (edit_root / "edited_frames").exists()
        backend = RecordingBackend()
        second = WorkerRunner(client, work_root=tmp_path / "worker2", storage_dir=storage,
                              backends=WorkerBackends(inpaint_backend=backend))
        result, manifest = second._handle_diffusion_apply(TaskClaim("task2", "a2", "proj-1", None, "diffusion_apply", payload))
    assert result["frames_resumed"] == 1
    assert backend.calls == 0
    assert manifest is None
    assert len(list((edit_root / "edited_frames").glob("*.png"))) == 2
