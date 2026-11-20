from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import pytest

from room3d_worker.common.errors import StageExecutionError
from room3d_worker.pipeline.s06_diffusion import (
    DiffusionConfig,
    _DiffusersFluxKleinBackend,
    _DiffusersSdxlBackend,
    assert_unmasked_unchanged,
    composite_masked_region,
    composite_strict_mask,
    generate_first_frame_seeds,
    load_diffusion_backend,
    pad_to_multiple,
    propagate_masked_edits,
    resolve_t4_dtype,
)


class _RecordingDiffusionBackend:
    def __init__(self, fill_rgb: tuple[int, int, int] = (200, 40, 20)) -> None:
        self.fill_rgb = fill_rgb
        self.calls: list[tuple[tuple[int, int], int, bool, tuple[int, int] | None]] = []

    def inpaint(
        self,
        image: NDArray[np.uint8],
        mask: NDArray[np.uint8],
        prompt: str,
        seed: int,
        image_reference: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        _ = (mask, prompt)
        ref_hw = image_reference.shape[:2] if image_reference is not None else None
        self.calls.append((image.shape[:2], int(seed), image_reference is not None, ref_hw))
        return np.full_like(image, self.fill_rgb, dtype=np.uint8)


def test_t4_compute_capability_and_license_guard() -> None:
    assert resolve_t4_dtype((7, 5)) == "float16"
    assert resolve_t4_dtype(None) == "float32"
    assert resolve_t4_dtype((8, 6)) == "bfloat16"
    with pytest.raises(StageExecutionError, match="license guardrails"):
        DiffusionConfig(model_id="black-forest-labs/FLUX.2-klein-base-9B")
    with pytest.raises(StageExecutionError, match="license guardrails"):
        DiffusionConfig(model_id="Qwen/Qwen-Image-2.1")


def test_strict_mask_compositor_validates_types_and_zero_unmasked_drift() -> None:
    rng = np.random.default_rng(7)
    orig = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    edited = np.full_like(orig, 255)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:40, 20:44] = 255

    composited = composite_strict_mask(orig, edited, mask, feather_px=3)
    assert assert_unmasked_unchanged(orig, composited, mask) == 0
    assert np.any(composited[mask == 255] != orig[mask == 255])

    with pytest.raises(StageExecutionError, match="2D uint8"):
        composite_strict_mask(orig, edited, mask.astype(np.int64))


def test_grid_oom_guard_empty_masks_and_ref_padding(tmp_path: Path) -> None:
    frame_paths: dict[int, Path] = {}
    mask_paths: dict[int, Path] = {}
    for idx in range(6):
        img = np.full((50, 70, 3), 20 + idx * 10, dtype=np.uint8)
        mask = np.zeros((50, 70), dtype=np.uint8)
        if idx != 5:
            mask[10:30, 15:45] = 255
        fp = tmp_path / f"f_{idx}.png"
        mp = tmp_path / f"m_{idx}.png"
        Image.fromarray(img).save(fp, format="PNG")
        Image.fromarray(mask).save(mp, format="PNG")
        frame_paths[idx] = fp
        mask_paths[idx] = mp

    seed_backend = _RecordingDiffusionBackend(fill_rgb=(200, 40, 20))
    seeds = generate_first_frame_seeds(frame_paths[0], mask_paths[0], "red velvet chair", [101, 202], tmp_path / "seeds", seed_backend)
    assert len(seeds) == 2

    ref_path = Path(seeds[0]["path"])
    prop_backend = _RecordingDiffusionBackend(fill_rgb=(10, 20, 240))
    out_dir = tmp_path / "out_grid"
    metrics = propagate_masked_edits(
        frame_paths=frame_paths,
        mask_paths=mask_paths,
        ref_frame_idx=0,
        ref_edited_path=ref_path,
        prompt="red velvet chair",
        seed=101,
        output_dir=out_dir,
        backend=prop_backend,
        skip_frames=set(),
        config=DiffusionConfig(grid_2x2=True),
    )
    # Frame 0 preserves the approved seed preview; frames 1..4 execute as one 2x2 grid batch; frame 5 is empty-mask.
    assert metrics["frames_edited"] == 5
    assert metrics["frames_inpainted"] == 4
    assert metrics["frames_skipped"] == 1
    assert metrics["max_unmasked_diff"] == 0
    assert metrics["frames"][0]["method"] == "reference_seed"
    assert metrics["frames"][5]["skipped"] is True and metrics["frames"][5]["method"] == "unchanged_original"

    ref_saved = np.asarray(Image.open(out_dir / "edited_frames" / "000000.png"))
    ref_orig = np.asarray(Image.open(frame_paths[0]))
    ref_seed = np.asarray(Image.open(ref_path))
    ref_mask = np.asarray(Image.open(mask_paths[0]))
    expected_ref = np.where(ref_mask[..., None] > 0, ref_seed, ref_orig).astype(np.uint8)
    assert np.array_equal(ref_saved, expected_ref)
    assert np.array_equal(np.asarray(Image.open(out_dir / "edited_frames" / "000005.png")), np.asarray(Image.open(frame_paths[5])))

    grid_call = next(c for c in prop_backend.calls if c[0] == (112, 144))
    assert grid_call[2] is True
    assert grid_call[3] == (112, 144)


def test_sequential_mode_and_oversized_grid_fallback(tmp_path: Path) -> None:
    frame_paths: dict[int, Path] = {}
    mask_paths: dict[int, Path] = {}
    for idx in range(4):
        img = np.full((600, 600, 3), 50, dtype=np.uint8)
        mask = np.zeros((600, 600), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        fp, mp = tmp_path / f"big_{idx}.png", tmp_path / f"big_m_{idx}.png"
        Image.fromarray(img).save(fp, format="PNG")
        Image.fromarray(mask).save(mp, format="PNG")
        frame_paths[idx], mask_paths[idx] = fp, mp

    backend = _RecordingDiffusionBackend()
    padded_img, padded_mask, orig_hw = pad_to_multiple(np.zeros((50, 70, 3), dtype=np.uint8), np.zeros((50, 70), dtype=np.uint8), 16)
    assert (padded_img.shape[:2], padded_mask.shape[:2], orig_hw) == ((64, 80), (64, 80), (50, 70))

    metrics = propagate_masked_edits(
        frame_paths=frame_paths,
        mask_paths=mask_paths,
        ref_frame_idx=0,
        ref_edited_path=frame_paths[0],
        prompt="blue armchair",
        seed=7,
        output_dir=tmp_path / "out_big",
        backend=backend,
        config=DiffusionConfig(grid_2x2=True),
    )
    assert metrics["frames_edited"] == 4
    assert metrics["frames_inpainted"] == 3
    assert all(call[0][0] < 1200 for call in backend.calls)


def test_load_diffusion_backend_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _FakePipe:
        def to(self, device: str) -> None:
            calls.append(f"to:{device}")

        def enable_vae_tiling(self) -> None:
            calls.append("tiling")

        def enable_vae_slicing(self) -> None:
            calls.append("slicing")

    class _FakeFlux:
        @classmethod
        def from_pretrained(cls, _repo: str, torch_dtype: Any, token: Any = None, revision: str | None = None) -> _FakePipe:
            calls.append(f"dtype:{torch_dtype}")
            return _FakePipe()

    fake_torch = SimpleNamespace(
        float32="fp32",
        float16="fp16",
        bfloat16="bf16",
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    fake_diffusers = SimpleNamespace(Flux2KleinInpaintPipeline=_FakeFlux)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    backend = load_diffusion_backend(DiffusionConfig(device="cuda:0"))
    assert backend is not None
    assert "dtype:fp32" in calls
    assert "to:cpu" in calls
    assert "slicing" in calls


def test_load_diffusion_backend_respects_model_variant_and_default_device(monkeypatch: pytest.MonkeyPatch) -> None:
    assert DiffusionConfig().device == "cuda:0"
    loaded: list[tuple[str, str]] = []

    class _FakePipe:
        def to(self, _device: str) -> None:
            return None

        def enable_vae_tiling(self) -> None:
            return None

        def enable_vae_slicing(self) -> None:
            return None

    class _FakeFlux:
        @classmethod
        def from_pretrained(cls, repo: str, torch_dtype: Any, token: Any = None, revision: str | None = None) -> _FakePipe:
            _ = (torch_dtype, token)
            loaded.append(("flux", repo))
            return _FakePipe()

    class _FakeSdxl:
        @classmethod
        def from_pretrained(cls, repo: str, torch_dtype: Any, token: Any = None, revision: str | None = None) -> _FakePipe:
            _ = (torch_dtype, token)
            loaded.append(("sdxl", repo))
            return _FakePipe()

    fake_torch = SimpleNamespace(
        float32="fp32",
        float16="fp16",
        bfloat16="bf16",
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    fake_diffusers = SimpleNamespace(Flux2KleinInpaintPipeline=_FakeFlux, StableDiffusionXLInpaintPipeline=_FakeSdxl)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    sdxl_backend = load_diffusion_backend(DiffusionConfig(model_variant="sdxl_inpaint_t4"))
    assert isinstance(sdxl_backend, _DiffusersSdxlBackend)
    assert loaded[-1] == ("sdxl", DiffusionConfig().fallback_model_id)

    flux_backend = load_diffusion_backend(DiffusionConfig(model_variant="flux2_klein_4b"))
    assert isinstance(flux_backend, _DiffusersFluxKleinBackend)
    assert loaded[-1] == ("flux", DiffusionConfig().model_id)

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(StableDiffusionXLInpaintPipeline=_FakeSdxl))
    fallback_backend = load_diffusion_backend(DiffusionConfig(model_variant="flux2_klein_4b", allow_model_fallback=True))
    assert isinstance(fallback_backend, _DiffusersSdxlBackend)


def test_skip_ref_frame_no_double_feather_negative_prompt_and_sdxl_color_match(tmp_path: Path) -> None:
    orig = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    # Already-feathered boundary at value 100 must NOT be feathered down to 50 inside propagate_masked_edits.
    feathered_seed = orig.copy()
    feathered_seed[mask == 255] = 100
    fp = tmp_path / "f0.png"
    mp = tmp_path / "m0.png"
    sp = tmp_path / "seed.png"
    Image.fromarray(orig).save(fp, format="PNG")
    Image.fromarray(mask).save(mp, format="PNG")
    Image.fromarray(feathered_seed).save(sp, format="PNG")

    class _DummyBackend:
        def inpaint(self, image: NDArray[np.uint8], _m: NDArray[np.uint8], _p: str, seed: int, image_reference=None):
            _ = (seed, image_reference)
            return image

    metrics_keep = propagate_masked_edits(
        frame_paths={0: fp},
        mask_paths={0: mp},
        ref_frame_idx=0,
        ref_edited_path=sp,
        prompt="chair",
        seed=1,
        output_dir=tmp_path / "keep_ref",
        backend=_DummyBackend(),
        config=DiffusionConfig(feather_px=3, grid_2x2=False),
    )
    assert metrics_keep["frames"][0]["method"] == "reference_seed"
    saved = np.asarray(Image.open(tmp_path / "keep_ref" / "edited_frames" / "000000.png"))
    assert int(saved[8, 8, 0]) == 100

    metrics_skip = propagate_masked_edits(
        frame_paths={0: fp},
        mask_paths={0: mp},
        ref_frame_idx=0,
        ref_edited_path=sp,
        prompt="chair",
        seed=1,
        output_dir=tmp_path / "skip_ref",
        backend=_DummyBackend(),
        skip_frames={0},
        config=DiffusionConfig(feather_px=3, grid_2x2=False),
    )
    assert metrics_skip["frames"][0]["skipped"] is True
    assert metrics_skip["frames"][0]["method"] == "unchanged_original"
    assert np.array_equal(np.asarray(Image.open(tmp_path / "skip_ref" / "edited_frames" / "000000.png")), orig)

    seen_kwargs: list[dict[str, Any]] = []

    class _MockPipe:
        def __call__(self, **kwargs: Any) -> Any:
            seen_kwargs.append(kwargs)
            # Return flat blue RGB image inside mask; SDXL reference color match should shift toward red reference.
            img = Image.fromarray(np.full((32, 32, 3), (20, 40, 210), dtype=np.uint8))
            return SimpleNamespace(images=[img])

    class _Gen:
        def __init__(self, device: str) -> None:
            _ = device

        def manual_seed(self, seed: int) -> int:
            return seed

    fake_torch = SimpleNamespace(Generator=_Gen)
    cfg = DiffusionConfig(model_variant="sdxl_inpaint_t4", negative_prompt="blurry, distorted")
    sdxl = _DiffusersSdxlBackend(_MockPipe(), fake_torch, cfg, exec_device="cpu")
    ref_img = np.full((32, 32, 3), (220, 30, 30), dtype=np.uint8)
    matched = sdxl.inpaint(orig, mask, "red chair", seed=7, image_reference=ref_img, negative_prompt=cfg.negative_prompt)
    assert seen_kwargs[-1]["negative_prompt"] == "blurry, distorted"
    assert "image_reference" not in seen_kwargs[-1]
    assert sdxl.last_reference_conditioning == "lab_statistics_only"
    assert matched[16, 16, 0] > matched[16, 16, 2]

    class _StrictFluxPipe:
        def __call__(
            self,
            *,
            prompt: str,
            image: Image.Image,
            mask_image: Image.Image,
            num_inference_steps: int,
            guidance_scale: float,
            strength: float,
            generator: Any,
            image_reference: Image.Image | None = None,
        ) -> Any:
            seen_kwargs.append({"prompt": prompt, "has_ref": image_reference is not None})
            return SimpleNamespace(images=[image.copy()])

    flux = _DiffusersFluxKleinBackend(_StrictFluxPipe(), fake_torch, DiffusionConfig(negative_prompt="bad"), exec_device="cpu")
    generate_first_frame_seeds(fp, mp, "red chair", [5], tmp_path / "neg_seeds", flux, negative_prompt="lowres")
    assert "negative_prompt" not in seen_kwargs[-1]


def test_sdxl_color_match_samples_moved_reference_mask(tmp_path: Path) -> None:
    # Reference object is top-left (red) on dark-green background; target object moved to bottom-right.
    ref_orig = np.full((32, 32, 3), (20, 140, 20), dtype=np.uint8)
    ref_edit = ref_orig.copy()
    ref_mask = np.zeros((32, 32), dtype=np.uint8)
    ref_mask[2:12, 2:12] = 255
    ref_edit[ref_mask == 255] = (230, 30, 30)

    tgt_orig = np.full((32, 32, 3), (20, 140, 20), dtype=np.uint8)
    tgt_mask = np.zeros((32, 32), dtype=np.uint8)
    tgt_mask[20:30, 20:30] = 255

    f0, f1 = tmp_path / "f0.png", tmp_path / "f1.png"
    m0, m1 = tmp_path / "m0.png", tmp_path / "m1.png"
    ref_p = tmp_path / "ref_seed.png"
    Image.fromarray(ref_orig).save(f0, format="PNG")
    Image.fromarray(tgt_orig).save(f1, format="PNG")
    Image.fromarray(ref_mask).save(m0, format="PNG")
    Image.fromarray(tgt_mask).save(m1, format="PNG")
    Image.fromarray(ref_edit).save(ref_p, format="PNG")

    class _FakeSdxlPipe:
        def __call__(self, *, image: Image.Image, **_kwargs: Any) -> Any:
            return SimpleNamespace(images=[Image.fromarray(np.full((image.height, image.width, 3), (30, 40, 200), dtype=np.uint8))])

    class _Gen:
        def __init__(self, device: str) -> None:
            _ = device

        def manual_seed(self, seed: int) -> int:
            return seed

    fake_torch = SimpleNamespace(Generator=_Gen)
    sdxl = _DiffusersSdxlBackend(_FakeSdxlPipe(), fake_torch, DiffusionConfig(model_variant="sdxl_inpaint_t4"), exec_device="cpu")
    matched = sdxl.inpaint(tgt_orig, tgt_mask, "red chair", seed=42, image_reference=ref_edit, ref_mask=ref_mask)
    tgt_saved = composite_strict_mask(tgt_orig, matched, tgt_mask, feather_px=0)
    # Bottom-right masked interior should shift to red (ref_mask top-left), not dark green background.
    r_val, g_val, _ = tgt_saved[25, 25]
    assert int(r_val) > int(g_val) + 80
    # Unmasked region stays identically equal to tgt_orig.
    assert np.array_equal(tgt_saved[5, 5], tgt_orig[5, 5])
    with pytest.raises(StageExecutionError, match="not reference-image conditioning"):
        propagate_masked_edits({0: f0, 1: f1}, {0: m0, 1: m1}, 0, ref_p,
                               "red chair", 42, tmp_path / "moved_out", sdxl,
                               config=DiffusionConfig(device="cpu", grid_2x2=False))


def test_t4_quantization_text_encoder_offload_and_vram_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    assert resolve_t4_dtype((7, 5)) == "float16"
    assert resolve_t4_dtype((8, 0)) == "bfloat16"

    events: list[tuple[str, Any]] = []

    class _Pipe:
        def __init__(self) -> None:
            self.transformer = SimpleNamespace(name="flux_transformer")

        def enable_model_cpu_offload(self, gpu_id: int = 0) -> None:
            events.append(("model_cpu_offload", gpu_id))

        def enable_sequential_cpu_offload(self, gpu_id: int = 0) -> None:
            events.append(("seq_cpu_offload", gpu_id))

        def enable_vae_tiling(self) -> None:
            events.append(("vae_tiling", True))

        def enable_vae_slicing(self) -> None:
            events.append(("vae_slicing", True))

        def to(self, device: str) -> None:
            events.append(("to", device))

        def __call__(self, **kwargs: Any) -> Any:
            _ = kwargs
            return SimpleNamespace(images=[Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8))])

    class _FakeFlux:
        @classmethod
        def from_pretrained(cls, repo: str, **kwargs: Any) -> _Pipe:
            events.append(("from_pretrained", (repo, kwargs)))
            return _Pipe()

    class _PipeQuantCfg:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _Gen:
        def __init__(self, device: str) -> None:
            _ = device

        def manual_seed(self, seed: int) -> int:
            return seed

    fake_torch = SimpleNamespace(
        float16="fp16",
        float32="fp32",
        bfloat16="bf16",
        Generator=_Gen,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            device_count=lambda: 1,
            get_device_capability=lambda _idx: (7, 5),
            reset_peak_memory_stats=lambda _idx: events.append(("reset_vram", _idx)),
            max_memory_allocated=lambda _idx: 512 * 1024 * 1024,
        ),
    )
    fake_diffusers = SimpleNamespace(
        Flux2KleinInpaintPipeline=_FakeFlux,
        PipelineQuantizationConfig=_PipeQuantCfg,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace())

    backend_nf4 = load_diffusion_backend(
        DiffusionConfig(model_variant="flux2_klein_4b", quantization="nf4", offload_text_encoder=True, device="cuda:0")
    )
    assert isinstance(backend_nf4, _DiffusersFluxKleinBackend)
    assert backend_nf4.quantization_mode == "nf4"
    assert backend_nf4.torch_dtype_name == "float16"
    assert ("model_cpu_offload", 0) in events
    pretrained_kwargs = next(val[1] for name, val in events if name == "from_pretrained")
    assert pretrained_kwargs["dtype"] == "fp16"
    assert isinstance(pretrained_kwargs["quantization_config"], _PipeQuantCfg)

    # Test INT8 quanto fallback when bitsandbytes is absent and optimum.quanto is present.
    events.clear()
    fake_quanto = SimpleNamespace(
        qint8="qint8_spec",
        quantize=lambda mod, weights: events.append(("quanto_quantize", (mod.name, weights))),
        freeze=lambda mod: events.append(("quanto_freeze", mod.name)),
    )
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    monkeypatch.setitem(sys.modules, "optimum", SimpleNamespace(quanto=fake_quanto))
    monkeypatch.setitem(sys.modules, "optimum.quanto", fake_quanto)

    backend_int8 = load_diffusion_backend(
        DiffusionConfig(model_variant="flux2_klein_4b", quantization="nf4", offload_text_encoder=True, device="cuda:0")
    )
    assert isinstance(backend_int8, _DiffusersFluxKleinBackend)
    assert backend_int8.quantization_mode == "int8_quanto"
    assert ("quanto_quantize", ("flux_transformer", "qint8_spec")) in events
    assert ("quanto_freeze", "flux_transformer") in events

    # Verify PipelineQuantizationConfig missing does NOT fall back to transformers.BitsAndBytesConfig.
    events.clear()
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "optimum.quanto", None)
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(Flux2KleinInpaintPipeline=_FakeFlux))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(BitsAndBytesConfig=lambda **_kw: object()))
    backend_fp16_offload = load_diffusion_backend(
        DiffusionConfig(model_variant="flux2_klein_4b", quantization="nf4", offload_text_encoder=True, device="cuda:0")
    )
    assert isinstance(backend_fp16_offload, _DiffusersFluxKleinBackend)
    assert backend_fp16_offload.quantization_mode == "fp16_offload"
    no_quant_kwargs = next(val[1] for name, val in events if name == "from_pretrained")
    assert "quantization_config" not in no_quant_kwargs

    # Verify telemetry fields (sec_per_frame, peak_vram_mb) and uninitialized CUDA guard.
    fp = tmp_path / "f0.png"
    mp = tmp_path / "m0.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(fp)
    m = np.zeros((32, 32), dtype=np.uint8)
    m[8:24, 8:24] = 255
    Image.fromarray(m).save(mp)

    seeds = generate_first_frame_seeds(
        fp, mp, "chair", [101, 202], tmp_path / "seeds", backend_nf4, config=DiffusionConfig(device="cuda:0")
    )
    assert seeds[0]["sec_per_frame"] > 0.0
    assert seeds[0]["peak_vram_mb"] == pytest.approx(512.0, abs=0.01)

    fake_torch.cuda.is_initialized = lambda: False
    metrics = propagate_masked_edits(
        frame_paths={0: fp},
        mask_paths={0: mp},
        ref_frame_idx=0,
        ref_edited_path=Path(seeds[0]["path"]),
        prompt="chair",
        seed=101,
        output_dir=tmp_path / "prop_out",
        backend=backend_nf4,
        config=DiffusionConfig(device="cuda:0", quantization="nf4"),
    )
    assert metrics["peak_vram_mb"] == 0.0
    assert metrics["sec_per_frame"] is None
    assert metrics["sec_per_input_frame"] > 0.0
    disk_metrics = json.loads((tmp_path / "prop_out" / "diffusion_metrics.json").read_text(encoding="utf-8"))
    assert disk_metrics["quantization"] == "nf4"
    assert "sec_per_frame" in disk_metrics and "peak_vram_mb" in disk_metrics


def test_quantized_load_failure_falls_back_to_fp16_and_sdxl_on_gated_flux(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, bool]] = []

    class _Pipe:
        def enable_model_cpu_offload(self, gpu_id: int = 0) -> None:
            _ = gpu_id

        def enable_vae_tiling(self) -> None:
            return None

        def enable_vae_slicing(self) -> None:
            return None

    class _BnbFailingFlux:
        @classmethod
        def from_pretrained(cls, repo: str, **kwargs: Any) -> _Pipe:
            has_q = "quantization_config" in kwargs
            calls.append(("flux_bnb", repo, has_q))
            if has_q:
                raise RuntimeError("libbitsandbytes_cuda130.so not found")
            return _Pipe()

    class _PipeQuantCfg:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_torch = SimpleNamespace(
        float16="fp16",
        float32="fp32",
        bfloat16="bf16",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            device_count=lambda: 1,
            get_device_capability=lambda _idx: (7, 5),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinInpaintPipeline=_BnbFailingFlux, PipelineQuantizationConfig=_PipeQuantCfg),
    )

    backend_fp16 = load_diffusion_backend(DiffusionConfig(model_variant="flux2_klein_4b", quantization="nf4", device="cuda:0"))
    assert isinstance(backend_fp16, _DiffusersFluxKleinBackend)
    assert backend_fp16.quantization_mode == "fp16_offload"
    assert calls == [
        ("flux_bnb", DiffusionConfig.model_id, True),
        ("flux_bnb", DiffusionConfig.model_id, False),
    ]

    # When Flux repo raises OSError (gated without HF_TOKEN), fall back cleanly to public SDXL.
    calls.clear()

    class _GatedFlux:
        @classmethod
        def from_pretrained(cls, repo: str, **kwargs: Any) -> _Pipe:
            calls.append(("gated_flux", repo, "quantization_config" in kwargs))
            raise OSError("401 Gated repo")

    class _WorkingSdxl:
        @classmethod
        def from_pretrained(cls, repo: str, **kwargs: Any) -> _Pipe:
            calls.append(("sdxl", repo, "quantization_config" in kwargs))
            return _Pipe()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            Flux2KleinInpaintPipeline=_GatedFlux,
            StableDiffusionXLInpaintPipeline=_WorkingSdxl,
            PipelineQuantizationConfig=_PipeQuantCfg,
        ),
    )
    backend_sdxl = load_diffusion_backend(DiffusionConfig(model_variant="flux2_klein_4b", quantization="nf4", device="cuda:0", allow_model_fallback=True))
    assert isinstance(backend_sdxl, _DiffusersSdxlBackend)
    assert calls[-1][0] == "sdxl" and calls[-1][1] == DiffusionConfig.fallback_model_id
