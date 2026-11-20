from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol
from collections.abc import Callable

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from room3d_worker.common.errors import StageExecutionError
from room3d_worker.common.diffusion_checkpoints import DiffusionCheckpoints, diffusion_identity
from room3d_worker.common.generation_progress import GenerationProgress, generation_scope, generation_step
from room3d_worker.common.progress import report_progress

LOGGER = logging.getLogger(__name__)

RGBImage = NDArray[np.uint8]
MaskArray = NDArray[np.uint8]
MAX_GRID_PIXELS = 1024 * 1024
APPROVED_MODELS = {
    "black-forest-labs/FLUX.2-klein-4B": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    "black-forest-labs/FLUX.2-klein-base-4B": "a3b4f4849157f664bdbc776fd7453c2783562f4d",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1": "115134f363124c53c7d878647567d04daf26e41e",
}


ModelVariant = Literal["flux2_klein_4b", "sdxl_inpaint_t4"]
QuantizationMode = Literal["nf4", "int8", "none"]


@dataclass(frozen=True)
class DiffusionConfig:
    model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    fallback_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    model_variant: ModelVariant = "flux2_klein_4b"
    device: str = "cuda:0"
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    strength: float = 0.85
    feather_px: int = 3
    use_sequential_offload: bool = False
    grid_2x2: bool = True
    negative_prompt: str = ""
    quantization: QuantizationMode = "nf4"
    offload_text_encoder: bool = True
    quality_preset: Literal["fast", "quality"] = "fast"
    allow_model_fallback: bool = False
    model_revision: str | None = None

    def __post_init__(self) -> None:
        if self.quality_preset not in {"fast", "quality"}:
            raise ValueError("quality_preset must be fast or quality")
        if self.quality_preset == "quality" and self.model_variant == "flux2_klein_4b":
            object.__setattr__(self, "model_id", "black-forest-labs/FLUX.2-klein-base-4B")
            object.__setattr__(self, "num_inference_steps", 50)
            object.__setattr__(self, "guidance_scale", 4.0)
        if self.model_variant == "sdxl_inpaint_t4":
            object.__setattr__(self, "num_inference_steps", 30)
            object.__setattr__(self, "guidance_scale", 7.5)
        if self.model_id.endswith("base-4B") and self.num_inference_steps < 30:
            raise ValueError("Undistilled FLUX base needs a quality schedule; choose quality_preset='quality'")
        if self.model_variant not in {"flux2_klein_4b", "sdxl_inpaint_t4"}:
            raise ValueError(f"unsupported model_variant: {self.model_variant}")
        if self.quantization not in {"nf4", "int8", "none"}:
            raise ValueError(f"unsupported quantization: {self.quantization}")
        if not 0.0 < self.strength <= 1.0:
            raise ValueError("strength must be in (0.0, 1.0]")
        if self.num_inference_steps < 1 or self.guidance_scale < 0.0 or self.feather_px < 0:
            raise ValueError("invalid diffusion step, guidance, or feather setting")
        if not (self.device == "cpu" or self.device.startswith("cuda")):
            raise ValueError(f"unsupported device string: {self.device}")
        _assert_allowed_model_id(self.model_id)
        _assert_allowed_model_id(self.fallback_model_id)
        repository = self.fallback_model_id if self.model_variant == "sdxl_inpaint_t4" else self.model_id
        if self.model_revision is not None and self.model_revision != APPROVED_MODELS[repository]:
            raise StageExecutionError("Diffusion revision has not been reviewed in the approved model registry")
        object.__setattr__(self, "model_revision", APPROVED_MODELS[repository])


class InpaintBackend(Protocol):
    def inpaint(
        self,
        image: RGBImage,
        mask: MaskArray,
        prompt: str,
        seed: int,
        image_reference: RGBImage | None = None,
    ) -> RGBImage:
        ...


def _resize_mask_to_hw(mask: MaskArray, target_hw: tuple[int, int]) -> MaskArray:
    h, w = target_hw
    if mask.shape[:2] == (h, w):
        return mask
    resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.where(resized > 0, 255, 0).astype(np.uint8)


def _supports_pipe_kwarg(pipe: Any, name: str, allow_var_kw: bool) -> bool:
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return allow_var_kw and any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _match_reference_lab_colors(
    generated_rgb: RGBImage,
    reference_rgb: RGBImage,
    mask: MaskArray,
    ref_mask: MaskArray | None = None,
) -> RGBImage:
    fg = mask > 0
    if not np.any(fg):
        return generated_rgb
    ref_resized = _resize_to_hw(reference_rgb, generated_rgb.shape[:2])
    gen_lab = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_resized, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_fg_mask = _resize_mask_to_hw(ref_mask, generated_rgb.shape[:2]) > 0 if ref_mask is not None else fg
    if not np.any(ref_fg_mask):
        ref_fg_mask = fg
    gen_fg = gen_lab[fg]
    ref_fg = ref_lab[ref_fg_mask]
    gen_mean = gen_fg.mean(axis=0)
    ref_mean = ref_fg.mean(axis=0)
    gen_std_raw = gen_fg.std(axis=0)
    ref_std = np.maximum(ref_fg.std(axis=0), 1e-3)
    scale = np.where(gen_std_raw > 1e-3, np.clip(ref_std / np.maximum(gen_std_raw, 1e-3), 0.25, 4.0), 1.0)
    gen_lab[fg] = np.clip((gen_fg - gen_mean) * scale + ref_mean, 0.0, 255.0)
    matched = cv2.cvtColor(np.rint(gen_lab).astype(np.uint8), cv2.COLOR_LAB2RGB)
    out = generated_rgb.copy()
    out[fg] = matched[fg]
    return out


class _DiffusersBackend:
    def __init__(
        self,
        pipe: Any,
        torch_mod: Any,
        config: DiffusionConfig,
        supports_ref: bool,
        exec_device: str,
        quantization_mode: str = "fp16_offload",
        torch_dtype_name: str = "float16",
    ) -> None:
        self._pipe = pipe
        self._torch = torch_mod
        self._config = config
        self._supports_ref = supports_ref
        self._exec_device = exec_device
        self.quantization_mode = quantization_mode
        self.torch_dtype_name = torch_dtype_name
        self.offload_text_encoder = bool(exec_device.startswith("cuda") and config.offload_text_encoder)
        self.resolved_model_id: str | None = None
        self.resolved_revision: str | None = None
        self.model_load_sec: float | None = None

    def inpaint(
        self,
        image: RGBImage,
        mask: MaskArray,
        prompt: str,
        seed: int,
        image_reference: RGBImage | None = None,
        negative_prompt: str = "",
    ) -> RGBImage:
        generator = self._torch.Generator(device=self._exec_device).manual_seed(int(seed))
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": Image.fromarray(image),
            "mask_image": Image.fromarray(mask),
            "num_inference_steps": self._config.num_inference_steps,
            "guidance_scale": self._config.guidance_scale,
            "strength": self._config.strength,
            "generator": generator,
        }
        if _supports_pipe_kwarg(self._pipe, "callback_on_step_end", allow_var_kw=False):
            def step_callback(pipe: Any, step: int, timestep: Any, tensors: dict[str, Any]) -> dict[str, Any]:
                generation_step(step + 1, int(getattr(pipe, "num_timesteps", self._config.num_inference_steps)))
                return tensors

            kwargs["callback_on_step_end"] = step_callback
            kwargs["callback_on_step_end_tensor_inputs"] = []
        # Flux2KleinInpaintPipeline.__call__ omits negative_prompt; only pass if explicitly accepted.
        if negative_prompt and _supports_pipe_kwarg(self._pipe, "negative_prompt", allow_var_kw=not self._supports_ref):
            kwargs["negative_prompt"] = negative_prompt
        if self._supports_ref and image_reference is not None:
            kwargs["image_reference"] = Image.fromarray(image_reference)
        if self._supports_ref and not _supports_pipe_kwarg(self._pipe, "mask_image", allow_var_kw=True):
            # Official Klein edits reference images; the lossless compositor enforces the mask afterward.
            kwargs.pop("mask_image")
            kwargs.pop("strength")
            kwargs.pop("image_reference", None)
            kwargs["height"], kwargs["width"] = image.shape[:2]
            kwargs["image"] = [Image.fromarray(image), Image.fromarray(mask).convert("RGB")]
            instruction = "Image 1 is the current view. Image 2 is its selection mask (white is editable). "
            if image_reference is not None:
                kwargs["image"].append(Image.fromarray(image_reference))
                instruction += "Image 3 is the approved edited reference: match its object appearance in image 1. "
            kwargs["prompt"] = instruction + "Preserve everything outside the selection. Requested edit: " + prompt
        result = self._pipe(**kwargs).images[0]
        return np.array(result.convert("RGB"), dtype=np.uint8, copy=True)


class _DiffusersFluxKleinBackend(_DiffusersBackend):
    def __init__(
        self,
        pipe: Any,
        torch_mod: Any,
        config: DiffusionConfig,
        exec_device: str,
        quantization_mode: str = "fp16_offload",
        torch_dtype_name: str = "float16",
    ) -> None:
        super().__init__(
            pipe,
            torch_mod,
            config,
            supports_ref=True,
            exec_device=exec_device,
            quantization_mode=quantization_mode,
            torch_dtype_name=torch_dtype_name,
        )


class _DiffusersSdxlBackend(_DiffusersBackend):
    def __init__(
        self,
        pipe: Any,
        torch_mod: Any,
        config: DiffusionConfig,
        exec_device: str,
        quantization_mode: str = "fp16_offload",
        torch_dtype_name: str = "float16",
    ) -> None:
        super().__init__(
            pipe,
            torch_mod,
            config,
            supports_ref=False,
            exec_device=exec_device,
            quantization_mode=quantization_mode,
            torch_dtype_name=torch_dtype_name,
        )
        self.last_reference_conditioning: str | None = None

    def inpaint(
        self,
        image: RGBImage,
        mask: MaskArray,
        prompt: str,
        seed: int,
        image_reference: RGBImage | None = None,
        negative_prompt: str = "",
        ref_mask: MaskArray | None = None,
    ) -> RGBImage:
        out = super().inpaint(
            image,
            mask,
            prompt,
            seed,
            image_reference=None,
            negative_prompt=negative_prompt,
        )
        if image_reference is not None:
            self.last_reference_conditioning = "lab_statistics_only"
            LOGGER.info("SDXL fallback applying reference_conditioning=%s", self.last_reference_conditioning)
            return _match_reference_lab_colors(out, image_reference, mask, ref_mask=ref_mask)
        self.last_reference_conditioning = None
        return out


def _assert_allowed_model_id(model_id: str) -> None:
    if model_id not in APPROVED_MODELS:
        raise StageExecutionError(f"Model '{model_id}' violates commercial license guardrails: repository is not approved")


def resolve_t4_dtype(compute_capability: tuple[int, int] | None) -> str:
    """Enforce float32 on CPU, float16 on pre-Ampere (Turing T4 sm_75), float16/bfloat16 on >= sm_80."""
    if compute_capability is None:
        return "float32"
    if compute_capability[0] < 8:
        return "float16"
    return "bfloat16"


def _validate_rgb_and_mask(image: RGBImage, binary_mask: MaskArray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise StageExecutionError("image must be HxWx3 uint8 RGB")
    if binary_mask.ndim != 2 or binary_mask.dtype != np.uint8:
        raise StageExecutionError("binary_mask must be a 2D uint8 array")
    if image.shape[:2] != binary_mask.shape[:2]:
        raise StageExecutionError(f"mask shape {binary_mask.shape} != image shape {image.shape[:2]}")
    if np.any((binary_mask != 0) & (binary_mask != 255)):
        raise StageExecutionError("binary_mask must contain strictly 0 or 255")


def pad_to_multiple(image: RGBImage, binary_mask: MaskArray, multiple: int = 16) -> tuple[RGBImage, MaskArray, tuple[int, int]]:
    _validate_rgb_and_mask(image, binary_mask)
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, binary_mask, (h, w)
    padded_img = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    padded_mask = np.pad(binary_mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return np.asarray(padded_img, dtype=np.uint8), np.asarray(padded_mask, dtype=np.uint8), (h, w)


def _pad_rgb_only(image: RGBImage, target_hw: tuple[int, int]) -> RGBImage:
    h, w = image.shape[:2]
    th, tw = target_hw
    if (h, w) == (th, tw):
        return image
    return np.asarray(np.pad(image, ((0, th - h), (0, tw - w), (0, 0)), mode="edge"), dtype=np.uint8)


def composite_strict_mask(orig_rgb: RGBImage, edited_rgb: RGBImage, binary_mask: MaskArray, feather_px: int = 3) -> RGBImage:
    """Composite edited pixels strictly inside the mask; outside pixels keep orig_rgb bitwise identical."""
    _validate_rgb_and_mask(orig_rgb, binary_mask)
    if edited_rgb.shape != orig_rgb.shape or edited_rgb.dtype != np.uint8:
        raise StageExecutionError("edited_rgb must match orig_rgb HxWx3 uint8")
    out = orig_rgb.copy()
    fg = binary_mask == 255
    if not np.any(fg):
        return out
    if feather_px <= 0:
        out[fg] = edited_rgb[fg]
        return out
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather_px + 1, 2 * feather_px + 1))
    inner = cv2.erode(binary_mask, kernel, iterations=1) == 255
    alpha = np.where(fg, 0.5, 0.0).astype(np.float32)
    alpha[inner] = 1.0
    blended = np.rint((1.0 - alpha[..., None]) * orig_rgb + alpha[..., None] * edited_rgb).clip(0, 255).astype(np.uint8)
    out[fg] = blended[fg]
    return out


composite_masked_region = composite_strict_mask


def assert_unmasked_unchanged(orig_rgb: RGBImage, composited_rgb: RGBImage, binary_mask: MaskArray) -> int:
    bg = binary_mask == 0
    if not np.any(bg):
        return 0
    max_diff = int(np.max(np.abs(composited_rgb[bg].astype(np.int32) - orig_rgb[bg].astype(np.int32))))
    if max_diff != 0:
        raise StageExecutionError(f"Unmasked background drift detected: max_abs_diff={max_diff}")
    return max_diff


def _load_rgb(path: Path) -> RGBImage:
    if not path.exists():
        raise StageExecutionError(f"Missing frame image: {path}")
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path, hw: tuple[int, int]) -> MaskArray:
    if not path.exists():
        return np.zeros(hw, dtype=np.uint8)
    with Image.open(path) as img:
        img.load()
        if img.mode != "L":
            raise StageExecutionError(f"Mask must be mode 'L', got '{img.mode}': {path}")
        arr = np.asarray(img, dtype=np.uint8)
    if arr.shape != hw or np.any((arr != 0) & (arr != 255)):
        raise StageExecutionError(f"Invalid mask shape or pixel values: {path}")
    return arr


def _pad_mask_only(mask: MaskArray, target_hw: tuple[int, int]) -> MaskArray:
    h, w = mask.shape[:2]
    th, tw = target_hw
    if (h, w) == (th, tw):
        return mask
    return np.asarray(np.pad(mask, ((0, th - h), (0, tw - w)), mode="constant", constant_values=0), dtype=np.uint8)


def _call_inpaint(
    backend: InpaintBackend,
    image: RGBImage,
    mask: MaskArray,
    prompt: str,
    seed: int,
    image_reference: RGBImage | None = None,
    negative_prompt: str = "",
    ref_mask: MaskArray | None = None,
) -> RGBImage:
    kwargs: dict[str, Any] = {"image_reference": image_reference}
    params = inspect.signature(backend.inpaint).parameters
    accepts_var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if negative_prompt and ("negative_prompt" in params or accepts_var):
        kwargs["negative_prompt"] = negative_prompt
    if ref_mask is not None and ("ref_mask" in params or accepts_var):
        kwargs["ref_mask"] = ref_mask
    return backend.inpaint(image, mask, prompt, seed=seed, **kwargs)


def _run_padded_inpaint(
    orig: RGBImage,
    mask: MaskArray,
    prompt: str,
    seed: int,
    backend: InpaintBackend,
    feather_px: int,
    image_reference: RGBImage | None = None,
    negative_prompt: str = "",
    ref_mask: MaskArray | None = None,
) -> RGBImage:
    padded_img, padded_mask, (h, w) = pad_to_multiple(orig, mask, multiple=16)
    padded_ref = _pad_rgb_only(image_reference, padded_img.shape[:2]) if image_reference is not None else None
    padded_ref_mask = _pad_mask_only(ref_mask, padded_img.shape[:2]) if ref_mask is not None else None
    raw_out = _call_inpaint(
        backend,
        padded_img,
        padded_mask,
        prompt,
        seed=seed,
        image_reference=padded_ref,
        negative_prompt=negative_prompt,
        ref_mask=padded_ref_mask,
    )[:h, :w]
    composited = composite_strict_mask(orig, raw_out, mask, feather_px=feather_px)
    assert_unmasked_unchanged(orig, composited, mask)
    return composited


def _cuda_ordinal_if_active(device: str) -> tuple[Any, int] | None:
    if not device.startswith("cuda"):
        return None
    try:
        torch_mod = importlib.import_module("torch")
    except ImportError:
        return None
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)) or not cuda.is_available():
        return None
    is_init = getattr(cuda, "is_initialized", None)
    if callable(is_init) and not is_init():
        return None
    ordinal = int(device.split(":")[1]) if ":" in device else 0
    if hasattr(cuda, "device_count") and ordinal >= cuda.device_count():
        ordinal = 0
    return torch_mod, ordinal


def _reset_vram_peak(device: str) -> None:
    info = _cuda_ordinal_if_active(device)
    if info is None:
        return
    torch_mod, ordinal = info
    if hasattr(torch_mod.cuda, "reset_peak_memory_stats"):
        torch_mod.cuda.reset_peak_memory_stats(ordinal)


def _read_peak_vram_mb(device: str) -> float:
    info = _cuda_ordinal_if_active(device)
    if info is None:
        return 0.0
    torch_mod, ordinal = info
    if not hasattr(torch_mod.cuda, "max_memory_allocated"):
        return 0.0
    return round(float(torch_mod.cuda.max_memory_allocated(ordinal)) / (1024.0 * 1024.0), 2)


def generate_first_frame_seeds(
    ref_image_path: Path,
    ref_mask_path: Path,
    prompt: str,
    seeds: list[int],
    output_dir: Path,
    backend: InpaintBackend,
    config: DiffusionConfig | None = None,
    negative_prompt: str | None = None,
    checkpoint_root: Path | None = None,
    checkpoint_context: dict[str, object] | None = None,
    checkpoint_callback: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    cfg = config or DiffusionConfig()
    neg = cfg.negative_prompt if negative_prompt is None else negative_prompt
    orig = _load_rgb(ref_image_path)
    mask = _load_mask(ref_mask_path, orig.shape[:2])
    output_dir.mkdir(parents=True, exist_ok=True)
    _reset_vram_peak(cfg.device)
    t0 = time.perf_counter()
    previews: list[dict[str, Any]] = []
    settings = {"mode": "seed_preview", "config": asdict(cfg), "context": checkpoint_context,
                "prompt": prompt, "negative_prompt": neg, "seeds": seeds,
                "model": getattr(backend, "resolved_model_id", None), "revision": getattr(backend, "resolved_revision", None),
                "quantization": getattr(backend, "quantization_mode", cfg.quantization),
                "dtype": getattr(backend, "torch_dtype_name", "unknown"),
                "pipeline": type(getattr(backend, "_pipe", backend)).__name__}
    identity = diffusion_identity({0: ref_image_path}, {0: ref_mask_path}, ref_image_path, settings)
    checkpoints = DiffusionCheckpoints(checkpoint_root or output_dir / "checkpoints", identity)
    restored = {seed for seed in seeds if checkpoints.restore(seed, output_dir / f"seed_{seed}.png")}
    tracker = GenerationProgress(len(seeds), len(seeds), len(restored))
    for seed in seeds:
        out_path = output_dir / f"seed_{int(seed)}.png"
        if seed not in restored:
            tracker.begin([int(seed)])
            with generation_scope(tracker):
                comp = _run_padded_inpaint(orig, mask, prompt, int(seed), backend, cfg.feather_px, negative_prompt=neg)
            Image.fromarray(comp).save(out_path, format="PNG")
            checkpoints.commit(seed, out_path)
            if checkpoint_callback:
                checkpoint_callback()
            tracker.finish()
        else:
            assert_unmasked_unchanged(orig, _load_rgb(out_path), mask)
        previews.append({"seed_id": f"seed_{int(seed)}", "seed": int(seed), "path": str(out_path)})
    elapsed = max(time.perf_counter() - t0, 1e-6)
    generated_seeds = len(seeds) - len(restored)
    report_progress("06_diffusion", 95, f"Generated {len(seeds)}/{len(seeds)} seed previews",
                    phase="inference_complete", operations_completed=len(seeds), operations_total=len(seeds),
                    operations_resumed=len(restored), frames_to_inpaint=len(seeds))
    sec_per_frame = round(elapsed / generated_seeds, 4) if generated_seeds else None
    peak_vram_mb = _read_peak_vram_mb(cfg.device)
    for item in previews:
        item["sec_per_frame"] = sec_per_frame
        item["peak_vram_mb"] = peak_vram_mb
    (output_dir / "seed_metrics.json").write_text(
        json.dumps(
            {
                "num_seeds": len(seeds),
                "seeds_resumed": len(restored),
                "seeds_generated_this_attempt": generated_seeds,
                "elapsed_sec": round(elapsed, 4),
                "sec_per_frame": sec_per_frame,
                "peak_vram_mb": peak_vram_mb,
                "quantization_mode": getattr(backend, "quantization_mode", cfg.quantization),
                "dtype": getattr(backend, "torch_dtype_name", "unknown"),
                "model_id": getattr(backend, "resolved_model_id", None),
                "model_revision": getattr(backend, "resolved_revision", None),
                "model_load_sec": getattr(backend, "model_load_sec", None),
                "effective_config": asdict(getattr(backend, "_config", cfg)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return previews


def _tile_2x2(frames: list[RGBImage | MaskArray]) -> RGBImage | MaskArray:
    h, w = frames[0].shape[:2]
    shape = (2 * h, 2 * w, 3) if frames[0].ndim == 3 else (2 * h, 2 * w)
    canvas = np.zeros(shape, dtype=np.uint8)
    for idx, tile in enumerate(frames):
        r, c = divmod(idx, 2)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    return canvas


def _save_edited_frame(edited_dir: Path, idx: int, comp: RGBImage, orig: RGBImage, mask: MaskArray) -> int:
    Image.fromarray(comp).save(edited_dir / f"{idx:06d}.png", format="PNG")
    return assert_unmasked_unchanged(orig, comp, mask)


def _resize_to_hw(image: RGBImage, target_hw: tuple[int, int]) -> RGBImage:
    h, w = target_hw
    if image.shape[:2] == (h, w):
        return image
    return np.asarray(cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR), dtype=np.uint8)


def _mask_coverage(mask: MaskArray) -> float:
    return float(np.count_nonzero(mask == 255)) / float(mask.size) if mask.size > 0 else 0.0


def _propagate_sequential(
    indices: list[int],
    frame_paths: dict[int, Path],
    mask_paths: dict[int, Path],
    ref_edit: RGBImage,
    prompt: str,
    seed: int,
    edited_dir: Path,
    backend: InpaintBackend,
    feather_px: int,
    frame_records: dict[int, dict[str, Any]],
    negative_prompt: str = "",
    ref_mask: MaskArray | None = None,
) -> tuple[int, int]:
    edited_count, max_diff = 0, 0
    for idx in indices:
        orig = _load_rgb(frame_paths[idx])
        mask = _load_mask(mask_paths.get(idx, edited_dir / "_missing.png"), orig.shape[:2])
        cov = _mask_coverage(mask)
        if cov > 0.0:
            comp = _run_padded_inpaint(
                orig,
                mask,
                prompt,
                seed,
                backend,
                feather_px,
                image_reference=ref_edit,
                negative_prompt=negative_prompt,
                ref_mask=ref_mask,
            )
            edited_count += 1
            frame_records[idx] = {"frame_idx": idx, "skipped": False, "method": "diffusion_sequential", "mask_coverage": cov}
        else:
            comp = orig
            frame_records[idx] = {"frame_idx": idx, "skipped": True, "method": "unchanged_original", "mask_coverage": 0.0}
        max_diff = max(max_diff, _save_edited_frame(edited_dir, idx, comp, orig, mask))
    return edited_count, max_diff


def _propagate_grid_batch(
    batch: list[int],
    frame_paths: dict[int, Path],
    mask_paths: dict[int, Path],
    ref_edit: RGBImage,
    prompt: str,
    seed: int,
    edited_dir: Path,
    backend: InpaintBackend,
    feather_px: int,
    frame_records: dict[int, dict[str, Any]],
    negative_prompt: str = "",
    ref_mask: MaskArray | None = None,
) -> tuple[int, int]:
    origs = [_load_rgb(frame_paths[idx]) for idx in batch]
    h, w = origs[0].shape[:2]
    masks = [_load_mask(mask_paths.get(idx, edited_dir / "_missing.png"), (h, w)) for idx in batch]
    if len(batch) < 4 or (2 * h) * (2 * w) > MAX_GRID_PIXELS or any(o.shape[:2] != (h, w) for o in origs):
        return _propagate_sequential(
            batch,
            frame_paths,
            mask_paths,
            ref_edit,
            prompt,
            seed,
            edited_dir,
            backend,
            feather_px,
            frame_records,
            negative_prompt=negative_prompt,
            ref_mask=ref_mask,
        )
    grid_img = _tile_2x2(origs)
    grid_mask = _tile_2x2(masks)
    ref_tile = _resize_to_hw(ref_edit, (h, w))
    grid_ref = _tile_2x2([ref_tile, ref_tile, ref_tile, ref_tile])
    grid_ref_mask: MaskArray | None = None
    if ref_mask is not None:
        rm_tile = _resize_mask_to_hw(ref_mask, (h, w))
        grid_ref_mask = _tile_2x2([rm_tile, rm_tile, rm_tile, rm_tile])
    grid_out = _run_padded_inpaint(
        grid_img,
        grid_mask,
        prompt,
        seed,
        backend,
        feather_px,
        image_reference=grid_ref,
        negative_prompt=negative_prompt,
        ref_mask=grid_ref_mask,
    )
    edited_count, max_diff = 0, 0
    for slot, idx in enumerate(batch):
        r, c = divmod(slot, 2)
        tile_out = grid_out[r * h : (r + 1) * h, c * w : (c + 1) * w]
        comp = composite_strict_mask(origs[slot], tile_out, masks[slot], feather_px=feather_px)
        cov = _mask_coverage(masks[slot])
        edited_count += 1
        frame_records[idx] = {"frame_idx": idx, "skipped": False, "method": "diffusion_grid_2x2", "mask_coverage": cov}
        max_diff = max(max_diff, _save_edited_frame(edited_dir, idx, comp, origs[slot], masks[slot]))
    return edited_count, max_diff


def propagate_masked_edits(
    frame_paths: dict[int, Path],
    mask_paths: dict[int, Path],
    ref_frame_idx: int,
    ref_edited_path: Path,
    prompt: str,
    seed: int,
    output_dir: Path,
    backend: InpaintBackend,
    skip_frames: set[int] | None = None,
    config: DiffusionConfig | None = None,
    negative_prompt: str | None = None,
    checkpoint_root: Path | None = None,
    checkpoint_context: dict[str, object] | None = None,
    checkpoint_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Propagate an approved reference-frame edit across all active views with strict mask compositing."""
    cfg = config or DiffusionConfig()
    neg = cfg.negative_prompt if negative_prompt is None else negative_prompt
    ref_edit = _load_rgb(ref_edited_path)
    edited_dir = output_dir / "edited_frames"
    edited_dir.mkdir(parents=True, exist_ok=True)
    _reset_vram_peak(cfg.device)
    t0 = time.perf_counter()
    ref_mask_path = mask_paths.get(ref_frame_idx)
    ref_mask = _load_mask(ref_mask_path, ref_edit.shape[:2]) if ref_mask_path is not None and ref_mask_path.exists() else None
    skipped = skip_frames or set()
    frame_records: dict[int, dict[str, Any]] = {}
    inpaint_indices: list[int] = []
    max_diff = 0
    edited_count = 0

    for idx in sorted(frame_paths):
        orig = _load_rgb(frame_paths[idx])
        mask = _load_mask(mask_paths.get(idx, edited_dir / "_missing.png"), orig.shape[:2])
        cov = _mask_coverage(mask)
        if idx in skipped or (cov == 0.0 and idx != ref_frame_idx):
            max_diff = max(max_diff, _save_edited_frame(edited_dir, idx, orig, orig, mask))
            frame_records[idx] = {"frame_idx": idx, "skipped": True, "method": "unchanged_original", "mask_coverage": cov}
        elif idx == ref_frame_idx:
            # Copy already-feathered seed preview inside mask so boundaries are not feathered twice.
            ref_resized = _resize_to_hw(ref_edit, orig.shape[:2])
            comp = np.where(mask[..., None] > 0, ref_resized, orig).astype(np.uint8)
            max_diff = max(max_diff, _save_edited_frame(edited_dir, idx, comp, orig, mask))
            edited_count += 1
            frame_records[idx] = {"frame_idx": idx, "skipped": False, "method": "reference_seed", "mask_coverage": cov}
        else:
            inpaint_indices.append(idx)

    settings = {"config": asdict(cfg), "context": checkpoint_context, "prompt": prompt, "negative_prompt": neg, "seed": seed,
                "ref_frame_idx": ref_frame_idx, "skip_frames": sorted(skipped),
                "resolved_model": getattr(backend, "resolved_model_id", None),
                "resolved_revision": getattr(backend, "resolved_revision", None),
                "pipeline_class": type(getattr(backend, "_pipe", backend)).__name__,
                "dtype": getattr(backend, "torch_dtype_name", "unknown"),
                "quantization_mode": getattr(backend, "quantization_mode", cfg.quantization)}
    identity = diffusion_identity(frame_paths, mask_paths, ref_edited_path, settings)
    if inpaint_indices and isinstance(backend, _DiffusersSdxlBackend):
        raise StageExecutionError("SDXL lab color matching is not reference-image conditioning. Use FLUX Klein for multi-view generative edits.")
    checkpoints = DiffusionCheckpoints(checkpoint_root or output_dir / "checkpoints", identity)
    groups = _inference_groups(inpaint_indices, frame_paths, cfg.grid_2x2)
    pending: list[list[int]] = []
    for group in groups:
        restored = [checkpoints.restore(idx, edited_dir / f"{idx:06d}.png") for idx in group]
        if all(restored):
            for idx in group:
                orig = _load_rgb(frame_paths[idx])
                mask = _load_mask(mask_paths[idx], orig.shape[:2])
                assert_unmasked_unchanged(orig, _load_rgb(edited_dir / f"{idx:06d}.png"), mask)
                frame_records[idx] = {"frame_idx": idx, "skipped": False, "method": "checkpoint_resume",
                                      "mask_coverage": _mask_coverage(mask)}
        else:
            pending.append(group)
    resumed = len(groups) - len(pending)
    tracker = GenerationProgress(len(groups), len(inpaint_indices), resumed)
    with generation_scope(tracker):
        for group in pending:
            tracker.begin(group)
            _, diff = _propagate_grid_batch(
                group, frame_paths, mask_paths, ref_edit, prompt, seed, edited_dir, backend,
                cfg.feather_px, frame_records, negative_prompt=neg, ref_mask=ref_mask,
            )
            max_diff = max(max_diff, diff)
            for idx in group:
                checkpoints.commit(idx, edited_dir / f"{idx:06d}.png")
            if checkpoint_callback:
                checkpoint_callback()
            tracker.finish()
    inpainted_count = len(inpaint_indices)

    edited_count += inpainted_count
    report_progress("06_diffusion", 95, f"Prepared {len(frame_records)}/{len(frame_paths)} frames; inference complete",
                    phase="inference_complete", operations_completed=len(groups), operations_total=len(groups),
                    operations_resumed=resumed, frames_to_inpaint=len(inpaint_indices))
    skipped_count = sum(1 for rec in frame_records.values() if rec["skipped"])
    elapsed = max(time.perf_counter() - t0, 1e-6)
    generated_count = sum(len(group) for group in pending)
    sec_per_frame = round(elapsed / generated_count, 4) if generated_count else None
    peak_vram_mb = _read_peak_vram_mb(cfg.device)
    metrics = {
        "ref_frame_idx": ref_frame_idx,
        "frames_edited": edited_count,
        "frames_inpainted": inpainted_count,
        "frames_generated_this_attempt": generated_count,
        "frames_resumed": sum(len(group) for group in groups) - sum(len(group) for group in pending),
        "inference_operations": len(groups), "inference_operations_resumed": resumed,
        "checkpoint_identity": identity,
        "frames_skipped": skipped_count,
        "grid_2x2": cfg.grid_2x2,
        "max_unmasked_diff": max_diff,
        "seed": seed,
        "negative_prompt": neg,
        "model_id": getattr(backend, "resolved_model_id", None),
        "configured_model_id": cfg.model_id,
        "model_revision": getattr(backend, "resolved_revision", None),
        "pipeline_class": type(getattr(backend, "_pipe", backend)).__name__,
        "model_load_sec": getattr(backend, "model_load_sec", None),
        "effective_config": asdict(getattr(backend, "_config", cfg)),
        "reference_conditioning": getattr(backend, "last_reference_conditioning", "multi_image_reference"),
        "model_variant": cfg.model_variant,
        "quantization": cfg.quantization,
        "quantization_mode": getattr(backend, "quantization_mode", cfg.quantization),
        "dtype": getattr(backend, "torch_dtype_name", "unknown"),
        "elapsed_sec": round(elapsed, 4),
        "sec_per_frame": sec_per_frame,
        "sec_per_generated_frame": sec_per_frame,
        "sec_per_input_frame": round(elapsed / max(len(frame_paths), 1), 4),
        "peak_vram_mb": peak_vram_mb,
        "frames": [frame_records[idx] for idx in sorted(frame_records)],
    }
    (output_dir / "diffusion_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _inference_groups(indices: list[int], frames: dict[int, Path], use_grid: bool) -> list[list[int]]:
    if not use_grid:
        return [[idx] for idx in indices]
    groups: list[list[int]] = []
    for start in range(0, len(indices), 4):
        batch = indices[start:start + 4]
        sizes = []
        for idx in batch:
            with Image.open(frames[idx]) as image:
                sizes.append(image.size)
        w, h = sizes[0]
        if len(batch) == 4 and 4 * w * h <= MAX_GRID_PIXELS and len(set(sizes)) == 1:
            groups.append(batch)
        else:
            groups.extend([idx] for idx in batch)
    return groups


def _resolve_gpu_ordinal(device: str, torch_mod: Any) -> tuple[bool, int]:
    if not device.startswith("cuda") or not torch_mod.cuda.is_available():
        return False, 0
    ordinal = int(device.split(":")[1]) if ":" in device else 0
    return (True, ordinal if ordinal < torch_mod.cuda.device_count() else 0)


def _has_bitsandbytes() -> bool:
    if "bitsandbytes" in sys.modules:
        return sys.modules["bitsandbytes"] is not None
    try:
        return importlib.util.find_spec("bitsandbytes") is not None
    except (ImportError, ValueError):
        return False


def _build_quantization_kwargs(
    cfg: DiffusionConfig,
    torch_mod: Any,
    diffusers_mod: Any,
    is_cuda: bool,
) -> tuple[dict[str, Any], str]:
    if not is_cuda:
        return {}, "fp32_cpu"
    if cfg.quantization == "none":
        return {}, "fp16_offload"
    if cfg.quantization == "nf4":
        pipe_q_cls = getattr(diffusers_mod, "PipelineQuantizationConfig", None)
        if callable(pipe_q_cls) and _has_bitsandbytes():
            q_cfg = pipe_q_cls(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch_mod.float16,
                },
            )
            return {"quantization_config": q_cfg}, "nf4"
        LOGGER.info("NF4 requires diffusers.PipelineQuantizationConfig + bitsandbytes; probing optimum.quanto INT8 fallback")
    try:
        quanto_mod = importlib.import_module("optimum.quanto")
    except (ImportError, OSError, RuntimeError):
        return {}, "fp16_offload"
    if hasattr(quanto_mod, "quantize") and hasattr(quanto_mod, "freeze") and hasattr(quanto_mod, "qint8"):
        return {}, "int8_quanto"
    return {}, "fp16_offload"


def _apply_quanto_int8(pipe: Any) -> None:
    quanto_mod = importlib.import_module("optimum.quanto")
    for attr in ("transformer", "unet"):
        submod = getattr(pipe, attr, None)
        if submod is not None:
            quanto_mod.quantize(submod, weights=quanto_mod.qint8)
            quanto_mod.freeze(submod)


def _from_pretrained_with_quant(
    cls: Any,
    repo_id: str,
    dtype: Any,
    hf_token: str | None,
    quant_kwargs: dict[str, Any],
    quant_mode: str = "fp16_offload",
    revision: str | None = None,
) -> tuple[Any, str]:
    parameters = inspect.signature(cls.from_pretrained).parameters
    dtype_key = "torch_dtype" if "torch_dtype" in parameters and "dtype" not in parameters else "dtype"
    kwargs: dict[str, Any] = {dtype_key: dtype, "token": hf_token}
    revision = revision or APPROVED_MODELS[repo_id]
    if revision:
        kwargs["revision"] = revision
    applied_quant = False
    if quant_kwargs:
        try:
            params = inspect.signature(cls.from_pretrained).parameters
            accepts_var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            for k, v in quant_kwargs.items():
                if k in params or accepts_var:
                    kwargs[k] = v
                    applied_quant = True
        except (TypeError, ValueError):
            kwargs.update(quant_kwargs)
            applied_quant = True
    try:
        return cls.from_pretrained(repo_id, **kwargs), quant_mode
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        if not applied_quant:
            raise
        LOGGER.warning("Quantized from_pretrained(%s) failed (%s); retrying fp16_offload", repo_id, exc)
        fallback_kwargs = {key: value for key, value in kwargs.items() if key not in quant_kwargs}
        return cls.from_pretrained(repo_id, **fallback_kwargs), "fp16_offload"


def load_diffusion_backend(config: DiffusionConfig | None = None) -> InpaintBackend:
    """Load T4-safe float16 pipeline with NF4/INT8 quantization and CPU text-encoder offload."""
    cfg = config or DiffusionConfig()
    started = time.perf_counter()
    report_progress("06_diffusion", 0, f"Loading diffusion model {cfg.model_id}", phase="model_load")
    try:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
    except ImportError as exc:
        raise StageExecutionError("s06_diffusion requires torch and diffusers when loading live weights") from exc
    is_cuda, gpu_id = _resolve_gpu_ordinal(cfg.device, torch)
    cap = torch.cuda.get_device_capability(gpu_id) if is_cuda else None
    dtype_name = resolve_t4_dtype(cap)
    dtype = getattr(torch, dtype_name)
    flux_cls = getattr(diffusers, "Flux2KleinPipeline", None) or getattr(diffusers, "Flux2KleinInpaintPipeline", None) or getattr(diffusers, "FluxKleinInpaintPipeline", None)
    sdxl_cls = getattr(diffusers, "StableDiffusionXLInpaintPipeline", None)
    hf_token = os.environ.get("HF_TOKEN") or None
    exec_device = f"cuda:{gpu_id}" if is_cuda else "cpu"
    quant_kwargs, quant_mode = _build_quantization_kwargs(cfg, torch, diffusers, is_cuda)

    backend: _DiffusersBackend
    resolved_repo = cfg.model_id
    if cfg.model_variant == "sdxl_inpaint_t4":
        if sdxl_cls is None:
            raise StageExecutionError("diffusers must provide StableDiffusionXLInpaintPipeline for sdxl_inpaint_t4")
        sdxl_repo = cfg.fallback_model_id
        resolved_repo = sdxl_repo
        pipe, quant_mode = _from_pretrained_with_quant(sdxl_cls, sdxl_repo, dtype, hf_token, quant_kwargs, quant_mode)
        backend = _DiffusersSdxlBackend(pipe, torch, cfg, exec_device, quant_mode, dtype_name)
    elif flux_cls is not None:
        try:
            pipe, quant_mode = _from_pretrained_with_quant(
                flux_cls, cfg.model_id, dtype, hf_token, quant_kwargs, quant_mode, revision=cfg.model_revision
            )
            backend = _DiffusersFluxKleinBackend(pipe, torch, cfg, exec_device, quant_mode, dtype_name)
        except (OSError, ValueError, RuntimeError) as exc:
            if not cfg.allow_model_fallback or sdxl_cls is None:
                raise
            LOGGER.warning("Failed loading %s (%s); falling back to %s", cfg.model_id, exc, cfg.fallback_model_id)
            pipe, quant_mode = _from_pretrained_with_quant(
                sdxl_cls, cfg.fallback_model_id, dtype, hf_token, quant_kwargs, quant_mode
            )
            backend = _DiffusersSdxlBackend(pipe, torch, cfg, exec_device, quant_mode, dtype_name)
            resolved_repo = cfg.fallback_model_id
    elif sdxl_cls is not None and cfg.allow_model_fallback:
        pipe, quant_mode = _from_pretrained_with_quant(
            sdxl_cls, cfg.fallback_model_id, dtype, hf_token, quant_kwargs, quant_mode
        )
        backend = _DiffusersSdxlBackend(pipe, torch, cfg, exec_device, quant_mode, dtype_name)
        resolved_repo = cfg.fallback_model_id
    else:
        raise StageExecutionError("diffusers must provide Flux2KleinInpaintPipeline or StableDiffusionXLInpaintPipeline")

    if quant_mode == "int8_quanto":
        _apply_quanto_int8(pipe)

    if is_cuda:
        if cfg.use_sequential_offload:
            pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
        elif cfg.offload_text_encoder:
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            pipe.to(exec_device)
    else:
        pipe.to("cpu")
    for fn_name in ("enable_vae_tiling", "enable_vae_slicing"):
        if hasattr(pipe, fn_name):
            getattr(pipe, fn_name)()
    backend.resolved_model_id = resolved_repo
    backend.resolved_revision = APPROVED_MODELS[resolved_repo]
    if resolved_repo == cfg.fallback_model_id and cfg.model_variant != "sdxl_inpaint_t4":
        backend._config = replace(cfg, model_variant="sdxl_inpaint_t4", model_revision=None)
    backend.model_load_sec = round(time.perf_counter() - started, 4)
    report_progress("06_diffusion", 10, f"Model ready: {resolved_repo}", phase="model_ready",
                    model_load_sec=backend.model_load_sec)
    return backend
