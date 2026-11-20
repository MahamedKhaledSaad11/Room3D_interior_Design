# Current implementation note — 20 November 2026

The historical report below contains superseded implementation details (including the 8-step default, denylist description and SDXL batch color matching). Use the [current W_7/W_8 verification and recovery report](2026-11-20_week7_week8_verification.md) for the approved exact repository/revision registry, distilled 4-step versus base 50-step presets, real reference-image conditioning, checkpoints and measured verification limits. Historical synthetic timings are not real diffusion throughput or evidence of the required multi-view visual quality.

# Week 2 Diffusion & T4 (`sm_75`) Hardware Report

## 1. Live Kaggle `2× Tesla T4` Hardware Baseline (`S0.2b-v1`, Verified)
- **Kernel:** `mohamedelbasyouni55/room3d-t4-preflight` (version `1`, `KernelWorkerStatus.COMPLETE`)
- **OS / Python:** `Linux-6.18.55+-x86_64-with-glibc2.39`, Kaggle container Python `3.13`
- **GPUs:** `2 × Tesla T4` (`15360 MiB` each), CUDA compute capability `(7, 5)` (`sm_75`, Turing)
- **Free Working Disk:** `19.5 GB` (`/kaggle/working`)
- **Free VRAM (idle GPU 0):** `14.41 GB`
- **CUDA `float16` Matmul + `torch.isfinite` Guard:** `true` (bf16 NOT required)
- **Pre-installed Kaggle Image Packages:**
  - `diffusers` (`Flux2KleinInpaintPipeline` available: `true`, `StableDiffusionXLInpaintPipeline` available: `true`)
  - `accelerate` (`available: true`)
  - Missing in base image (requires worker/notebook bootstrap): `pycolmap-cuda12`, `av`, `gsplat`, `spz`, `ppisp`, `bitsandbytes`

## 2. Live Bootstrap & API Verification (`S0.2b-v2` -> `S0.2b-v3`, Verified)
- **`S0.2b-v2` Trap Discovered (`bitsandbytes`):** Installing unpinned `bitsandbytes` on Kaggle `torch 2.11.0+cu130` crashes `diffusers.loaders.lora_pipeline` at import time (`libbitsandbytes_cuda130.so` missing).
- **`S0.2b-v3` (`KernelWorkerStatus.COMPLETE`):** Removing `bitsandbytes` restored clean imports for `Flux2KleinInpaintPipeline` (`float16`, `cpu_offload`) and `StableDiffusionXLInpaintPipeline` (`float16`).
- **Verified `Flux2KleinInpaintPipeline.__call__` parameters:** `attention_kwargs`, `callback_on_step_end`, `callback_on_step_end_tensor_inputs`, `generator`, `guidance_scale`, `height`, `image`, `image_reference`, `latents`, `mask_image`, `max_sequence_length`, `negative_prompt_embeds`, `num_images_per_prompt`, `num_inference_steps`, `output_type`, `padding_mask_crop`, `prompt`, `prompt_embeds`, `return_dict`, `sigmas`, `strength`, `text_encoder_out_layers`, `width`.
- **Verified `pycolmap 4.2.1` signature:** `ALIKED_N16ROT` + `ALIKED_LIGHTGLUE` (`onnx_support: true`, `global_mapping: true`). Requires 3 positional args `pycolmap.global_mapping(database_path, image_path, output_path)` with `database.db` stored **outside** `image_path`.
- **Verified packaging trap:** PyPI `spz` has a circular import error (`BoundingBox`) on Python 3.13; use `scaniverse/spz` or pure-Python protobuf gzip writer + `@playcanvas/splat-transform` post-processing.

## 3. Implemented T4-Safe Diffusion & Compositor Safeguards (`s06_diffusion.py`, S2.3)

| Safeguard | Implementation (`worker/room3d_worker/pipeline/s06_diffusion.py`) | Rationale / Verified Constraint |
| :--- | :--- | :--- |
| **Commercial License Gate** | `_assert_allowed_model_id` (`FORBIDDEN_MODEL_PATTERNS = ("9b", "qwen")`) + `model_variant` dispatch | Permits `flux2_klein_4b` (`_DiffusersFluxKleinBackend`, `Apache-2.0`) and `sdxl_inpaint_t4` (`_DiffusersSdxlBackend`, `OpenRAIL++`); honors explicit `config.model_variant` selection, defaults `device="cuda:0"`, and blocks `FLUX.2 [klein] 9B` and `Qwen-Image-2.1`. |
| **Reference Seed & Empty-Mask Preservation** | `propagate_masked_edits` + `composite_masked_region` | Preserves the user-approved `ref_edited_path` via strict mask compositing on `ref_frame_idx` (`method="reference_seed"`) without re-running diffusion; copies original `orig_rgb` for empty-mask (`mask_coverage == 0.0`) or skipped frames (`skipped=True, method="unchanged_original"`). |
| **VRAM & Grid Pixel Guard** | `MAX_GRID_PIXELS = 1048576` (`1024 * 1024`), `enable_vae_tiling`, `enable_vae_slicing`, `resolve_t4_dtype` | Multi-frame 2x2 grid (`<= 4` active non-ref views via `_tile_2x2` / `_propagate_grid_batch`) automatically falls back to sequential per-frame generation (`_propagate_sequential`) conditioned on `image_reference` whenever total grid pixels exceed `1024^2` or batch size `< 4`. Defaults: `num_inference_steps=8`, `guidance_scale=3.5`, `strength=0.85`, `feather_px=3`. |
| **Bit-Exact Background Preservation** | `composite_strict_mask` (`composite_masked_region`) + `assert_unmasked_unchanged` | Enforces `max_abs_diff == 0` outside the binary mask (`worker/tests/test_diffusion.py`). |
| **Mod-16 & T4 Dtype Guard** | `pad_to_multiple(..., 16)` + `resolve_t4_dtype` | Pads non-mod-16 dimensions safely before inpainting and selects `torch.float16` on `sm_75` GPUs (CPU uses `float32`). |

---

## 4. Local Benchmark Measurements & Reference `2 × T4` Methodology

### Local Hardware Benchmark (`python -m room3d_worker.benchmarks`)
Measured on `Windows-11-10.0.26300-SP0`, Python `3.12.15`, `torch 2.14.1+cu126`, `NVIDIA GeForce RTX 2080` (`sm_75`, `8191.7 MB` VRAM) with synthetic inpaint backend (`8` frames, `160x128`):
- **First-Frame Seed Preview (2 seeds):** `10.318 ms`
- **2x2 Grid Compositing Propagation (8 frames):** `48.895 ms` (`6.112 ms/frame`)
- **Sequential Compositing Propagation (8 frames):** `32.433 ms` (`4.054 ms/frame`)
- **Invariants:** `max_unmasked_diff = 0`, `ref_method = "reference_seed"`, `empty_mask_skipped = true`, `t4_sm75_dtype = "float16"`

### Reference Kaggle `2 × T4` Execution Methodology
- `worker_runner.py` assigns `GPU 0` (`cuda:0`) exclusively to diffusion stages (`diffusion_preview`, `diffusion_apply`) using `torch.float16` + `enable_model_cpu_offload` (or `enable_sequential_cpu_offload`) + `enable_vae_tiling` + `enable_vae_slicing`, keeping peak VRAM within the `14.41 GB` free T4 envelope verified in `S0.2b`.

---

## 5. Week 2 Localized 3DGS Rebuild (`s07_rebuild.py`, S2.4)

- **`masked_finetune`:** Computes `compute_roi_gaussian_mask` over tracked masks (`roi_pad_px=8`), optimizes ONLY `roi_mask` parameters while keeping background Gaussians frozen in differentiable `gsplat.rasterization`, and enforces `assert_frozen_gaussians_unchanged` post-rebuild / pre-export so any background mutation fails loudly (`worker/tests/test_rebuild.py`).
- **`from_scratch`:** Re-initializes from `sparse/0/points3D.bin` for global re-optimization ablation.
- **Test Status:** `worker/tests/test_diffusion.py` (`6/6`), `worker/tests/test_track.py` (`8/8`), and `worker/tests/test_kaggle_preflight.py` (`3/3`) pass locally (`torch_gpu`).
