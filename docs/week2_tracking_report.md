# Week 2 Interactive Segmentation & 5-Tier Tracking Report (`s05_track.py`)

## 1. Warm Interactive SAM 2.1 Architecture

Week 2 interactive clicks (`segment_init`, `segment_refine`) cannot afford cold model weight initialization per click. `worker/room3d_worker/pipeline/s05_track.py` implements `WarmSamSession`:

- **Bounded LRU Frame Cache**: Calls `_to_cpu_if_tensor` recursively across `dict`, `list`, and `tuple` structures on cache insertion so up to `max_cache_items` (default `8`) encoded image states sit in CPU RAM inside an `OrderedDict` (evicting the oldest item via `popitem(last=False)`), while `_Sam2Interactive.predict()` restores cached `_features` back to `_predictor_device(self._pred)` via `_to_device_if_tensor` so warm-cache hits on `cuda:1` never hit CPU/CUDA device mismatches.
- **State Snapshot & Restoration**: `_Sam2Interactive` (`_Sam2Image`) captures and restores `_features`, `_orig_hw`, and `_is_image_set` when switching between cached reference frames (`set_image` runs only on cache miss).
- **Binary Mask Normalization**: `normalize_binary_mask` converts boolean arrays, raw float logits (`threshold=0.0`), `0..1` probabilities, or `0..255` arrays into canonical 1-channel `uint8` (`{0, 255}`) masks matching `(height, width)`, saved via `save_binary_mask_png`.

---

## 2. 5-Tier Propagation & Recovery Pipeline (`run_tracking_pipeline`)

To prevent object drift, ID switches on occlusion, and dropped masks across wide camera sweeps, `run_tracking_pipeline` orchestrates five recovery tiers:

| Tier | Helper / Function | Condition / Threshold | Action Taken |
| :--- | :--- | :--- | :--- |
| **Tier 1** | `track_vos` (`_Sam2Video`) | Default pass on ordered frames | Stages ordered RGB JPEGs inside a `tempfile.TemporaryDirectory`, converts CUDA logits safely via `.detach().cpu().numpy()`, and uses `inspect.signature` to run bidirectional propagation (`start_frame_idx=ref_pos, reverse=False` and `reverse=True` when `ref_pos > 0`) with fallback for single-pass mocks. |
| **Tier 2** | `filter_neighbor_iou` & `_neighbor_drift_bad` | `min_iou=0.25`, `max_area_ratio=3.5` | Flags sudden neighbor mask drops or runaway area expansion relative to valid neighbors. |
| **Tier 3** | `_lift_foreground_points` & `lift_and_reproject_masks` | Reference + target depth maps present (`occlusion_tol=0.25`) | Trusts VOS masks directly (`"no_depth"`) whenever reference or target depth maps are missing (`depth_map is None`); applies 3D back-projection, visibility classification (`visible`, `occluded`, `out_of_frustum`), and re-entry re-prompting only when real depth maps exist for both reference and target frames. |
| **Tier 4** | `score_reid_crops` (`_HistogramReidEncoder`) | Reference similarity `< min_similarity` (`0.65`) | Evaluates masked HSV histogram similarity (`_crop_masked_rgb`) against anchor appearance and rejects identity switches. |
| **Tier 5** | `apply_manual_mask_overrides` (+ standalone `project_3dgs_mask_hints`) | Unrecovered flagged frames / user overrides in UI | Zeroes unrecovered (`iou_flagged - recovered`) and `reid_rejected` masks, applies per-frame `ManualOverride(frame_idx, mask_path, skip_frame)` from `PUT /api/edits/{id}/masks/{frame_idx}`, and ships `project_3dgs_mask_hints` as a standalone 3DGS-to-2D projection helper. |

---

## 3. Verified Local Behavior & Benchmark Measurements

### Local Hardware Benchmark (`python -m room3d_worker.benchmarks`)
Measured on `Windows-11-10.0.26300-SP0`, Python `3.12.15`, `torch 2.14.1+cu126`, `NVIDIA GeForce RTX 2080` (`sm_75`, `8191.7 MB` VRAM):
- **Warm SAM Click Cache Miss / Hit:** `0.311 ms` / `0.258 ms` (synthetic encoder, bounded LRU CPU cache).
- **Full 5-Tier Tracking Pipeline (8 frames, `160x128`):** `169.753 ms` total (`21.219 ms/frame`), `mean_iou = 1.0`.
- **Missing-Depth VOS Preservation:** `vos_preserved_without_depth = true` (`depth_dir=None` preserves masks verbatim).

### Reference `2 × T4` Methodology
On Kaggle `2 × Tesla T4` (`worker_runner.py`), SAM 2.1 interactive session and VOS tracking execute on `GPU 1` (`cuda:1`, with `inference_state_device="cpu"` offload for video frames) while `GPU 0` remains isolated for Diffusion.

### Automated Tests (`worker/tests/test_track.py`)
All 8 tests in `worker/tests/test_track.py` pass locally:
1. Warm SAM session cache hits bypass redundant encoder passes and evict excess states cleanly.
2. Nested `dict`/`list`/`tuple` tensor snapshots move to CPU on cache insert and back to `predictor.device` on cache hit.
3. `_Sam2Video` runs bidirectional forward+reverse propagation when `ref_frame_idx > 0`, safely detaches CUDA logits, and falls back for single-pass mocks.
4. Multi-point frustum/depth lifting separates temporary occlusion from scene exit when depth is present, and preserves VOS masks when reference or target depth is missing.
5. Masked HSV Re-ID rejects drifted identities cleanly after simulated object swap.
6. Manual overrides (`skip_frame` + custom PNG masks) take precedence in `run_tracking_pipeline` and write `tracking_report.json`.
