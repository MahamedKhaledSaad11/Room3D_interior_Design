# Week 1 Reconstruction & Pose Backend Benchmark Report

## 1. Purpose & Evidence Classification

This report documents the hardware preflight measurements on Kaggle Dual Tesla T4 GPUs, the implemented Week 1 reconstruction pipeline (`s01_extract` through `s04_export`), and the verified local synthetic invariants. Every figure below is explicitly labeled by verification scope:

- **Verified (Kaggle `S0.2b`)**: Measured directly inside `mohamedelbasyouni55/room3d-t4-preflight` (version `1` and version `3`, `KernelWorkerStatus.COMPLETE`).
- **Verified (Local RTX 2080 `sm_75` + CPU + Playwright)**: Measured locally via `python -m room3d_worker.benchmarks` and automated unit/E2E suites in `worker/tests/` and `tests/e2e/test_s18_week1_vertical.py`.
- **Reference T4x2 Methodology / Protocol Target**: Dual-T4 execution layout (`worker/room3d_worker/kaggle/worker_runner.py`), assigning `GPU 0` to Diffusion and `GPU 1` to SAM 2.1 + 3DGS (`WorkerClient.BATCH_GPU`) for full multi-minute video runs on Kaggle.

---

## 2. Verified Hardware Baseline (`Kaggle Dual Tesla T4`)

Measured by `worker/room3d_worker/kaggle/preflight_runner.py` in Kaggle runs `S0.2b-v1`..`v3`:

| Metric | Verified Value | Engineering Impact |
| :--- | :--- | :--- |
| GPU Hardware | `2 × NVIDIA Tesla T4` (`15360 MiB` each) | Dedicated `cuda:1` for SAM 2.1 + 3DGS, `cuda:0` for Diffusion (`WorkerClient.BATCH_GPU`) |
| Compute Capability | `(7, 5)` (`sm_75`, Turing) | Hardware `bfloat16` absent; pipeline enforces `float16` / `np.isfinite` checks and Kaggle `torch.isfinite` preflight probes |
| Idle GPU 0 VRAM | `14.41 GB` free | Requires CPU offload and bounded pixel/tile budgets for 4B diffusion models |
| `/kaggle/working` Disk | `19.5 GB` free | Requires artifact cleanup and compressed `.spz` output (~10× smaller than raw `.ply`) |
| OS / Runtime | `Linux-6.18.55+`, Python `3.13`, `torch 2.11.0+cu130` | Unpinned `bitsandbytes` fails `diffusers` import (`libbitsandbytes_cuda130.so`); excluded |
| `pycolmap` Version | `4.2.1` (`pycolmap-cuda12`) | Verified `ALIKED_N16ROT`, `ALIKED_LIGHTGLUE`, `onnx_support: true`, `global_mapping` |

---

## 3. Pose Backend Comparison & Selection Logic (`s02_poses.py`)

| Backend Identifier | Library / Method | License Status | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| `colmap_global` | `pycolmap` (`ALIKED_N16ROT` + `ALIKED_LIGHTGLUE` -> `global_mapping`) | `BSD-3-Clause` (Allowed) | Primary Week 1 backend (`PyColmapBackend("global")`) |
| `colmap_incremental` | `pycolmap.incremental_mapping` (`ba_use_gpu=True`) | `BSD-3-Clause` (Allowed) | Automatic fallback when registration ratio `< 0.85` |
| `vggt_chunked` | `VGGTChunkedBackend` (`chunk_size=32`, `overlap=6`, `fp16`, Kabsch `align_sim3`) | Benchmark/Commercial config | Feed-forward fallback for low-texture walls (`Sim(3)` overlap stitching) |
| `map_anything` | `MapAnythingBackend` (`fp16`) | `Apache-2.0` (Allowed) | Single-pass metric/pose feed-forward alternative |
| `DUSt3R` / `MASt3R` / `Pi3` | Excluded by policy | Non-commercial | Forbidden in runtime and dependencies |

### Key Verified Pose & Coordinate Safeguards
1. **Database Isolation (`s02_poses.py`)**: Enforces `database.db` outside `image_dir` before invoking `pycolmap.global_mapping(database_path, image_path, output_path)` (discovered during `S0.2b-v3` preflight).
2. **Mod-16 & Principal Point Sync (`coords.mod16_crop_or_pad_params`)**: Ensures undistorted `PINHOLE` frames are divisible by `16` and adjusts `cx`, `cy` by exact crop offsets `(x0, y0)` without altering `fx`, `fy`.
3. **Non-Destructive Gravity Alignment (`coords.align_gravity_up`)**: Computes floor normal via seeded RANSAC with SVD fallback, guarantees `det(R) == +1`, and writes `T_viewer_from_colmap` to `manifest.json` without mutating `sparse/0/`.

---

## 4. Extraction, 3DGS Training & Export Verification (`s01`, `s03`, `s04`)

| Stage | Module | Verified Local Test Checks |
| :--- | :--- | :--- |
| `01_extract` | `s01_extract.py` | Drops blurry/static frames (`baseline * 0.35`, `min_parallax_frac=0.04`), outputs `mod 16` JPEGs, verifies zero EXIF/GPS/audio metadata (`worker/tests/test_extract.py`). |
| `02_poses` | `s02_poses.py` | Tests `< 0.85` registration fallback from `colmap_global` to `colmap_incremental`, Sim(3) recovery (`atol=1e-4`), `sparse/0` layout, `mod 16` intrinsics (`worker/tests/test_poses.py`). |
| `03_train` | `s03_train.py` | Supports full `(N,3)` scales, `(N,4)` quaternions, `test_every=0`, `gsplat` + synthetic fallback, `depths/{idx:05d}.npy` export (`worker/tests/test_train.py`, `worker/tests/test_export.py`). |
| `04_export` | `s04_export.py` | Validates SPZ v2 (`0x5053474e`) + 3DGS PLY + `sparse_points.json` + Pydantic `SceneManifest` (`worker/tests/test_export.py`, `tests/e2e/test_s18_week1_vertical.py`). |

---

## 5. Local Hardware Measurement Output (`room3d_worker.benchmarks`)

Measured on local workstation (`Windows-11-10.0.26300-SP0`, Python `3.12.15`, `torch 2.14.1+cu126`, `NVIDIA GeForce RTX 2080` `sm_75` `8191.7 MB` VRAM, `12` CPUs) using `python -m room3d_worker.benchmarks` (`8` synthetic `160x128` frames, `10,000` Gaussians):

| Stage | Benchmark Scope | Measured Local Value |
| :--- | :--- | :--- |
| `s02_poses` (`colmap_global` synth) | 8 frames, `mod 16` + gravity alignment + manifest | `19.207 ms` (`8/8` registered, `manifest_mod16_ok: true`) |
| `s02_poses` (`vggt_chunked` synth) | 8 frames, overlapping chunks + `Sim(3)` alignment + `sparse/0` | `13.4 ms` (`8/8` registered) |
| `s04_export` (`export_scene`) | 10,000 Gaussians -> `.ply` (`710,475 B`) + SPZ v2 (`128,899 B`) | `135.162 ms` (`5.51x` compression, magic `0x5053474e`, v2) |

### Reference Kaggle `2 × T4` Benchmark Methodology
1. Generate the dual-T4 pull-worker notebook via `room3d_worker.kaggle.worker_runner.generate_worker_notebook(output_path, backend_url, worker_token)`.
2. Cell 1 verifies `2 × Tesla T4` (`sm_75`, `TORCH_CUDA_ARCH_LIST="7.5"`, `bf16_required=False`) and partitions `GPU 0` (Diffusion) and `GPU 1` (SAM 2.1 + 3DGS).
3. Cell 2 installs pinned Apache-2.0/MIT/BSD packages (`av`, `pycolmap-cuda12==4.2.1`, `gsplat`, `sam2`, `diffusers`, `transformers`, `accelerate`) without `bitsandbytes`.
4. Cell 3 launches `python -m room3d_worker --backend-url ... --worker-token ... --heartbeat-interval 15` to claim real `reconstruct_week1` tasks and report stage telemetry/VRAM back to the control plane.
