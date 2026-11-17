from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from room3d_worker.kaggle.preflight_runner import validate_notebook_syntax

DEFAULT_REPO_URL = "https://github.com/MahamedKhaledSaad11/Room3D_interior_Design.git"


def _cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def _cell_01_gpu_topology() -> dict[str, Any]:
    return _cell(
        '''# Cell 1: GPU Topology & Turing T4 (sm_75) Verification
import json
import os
import subprocess
import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
cuda_ok = torch.cuda.is_available()
device_count = torch.cuda.device_count() if cuda_ok else 0
DIFFUSION_CUDA_ID = "0"
CV_CUDA_ID = "1" if device_count >= 2 else "0"

topology = {
    "cuda_available": cuda_ok,
    "device_count": device_count,
    "torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
    "bf16_required": False,
    "t4_dtype": "float16",
    "gpus": [],
    "assignment": {
        "diffusion": f"cuda:{DIFFUSION_CUDA_ID}",
        "sam2_3dgs_sfm": f"cuda:{CV_CUDA_ID}",
    },
}
if cuda_ok:
    for idx in range(device_count):
        free_b, total_b = torch.cuda.mem_get_info(idx)
        cap = list(torch.cuda.get_device_capability(idx))
        topology["gpus"].append({
            "index": idx,
            "name": torch.cuda.get_device_name(idx),
            "capability": cap,
            "sm_75": cap == [7, 5],
            "free_gb": round(free_b / 1024**3, 2),
            "total_gb": round(total_b / 1024**3, 2),
            "role": "Diffusion (GPU 0)" if idx == 0 else "SAM 2.1 + 3DGS (GPU 1)",
        })
try:
    topology["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
except (FileNotFoundError, subprocess.CalledProcessError) as exc:
    topology["nvidia_smi_error"] = str(exc)

print(json.dumps(topology, indent=2))
'''
    )


def _cell_02_install_deps(repo_url: str | None) -> dict[str, Any]:
    effective_repo = repo_url or DEFAULT_REPO_URL
    return _cell(
        f'''# Cell 2: Install Apache-2.0/MIT/BSD Stack, Sync Repo & Discover Datasets
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = {json.dumps(effective_repo)}
REPO_DIR = Path("/kaggle/working/room3d")
PINNED_PACKAGES = [
    "av>=12.0.0",
    "pycolmap-cuda12==4.2.1",
    "gsplat>=1.4.0",
    "diffusers>=0.32.0",
    "transformers>=4.46.0",
    "accelerate>=1.0.0",
    "bitsandbytes>=0.45.0",
    "optimum-quanto>=0.2.5",
    "scipy>=1.13.0",
    "pydantic>=2.7",
    "httpx>=0.27.0",
    "Pillow>=10.3.0",
    "opencv-python-headless>=4.10.0",
    "xxhash>=3.4.0",
    "matplotlib>=3.8.0",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINNED_PACKAGES], check=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/facebookresearch/sam2.git"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/facebookresearch/vggt.git"],
    check=False,
)
os.environ.setdefault("ROOM3D_VGGT_MODEL", "facebook/VGGT-1B")
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
os.chdir(REPO_DIR)
if (REPO_DIR / "worker" / "pyproject.toml").exists():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "worker"], check=True)
if str(REPO_DIR / "worker") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "worker"))

for mod in ("av", "pycolmap", "gsplat", "diffusers", "transformers", "accelerate", "sam2"):
    importlib.import_module(mod)

DB_SCENES = {{}}
for root in (Path("/kaggle/input"), REPO_DIR / "db"):
    if not root.exists():
        continue
    for name in ("playroom", "drjohnson"):
        for candidate in root.rglob(name):
            if candidate.is_dir() and (candidate / "sparse" / "0" / "cameras.bin").exists():
                DB_SCENES[name] = str(candidate)
                break
print("Discovered precomputed scenes:", json.dumps(DB_SCENES, indent=2))
'''
    )


def _cell_03_lab_config() -> dict[str, Any]:
    return _cell(
        '''# Cell 3: Interactive Lab Config & Subprocess Stage Helper
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCENE_NAME = "playroom"
PIPELINE_MODE = "precomputed_colmap"  # "precomputed_colmap" or "full_w7_pipeline"
VIDEO_PATH = str(Path("/kaggle/working/room3d/db/sample_room.mp4"))
MAX_LONG_SIDE = 960  # -> 960x624 mod-16 for db/playroom & db/drjohnson
TRAIN_ITERATIONS = 7000
REF_FRAME_IDX = 0
CLICK_POINTS = [{"x": 480, "y": 312, "label": 1}]
EDIT_PROMPT = "modern emerald velvet armchair"
NEGATIVE_PROMPT = ""
DIFFUSION_VARIANT = "flux2_klein_4b"  # "flux2_klein_4b" or "sdxl_inpaint_t4"
DIFFUSION_QUANTIZATION = "nf4"  # "nf4", "int8", or "none"
VGGT_MODEL_ID = os.environ.get("ROOM3D_VGGT_MODEL", "facebook/VGGT-1B")
SEEDS = [101, 202, 303, 404]
CHOSEN_SEED = 101
REBUILD_MODE = "masked_finetune"

WORK_ROOT = Path("/kaggle/working/runs") / SCENE_NAME
V1_DIR = WORK_ROOT / "v1"
V2_DIR = WORK_ROOT / "v2"
TRACK_DIR = WORK_ROOT / "track"
DIFF_DIR = WORK_ROOT / "diffusion"
BENCH_DIR = WORK_ROOT / "benchmarks"
for d in (V1_DIR, V2_DIR, TRACK_DIR, DIFF_DIR, BENCH_DIR):
    d.mkdir(parents=True, exist_ok=True)

STAGE_METRICS = {}

def run_stage_subprocess(stage: str, spec: dict, gpu_ordinal: str, out_json: Path) -> dict:
    spec_path = out_json.with_suffix(".spec.json")
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = "7.5"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_ordinal)
    env["ROOM3D_VGGT_MODEL"] = VGGT_MODEL_ID
    worker_pkg = Path.cwd() / "worker"
    if worker_pkg.exists():
        prev_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{worker_pkg}:{prev_pp}" if prev_pp else str(worker_pkg)
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "room3d_worker.stage_cli", stage, "--spec", str(spec_path), "--out", str(out_json)],
        env=env,
        check=True,
    )
    res = json.loads(out_json.read_text(encoding="utf-8"))
    STAGE_METRICS[stage] = {"elapsed_sec": round(time.perf_counter() - t0, 3), "gpu": f"cuda:{gpu_ordinal}"}
    return res

print(json.dumps({"scene": SCENE_NAME, "mode": PIPELINE_MODE, "work_root": str(WORK_ROOT)}, indent=2))
'''
    )


def _cell_04_extract() -> dict[str, Any]:
    return _cell(
        '''# Cell 4: Stage 1 (s01_extract / Precomputed COLMAP 960x624 Mod-16 Prep Subprocess)
from pathlib import Path
import av
import matplotlib.pyplot as plt
from PIL import Image

if PIPELINE_MODE == "precomputed_colmap":
    source_dir = DB_SCENES.get(SCENE_NAME)
    if not source_dir:
        raise FileNotFoundError(f"Scene {SCENE_NAME} not found in DB_SCENES={DB_SCENES}")
    extract_res = run_stage_subprocess(
        "poses",
        {
            "pose_backend": "precomputed_colmap",
            "source_scene_dir": source_dir,
            "image_dir": str(V1_DIR / "images"),
            "output_dir": str(V1_DIR),
            "total_frames": 0,
            "width": 960,
            "height": 624,
            "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 480.0, "cy": 312.0},
            "project_id": SCENE_NAME,
            "version_id": "v1",
            "max_long_side": MAX_LONG_SIDE,
        },
        CV_CUDA_ID,
        V1_DIR / "precomputed_prep.json",
    )
else:
    if not Path(VIDEO_PATH).exists() and SCENE_NAME in DB_SCENES:
        img_src = Path(DB_SCENES[SCENE_NAME]) / "images"
        src_frames = sorted(p for p in img_src.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})[:120]
        if not src_frames:
            raise FileNotFoundError(f"No images found under {img_src} to synthesize {VIDEO_PATH}")
        Path(VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_frames[0]) as first_im:
            vw, vh = first_im.size
        with av.open(VIDEO_PATH, mode="w") as container:
            stream = container.add_stream("libx264", rate=4)
            stream.width = vw - (vw % 2)
            stream.height = vh - (vh % 2)
            stream.pix_fmt = "yuv420p"
            for fp in src_frames:
                with Image.open(fp) as im:
                    rgb = im.convert("RGB").resize((stream.width, stream.height), Image.Resampling.BILINEAR)
                    frame = av.VideoFrame.from_image(rgb)
                    for pkt in stream.encode(frame):
                        container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
    extract_res = run_stage_subprocess(
        "extract",
        {
            "video_path": VIDEO_PATH,
            "output_dir": str(V1_DIR),
            "max_long_side": MAX_LONG_SIDE,
            "target_fps": 4.0,
            "max_frames": 180,
        },
        CV_CUDA_ID,
        V1_DIR / "extract_out.json",
    )

frame_files = sorted(p for p in (V1_DIR / "images").iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
if frame_files:
    with Image.open(frame_files[0]) as sample:
        w, h = sample.size
    print({"frame_count": len(frame_files), "width": w, "height": h, "mod16": (w % 16 == 0 and h % 16 == 0)})
    fig, axes = plt.subplots(1, min(4, len(frame_files)), figsize=(14, 3))
    axes_list = axes if hasattr(axes, "__len__") else [axes]
    for ax, fp in zip(axes_list, frame_files[:4]):
        ax.imshow(Image.open(fp))
        ax.set_title(fp.name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
'''
    )


def _cell_05_poses() -> dict[str, Any]:
    return _cell(
        '''# Cell 5: Stage 2 (s02_poses Subprocess on GPU 1 + Sparse Cloud Plot)
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if PIPELINE_MODE == "full_w7_pipeline":
    idx_path = Path(extract_res.get("index_path") or (V1_DIR / "extract_index.json"))
    idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
    kept = idx_data.get("frames", [])
    with Image.open(V1_DIR / kept[0]["image_relpath"]) as img0:
        w, h = img0.size
    intr = {"fx": float(max(w, h)), "fy": float(max(w, h)), "cx": w / 2.0, "cy": h / 2.0}
    vggt_out = V1_DIR / "poses_vggt"
    vggt_out.mkdir(exist_ok=True)
    vggt_res = None
    try:
        vggt_res = run_stage_subprocess(
            "poses",
            {
                "pose_backend": "vggt_chunked",
                "image_dir": str(V1_DIR / "images"),
                "output_dir": str(vggt_out),
                "total_frames": len(kept),
                "width": w,
                "height": h,
                "intrinsics": intr,
                "project_id": SCENE_NAME,
                "version_id": "v1",
                "frame_records": kept,
            },
            CV_CUDA_ID,
            V1_DIR / "poses_vggt_out.json",
        )
    except Exception as vggt_err:
        vggt_res = {"vggt_skipped_or_failed": str(vggt_err)}
        print("W_7.md §4.1 VGGT notice:", json.dumps(vggt_res))
    poses_res = run_stage_subprocess(
        "poses",
        {
            "pose_backend": "colmap_global",
            "image_dir": str(V1_DIR / "images"),
            "output_dir": str(V1_DIR),
            "total_frames": len(kept),
            "width": w,
            "height": h,
            "intrinsics": intr,
            "project_id": SCENE_NAME,
            "version_id": "v1",
            "frame_records": kept,
        },
        CV_CUDA_ID,
        V1_DIR / "poses_colmap_out.json",
    )
    print("W_7.md §4.1 Comparison:", json.dumps({"vggt_chunked": vggt_res, "colmap_global": poses_res}, indent=2))

manifest = json.loads((V1_DIR / "manifest.json").read_text(encoding="utf-8"))
print("Verified scaled intrinsics:", manifest["width"], manifest["height"], manifest["intrinsics"])
sp_path = V1_DIR / "sparse_points.json"
if sp_path.exists():
    sp_raw = json.loads(sp_path.read_text(encoding="utf-8"))
    if isinstance(sp_raw, list) and sp_raw and isinstance(sp_raw[0], dict):
        pts = np.asarray([p["position"] for p in sp_raw], dtype=np.float32)
        cols = np.asarray([p.get("color", [0, 180, 180]) for p in sp_raw], dtype=np.float32) / 255.0
    else:
        raw_pts = sp_raw.get("points", []) if isinstance(sp_raw, dict) else sp_raw
        pts = np.asarray(raw_pts, dtype=np.float32)
        cols = np.clip(pts[:, 3:6] / 255.0, 0.0, 1.0) if pts.ndim == 2 and pts.shape[1] >= 6 else "teal"
    if len(pts) > 0:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        step = max(1, len(pts) // 2000)
        sub = pts[::step]
        sub_cols = cols[::step] if isinstance(cols, np.ndarray) else cols
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=1, c=sub_cols)
        ax.set_title(f"{SCENE_NAME} Sparse Points ({len(pts)})")
        plt.show()
'''
    )


def _cell_06_train() -> dict[str, Any]:
    return _cell(
        '''# Cell 6: Stage 3 (s03_train gsplat MCMC Subprocess on GPU 1)
import json
import torch

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(int(CV_CUDA_ID))

train_res = run_stage_subprocess(
    "train",
    {
        "output_dir": str(V1_DIR),
        "manifest_path": str(V1_DIR / "manifest.json"),
        "device": "cuda:0",
        "iterations": TRAIN_ITERATIONS,
    },
    CV_CUDA_ID,
    V1_DIR / "train_out.json",
)
print(json.dumps({"train": train_res, "timing": STAGE_METRICS.get("train", {})}, indent=2))
'''
    )


def _cell_07_export() -> dict[str, Any]:
    return _cell(
        '''# Cell 7: Stage 4 (s04_export .ply + .spz v2 Subprocess & Week 1 Metrics Table)
import json

export_res = run_stage_subprocess(
    "export",
    {
        "output_dir": str(V1_DIR),
        "manifest_path": str(V1_DIR / "manifest.json"),
        "ckpt_path": str(V1_DIR / "gs_state.npz"),
    },
    CV_CUDA_ID,
    V1_DIR / "export_out.json",
)
print("W_7.md §5.5 Stage Metrics Table:")
for stage_name, info in STAGE_METRICS.items():
    print(f"  {stage_name:14s} | gpu={info['gpu']} | elapsed={info['elapsed_sec']}s")
print(json.dumps(export_res, indent=2))
'''
    )


def _cell_08_segment_track() -> dict[str, Any]:
    return _cell(
        '''# Cell 8: Stage 5 (s05_track SAM 2.1 Subprocess on GPU 1 + Mask Filmstrip)
import json
import urllib.request
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
ckpt_path = Path("/kaggle/working/checkpoints/sam2.1_hiera_small.pt")
if not ckpt_path.exists():
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SAM2_URL, ckpt_path)

manifest = json.loads((V1_DIR / "manifest.json").read_text(encoding="utf-8"))
image_paths = {int(f["frame_idx"]): str(V1_DIR / f["image_relpath"]) for f in manifest["frames"] if f.get("cam_from_world")}

fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(Image.open(image_paths[REF_FRAME_IDX]))
for pt in CLICK_POINTS:
    ax.plot(pt["x"], pt["y"], "go" if pt["label"] == 1 else "rx", markersize=10)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_title(f"Reference Frame {REF_FRAME_IDX} Click Prompts")
plt.show()

track_res = run_stage_subprocess(
    "track",
    {
        "manifest_path": str(V1_DIR / "manifest.json"),
        "image_paths": image_paths,
        "ref_frame_idx": REF_FRAME_IDX,
        "points": CLICK_POINTS,
        "output_dir": str(TRACK_DIR),
        "depth_dir": str(V1_DIR / "depths"),
        "sam2_config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2_checkpoint": str(ckpt_path),
        "device": "cuda:0",
    },
    CV_CUDA_ID,
    TRACK_DIR / "track_out.json",
)
print(json.dumps({"stats": track_res.get("stats", {}), "timing": STAGE_METRICS.get("track", {})}, indent=2))

mask_items = sorted((int(k), Path(v)) for k, v in track_res.get("saved_masks", {}).items())[:4]
if mask_items:
    fig, axes = plt.subplots(1, len(mask_items), figsize=(14, 3))
    axes_list = axes if hasattr(axes, "__len__") else [axes]
    for ax, (idx, mp) in zip(axes_list, mask_items):
        ax.imshow(np.asarray(Image.open(mp)), cmap="gray")
        ax.set_title(f"Mask {idx}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
'''
    )


def _cell_09a_diffusion_preview() -> dict[str, Any]:
    return _cell(
        '''# Cell 9a: Stage 6a (s06_diffusion Seed Previews on GPU 0 — float16 + NF4/INT8 + Offload)
import json
import matplotlib.pyplot as plt
import torch
from PIL import Image
from room3d_worker.pipeline.s06_diffusion import resolve_t4_dtype

if torch.cuda.is_available():
    t4_cap = torch.cuda.get_device_capability(int(DIFFUSION_CUDA_ID))
    assert resolve_t4_dtype(t4_cap) in {"float16", "bfloat16"}
    if t4_cap[0] < 8:
        assert resolve_t4_dtype(t4_cap) == "float16", "T4 sm_75 must enforce float16"

track_out = json.loads((TRACK_DIR / "track_out.json").read_text(encoding="utf-8"))
manifest = json.loads((V1_DIR / "manifest.json").read_text(encoding="utf-8"))
ref_img_path = str(V1_DIR / next(f["image_relpath"] for f in manifest["frames"] if int(f["frame_idx"]) == REF_FRAME_IDX))
ref_mask_path = track_out["saved_masks"][str(REF_FRAME_IDX)]

preview_dir = DIFF_DIR / "previews"
preview_res = run_stage_subprocess(
    "diffusion_preview",
    {
        "ref_image_path": ref_img_path,
        "ref_mask_path": ref_mask_path,
        "prompt": EDIT_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "model_variant": DIFFUSION_VARIANT,
        "quantization": DIFFUSION_QUANTIZATION,
        "offload_text_encoder": True,
        "seeds": SEEDS,
        "output_dir": str(preview_dir),
        "device": "cuda:0",
    },
    DIFFUSION_CUDA_ID,
    DIFF_DIR / "preview_out.json",
)
print("Preview telemetry:", json.dumps(preview_res, indent=2))

seed_entries = preview_res.get("seeds", [])
if seed_entries:
    fig, axes = plt.subplots(1, len(seed_entries), figsize=(14, 3.5))
    axes_list = axes if hasattr(axes, "__len__") else [axes]
    for ax, entry in zip(axes_list, seed_entries):
        ax.imshow(Image.open(entry["path"]))
        ax.set_title(f"Seed {entry['seed']} ({entry.get('sec_per_frame', 0):.2f}s/f)")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
'''
    )


def _cell_09b_diffusion_apply() -> dict[str, Any]:
    return _cell(
        '''# Cell 9b: Stage 6b (s06_diffusion Propagate Approved Seed on GPU 0)
import json
import matplotlib.pyplot as plt
from PIL import Image

manifest = json.loads((V1_DIR / "manifest.json").read_text(encoding="utf-8"))
track_out = json.loads((TRACK_DIR / "track_out.json").read_text(encoding="utf-8"))
frame_paths = {int(f["frame_idx"]): str(V1_DIR / f["image_relpath"]) for f in manifest["frames"] if f.get("cam_from_world")}
ref_seed_path = str(DIFF_DIR / "previews" / f"seed_{CHOSEN_SEED}.png")

apply_res = run_stage_subprocess(
    "diffusion_apply",
    {
        "frame_paths": frame_paths,
        "mask_paths": track_out["saved_masks"],
        "ref_frame_idx": REF_FRAME_IDX,
        "ref_edited_path": ref_seed_path,
        "prompt": EDIT_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "model_variant": DIFFUSION_VARIANT,
        "quantization": DIFFUSION_QUANTIZATION,
        "offload_text_encoder": True,
        "seed": CHOSEN_SEED,
        "output_dir": str(DIFF_DIR),
        "device": "cuda:0",
    },
    DIFFUSION_CUDA_ID,
    DIFF_DIR / "apply_out.json",
)
assert apply_res.get("max_unmasked_diff", -1) == 0, "Unmasked background pixels must remain identical"
print("Diffusion propagation telemetry:", json.dumps({
    "sec_per_frame": apply_res.get("sec_per_frame"),
    "peak_vram_mb": apply_res.get("peak_vram_mb"),
    "quantization_mode": apply_res.get("quantization_mode"),
    "max_unmasked_diff": apply_res.get("max_unmasked_diff"),
}, indent=2))

edited_imgs = sorted((DIFF_DIR / "edited_frames").glob("*.png"))[:4]
if edited_imgs:
    fig, axes = plt.subplots(1, len(edited_imgs), figsize=(14, 3))
    axes_list = axes if hasattr(axes, "__len__") else [axes]
    for ax, ep in zip(axes_list, edited_imgs):
        ax.imshow(Image.open(ep))
        ax.set_title(ep.name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
'''
    )


def _cell_10_rebuild() -> dict[str, Any]:
    return _cell(
        '''# Cell 10: Stage 7 (s07_rebuild Masked 3DGS Finetune Subprocess on GPU 1 -> v2 scene.spz)
import json

rebuild_res = run_stage_subprocess(
    "rebuild",
    {
        "base_dir": str(V1_DIR),
        "edited_frames_dir": str(DIFF_DIR / "edited_frames"),
        "masks_dir": str(TRACK_DIR / "masks"),
        "output_dir": str(V2_DIR),
        "project_id": SCENE_NAME,
        "version_id": "v2",
        "parent_version_id": "v1",
        "mode": REBUILD_MODE,
        "device": "cuda:0",
    },
    CV_CUDA_ID,
    V2_DIR / "rebuild_out.json",
)
print(json.dumps({"rebuild": rebuild_res, "timing": STAGE_METRICS.get("rebuild", {})}, indent=2))
'''
    )


def _cell_11_bundle_zip() -> dict[str, Any]:
    return _cell(
        '''# Cell 11: Persist Complete v1 + v2 Scene Bundle & Benchmark Reports to ZIP
import json
import zipfile
from pathlib import Path

zip_path = Path("/kaggle/working") / f"{SCENE_NAME}_room3d_bundle.zip"
summary = {
    "scene_name": SCENE_NAME,
    "pipeline_mode": PIPELINE_MODE,
    "stage_metrics": STAGE_METRICS,
}
(BENCH_DIR / "kaggle_lab_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for item in WORK_ROOT.rglob("*"):
        if item.is_file():
            zf.write(item, item.relative_to(WORK_ROOT))

print(json.dumps({"bundle_zip": str(zip_path), "size_mb": round(zip_path.stat().st_size / 1024**2, 2)}, indent=2))
'''
    )


def _cell_12_daemon(backend_url: str, worker_token: str, worker_id: str) -> dict[str, Any]:
    return _cell(
        f'''# Cell 12: Optional Live Pull-Worker Daemon Mode
import os
import subprocess
import sys
from pathlib import Path

BACKEND_URL = {json.dumps(backend_url)}
WORKER_TOKEN = os.environ.get("ROOM3D_WORKER_TOKEN") or {json.dumps(worker_token)}
WORKER_ID = os.environ.get("ROOM3D_WORKER_ID", {json.dumps(worker_id)})
RUN_DAEMON = os.environ.get("ROOM3D_RUN_DAEMON", "0").lower() in {{"1", "true", "yes"}}

env = os.environ.copy()
env["TORCH_CUDA_ARCH_LIST"] = "7.5"
env["ROOM3D_BACKEND_URL"] = BACKEND_URL
env["ROOM3D_WORKER_TOKEN"] = WORKER_TOKEN
worker_pkg = Path.cwd() / "worker"
if worker_pkg.exists():
    prev_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{{worker_pkg}}:{{prev_pp}}" if prev_pp else str(worker_pkg)

cmd = [
    sys.executable,
    "-m",
    "room3d_worker",
    "--backend-url",
    BACKEND_URL,
    "--worker-token",
    WORKER_TOKEN,
    "--worker-id",
    WORKER_ID,
    "--heartbeat-interval",
    "15",
]
if RUN_DAEMON:
    subprocess.run(cmd, env=env, check=True)
else:
    print("Pull-worker command ready (set ROOM3D_RUN_DAEMON=1 or RUN_DAEMON=True to connect):", " ".join(cmd))
'''
    )


def build_worker_notebook(
    backend_url: str = "https://example.invalid",
    worker_token: str = "room3d-dev-worker-token",
    *,
    worker_id: str = "kaggle-t4x2",
    repo_url: str | None = DEFAULT_REPO_URL,
) -> dict[str, Any]:
    cleaned_url = backend_url.strip().rstrip("/")
    if not cleaned_url.startswith(("http://", "https://")):
        raise ValueError("backend_url must start with http:// or https://")
    if not worker_token.strip():
        raise ValueError("worker_token must be non-empty")
    cells = [
        _cell_01_gpu_topology(),
        _cell_02_install_deps(repo_url),
        _cell_03_lab_config(),
        _cell_04_extract(),
        _cell_05_poses(),
        _cell_06_train(),
        _cell_07_export(),
        _cell_08_segment_track(),
        _cell_09a_diffusion_preview(),
        _cell_09b_diffusion_apply(),
        _cell_10_rebuild(),
        _cell_11_bundle_zip(),
        _cell_12_daemon(cleaned_url, worker_token.strip(), worker_id.strip() or "kaggle-t4x2"),
    ]
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "cells": cells,
    }
    errors = validate_notebook_syntax(notebook)
    if errors:
        raise ValueError("invalid generated worker notebook: " + "; ".join(errors))
    return notebook


def generate_worker_notebook(
    output_path: Path,
    backend_url: str = "https://example.invalid",
    worker_token: str = "room3d-dev-worker-token",
    *,
    worker_id: str = "kaggle-t4x2",
    repo_url: str | None = DEFAULT_REPO_URL,
) -> Path:
    """Generate the 12-cell hybrid Kaggle 2xT4 lab and pull-worker notebook (.ipynb)."""
    notebook = build_worker_notebook(backend_url, worker_token, worker_id=worker_id, repo_url=repo_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Room3D 12-cell Kaggle dual-T4 lab + worker notebook.")
    parser.add_argument("--output", type=Path, default=Path("kaggle_bundle/room3d_kaggle_lab.ipynb"))
    parser.add_argument("--backend-url", default="https://example.invalid")
    parser.add_argument("--worker-token", default="room3d-dev-worker-token")
    parser.add_argument("--worker-id", default="kaggle-t4x2")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    args = parser.parse_args(argv)
    written = generate_worker_notebook(
        args.output,
        args.backend_url,
        args.worker_token,
        worker_id=args.worker_id,
        repo_url=args.repo_url,
    )
    print(str(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
