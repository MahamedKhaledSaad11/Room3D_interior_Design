from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from room3d_worker.kaggle.preflight_runner import validate_notebook_syntax


def _cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def _cell_gpu_topology() -> dict[str, Any]:
    return _cell(
        '''import json
import os
import subprocess
import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
topology = {
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
    "bf16_required": False,
    "gpus": [],
}
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(idx)
        topology["gpus"].append({
            "index": idx,
            "name": torch.cuda.get_device_name(idx),
            "capability": list(torch.cuda.get_device_capability(idx)),
            "free_gb": round(free_b / 1024**3, 2),
            "total_gb": round(total_b / 1024**3, 2),
            "role": "Diffusion (GPU 0)" if idx == 0 else "SAM 2.1 + 3DGS (GPU 1)",
        })
try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    topology["nvidia_smi"] = smi
except (FileNotFoundError, subprocess.CalledProcessError) as exc:
    topology["nvidia_smi_error"] = str(exc)

topology["assignment"] = {
    "diffusion": "cuda:0",
    "sam2_and_3dgs": "cuda:1" if topology["device_count"] >= 2 else "cuda:0",
}
print(json.dumps(topology, indent=2))
'''
    )


def _cell_install_dependencies(repo_url: str | None) -> dict[str, Any]:
    repo_block = (
        f'REPO_URL = {json.dumps(repo_url)}\n'
        'if not os.path.exists("/kaggle/working/room3d"):\n'
        '    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/kaggle/working/room3d"], check=True)\n'
        'os.chdir("/kaggle/working/room3d")\n'
        if repo_url
        else 'if os.path.exists("/kaggle/working/room3d"):\n'
        '    os.chdir("/kaggle/working/room3d")\n'
    )
    return _cell(
        f'''import importlib
import os
import subprocess
import sys

# Install Apache-2.0 / MIT / BSD-3-Clause runtime stack for Kaggle 2xT4 (cu130).
PINNED_PACKAGES = [
    "av>=12.0.0",
    "pycolmap-cuda12==4.2.1",
    "gsplat>=1.4.0",
    "diffusers>=0.32.0",
    "transformers>=4.46.0",
    "accelerate>=1.0.0",
    "scipy>=1.13.0",
    "pydantic>=2.7",
    "httpx>=0.27.0",
    "Pillow>=10.3.0",
    "opencv-python-headless>=4.10.0",
    "xxhash>=3.4.0",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINNED_PACKAGES], check=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/facebookresearch/sam2.git"],
    check=True,
)
{repo_block}if os.path.exists("worker/pyproject.toml"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "worker"], check=True)

for mod in ("pycolmap", "gsplat", "sam2", "diffusers", "av"):
    importlib.import_module(mod)
print("Verified Apache-2.0/MIT/BSD dependencies: pycolmap, gsplat, sam2, diffusers, av")
'''
    )


def _cell_launch_worker(backend_url: str, worker_token: str, worker_id: str) -> dict[str, Any]:
    return _cell(
        f'''import os
import subprocess
import sys
from pathlib import Path

BACKEND_URL = {json.dumps(backend_url)}
WORKER_TOKEN = os.environ.get("ROOM3D_WORKER_TOKEN") or {json.dumps(worker_token)}
WORKER_ID = os.environ.get("ROOM3D_WORKER_ID", {json.dumps(worker_id)})

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
subprocess.run(cmd, env=env, check=True)
'''
    )


def build_worker_notebook(
    backend_url: str,
    worker_token: str,
    *,
    worker_id: str = "kaggle-t4x2",
    repo_url: str | None = None,
) -> dict[str, Any]:
    cleaned_url = backend_url.strip().rstrip("/")
    if not cleaned_url.startswith(("http://", "https://")):
        raise ValueError("backend_url must start with http:// or https://")
    if not worker_token.strip():
        raise ValueError("worker_token must be non-empty")
    cells = [
        _cell_gpu_topology(),
        _cell_install_dependencies(repo_url),
        _cell_launch_worker(cleaned_url, worker_token.strip(), worker_id.strip() or "kaggle-t4x2"),
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
    backend_url: str,
    worker_token: str,
    *,
    worker_id: str = "kaggle-t4x2",
    repo_url: str | None = None,
) -> Path:
    """Generate a Kaggle 2xT4 pull-worker notebook (.ipynb) with GPU checks, pinned deps, and daemon launch."""
    notebook = build_worker_notebook(backend_url, worker_token, worker_id=worker_id, repo_url=repo_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Kaggle dual-T4 Room3D pull-worker notebook.")
    parser.add_argument("--output", type=Path, default=Path("kaggle_bundle/room3d_t4_worker.ipynb"))
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--worker-token", required=True)
    parser.add_argument("--worker-id", default="kaggle-t4x2")
    parser.add_argument("--repo-url", default=None)
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
