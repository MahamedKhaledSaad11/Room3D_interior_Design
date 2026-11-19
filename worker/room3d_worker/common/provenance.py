from __future__ import annotations

import os
import subprocess
import sys
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def runtime_provenance() -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for name in ("room3d-worker", "torch", "gsplat", "diffusers", "transformers", "accelerate", "bitsandbytes", "sam2", "av"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    commit = os.environ.get("ROOM3D_SOURCE_COMMIT")
    if commit is None:
        try:
            result = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, timeout=5, check=False)
            commit = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            commit = None
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for source in sorted(root.rglob("*.py")):
        digest.update(source.relative_to(root).as_posix().encode())
        digest.update(source.read_bytes())
    return {"source_commit": commit, "worker_source_sha256": digest.hexdigest(), "python": sys.version, "packages": packages}
