"""Import a Kaggle hybrid lab scene bundle (.zip or directory) into Room3D storage and DB."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import get_settings  # noqa: E402
from app.db import Base  # noqa: E402
from app.models import Project, SceneVersion  # noqa: E402

_SCENE_FILES = (
    "manifest.json",
    "scene.spz",
    "scene.ply",
    "sparse_points.json",
    "gs_state.npz",
    "gs_ckpt.pt",
    "extract_index.json",
)
_SCENE_DIRS = ("images", "depths", "masks", "edited_frames")
_MANIFEST_ARTIFACT_KEYS = (
    "spz_relpath",
    "sog_relpath",
    "gs_ckpt_relpath",
    "sparse_points_relpath",
    "collision_glb_relpath",
    "depth_dir",
    "sparse_dir",
)
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,35}$")
_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_safe_id(value: str, label: str) -> str:
    if not value or ".." in value or "/" in value or "\\" in value or not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(f"Invalid {label} '{value}': must match ^[A-Za-z0-9][A-Za-z0-9._-]{{0,35}}$ without traversal")
    return value


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    dest_root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (dest_root / member.filename).resolve()
            if not member_path.is_relative_to(dest_root):
                raise ValueError(f"Unsafe archive member path: {member.filename}")
        zf.extractall(dest_root)
    return dest_root


def _resolve_bundle_root(extracted_root: Path) -> Path:
    if (extracted_root / "v1").is_dir():
        return extracted_root
    subdirs = [p for p in extracted_root.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and (subdirs[0] / "v1").is_dir():
        return subdirs[0]
    if any(p.is_dir() and (p / "manifest.json").is_file() for p in subdirs):
        return extracted_root
    if len(subdirs) == 1 and any(p.is_dir() and (p / "manifest.json").is_file() for p in subdirs[0].iterdir()):
        return subdirs[0]
    raise ValueError(f"No version directories (e.g. 'v1/') found inside bundle {extracted_root}")


def _discover_version_dirs(bundle_root: Path) -> list[tuple[str, Path]]:
    found: dict[str, Path] = {}
    for entry in bundle_root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name in {"v1", "v2"} or (entry / "manifest.json").is_file():
            found[entry.name] = entry
    if not found:
        raise ValueError(f"Bundle root {bundle_root} contains no version directories")
    ordered_names = [name for name in ("v1", "v2") if name in found]
    ordered_names.extend(sorted(name for name in found if name not in {"v1", "v2"}))
    return [(name, found[name]) for name in ordered_names]


def _copy_manifest_declared_artifacts(
    src_dir: Path,
    dst_dir: Path,
    manifest_data: dict[str, Any],
) -> None:
    artifacts = manifest_data.get("artifacts")
    if not isinstance(artifacts, dict):
        return
    src_root = src_dir.resolve()
    dst_root = dst_dir.resolve()
    for key in _MANIFEST_ARTIFACT_KEYS:
        raw_rel = artifacts.get(key)
        if not isinstance(raw_rel, str) or not raw_rel.strip():
            continue
        rel_path = Path(raw_rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"Unsafe manifest artifact path for '{key}': {raw_rel}")
        src_art = (src_root / rel_path).resolve()
        dst_art = (dst_root / rel_path).resolve()
        if not src_art.is_relative_to(src_root) or not dst_art.is_relative_to(dst_root):
            raise ValueError(f"Manifest artifact path escapes bundle root for '{key}': {raw_rel}")
        if src_art.is_file():
            dst_art.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_art, dst_art)
        elif src_art.is_dir():
            shutil.copytree(src_art, dst_art, dirs_exist_ok=True)


def _copy_version_artifacts(src_dir: Path, dst_dir: Path, manifest_data: dict[str, Any]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for filename in _SCENE_FILES:
        src_file = src_dir / filename
        if src_file.is_file():
            shutil.copy2(src_file, dst_dir / filename)
    for rel_dir in _SCENE_DIRS:
        src_sub = src_dir / rel_dir
        if src_sub.is_dir():
            shutil.copytree(src_sub, dst_dir / rel_dir, dirs_exist_ok=True)
    sparse_zero = src_dir / "sparse" / "0"
    if sparse_zero.is_dir():
        shutil.copytree(sparse_zero, dst_dir / "sparse" / "0", dirs_exist_ok=True)
    _copy_manifest_declared_artifacts(src_dir, dst_dir, manifest_data)


def _assert_core_artifacts(dst_dir: Path, raw_ver: str) -> None:
    has_manifest = (dst_dir / "manifest.json").is_file()
    has_geometry = (
        (dst_dir / "scene.spz").is_file()
        or (dst_dir / "sparse_points.json").is_file()
        or (dst_dir / "sparse" / "0" / "points3D.bin").is_file()
    )
    if not (has_manifest and has_geometry):
        raise ValueError(f"Imported version '{raw_ver}' is missing required core artifacts under {dst_dir}")


def _read_manifest_dict(version_dir: Path) -> dict[str, Any]:
    manifest_file = version_dir / "manifest.json"
    if not manifest_file.is_file():
        return {}
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _resolve_target_version_id(session: Any, project_id: str, raw_ver_id: str) -> str:
    if not raw_ver_id or ".." in raw_ver_id or not _SAFE_SEGMENT_RE.fullmatch(raw_ver_id):
        raise ValueError(f"Invalid version directory name '{raw_ver_id}'")
    if len(raw_ver_id) > 36:
        return _validate_safe_id(str(uuid.uuid4()), "version_id")
    existing = session.get(SceneVersion, raw_ver_id)
    if existing is None or existing.project_id == project_id:
        return _validate_safe_id(raw_ver_id, "version_id")
    return _validate_safe_id(str(uuid.uuid4()), "version_id")


def import_kaggle_bundle(
    bundle_path: Path,
    *,
    project_id: str | None = None,
    project_name: str | None = None,
    storage_dir: Path | None = None,
    db_url: str | None = None,
) -> dict[str, Any]:
    """Copy a Kaggle Room3D bundle (.zip or directory) into local storage and upsert Project/SceneVersion rows."""
    bundle = Path(bundle_path).resolve()
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle}")

    settings = get_settings()
    resolved_storage = Path(storage_dir).resolve() if storage_dir is not None else settings.storage_root
    resolved_db_url = db_url or settings.database_url

    with tempfile.TemporaryDirectory(prefix="room3d_import_") as tmp_dir:
        base_root = _safe_extract_zip(bundle, Path(tmp_dir)) if bundle.is_file() else bundle
        bundle_root = _resolve_bundle_root(base_root)
        version_dirs = _discover_version_dirs(bundle_root)

        first_manifest = _read_manifest_dict(version_dirs[0][1])
        fallback_stem = bundle.stem.removesuffix("_room3d_bundle") or "kaggle-room"
        resolved_project_id = _validate_safe_id(
            str(project_id or first_manifest.get("project_id") or fallback_stem),
            "project_id",
        )
        resolved_project_name = project_name or resolved_project_id.replace("-", " ").replace("_", " ").title()

        connect_args = {"check_same_thread": False} if resolved_db_url.startswith("sqlite") else {}
        engine = create_engine(resolved_db_url, connect_args=connect_args)
        try:
            Base.metadata.create_all(engine)
            session_factory = sessionmaker(engine, expire_on_commit=False)
            with session_factory() as session:
                proj = session.get(Project, resolved_project_id)
                if proj is None:
                    proj = Project(id=resolved_project_id, name=resolved_project_name, status="completed")
                    session.add(proj)
                else:
                    proj.name = project_name or proj.name
                    proj.status = "completed"

                version_map = {
                    raw_ver: _resolve_target_version_id(session, resolved_project_id, raw_ver)
                    for raw_ver, _ in version_dirs
                }
                imported_versions: list[str] = []
                for raw_ver, src_dir in version_dirs:
                    target_ver = _validate_safe_id(version_map[raw_ver], "version_id")
                    dst_dir = (resolved_storage / resolved_project_id / target_ver).resolve()
                    if not dst_dir.is_relative_to(resolved_storage):
                        raise ValueError(f"Destination escapes storage root: {dst_dir}")

                    manifest_data = _read_manifest_dict(src_dir)
                    _copy_version_artifacts(src_dir, dst_dir, manifest_data)

                    parent_raw = manifest_data.get("parent_version_id") or ("v1" if raw_ver == "v2" else None)
                    parent_ver = version_map.get(str(parent_raw), str(parent_raw)) if parent_raw else None
                    if manifest_data:
                        manifest_data["project_id"] = resolved_project_id
                        manifest_data["version_id"] = target_ver
                        if parent_ver is not None:
                            manifest_data["parent_version_id"] = parent_ver
                        (dst_dir / "manifest.json").write_text(
                            json.dumps(manifest_data, indent=2), encoding="utf-8"
                        )
                    _assert_core_artifacts(dst_dir, raw_ver)

                    sv = session.get(SceneVersion, target_ver)
                    if sv is None:
                        sv = SceneVersion(id=target_ver, project_id=resolved_project_id, label=raw_ver.upper())
                        session.add(sv)
                    sv.parent_version_id = parent_ver
                    sv.manifest_relpath = (
                        f"{resolved_project_id}/{target_ver}/manifest.json"
                        if (dst_dir / "manifest.json").is_file()
                        else None
                    )
                    sv.spz_relpath = (
                        f"{resolved_project_id}/{target_ver}/scene.spz"
                        if (dst_dir / "scene.spz").is_file()
                        else None
                    )
                    sv.ply_relpath = (
                        f"{resolved_project_id}/{target_ver}/scene.ply"
                        if (dst_dir / "scene.ply").is_file()
                        else None
                    )
                    collision_rel = (
                        manifest_data.get("artifacts", {}).get("collision_glb_relpath")
                        if isinstance(manifest_data.get("artifacts"), dict)
                        else None
                    )
                    if isinstance(collision_rel, str) and collision_rel and (dst_dir / collision_rel).is_file():
                        sv.collision_glb_relpath = f"{resolved_project_id}/{target_ver}/{collision_rel}"
                    metrics = manifest_data.get("metrics")
                    if isinstance(metrics, dict):
                        sv.metrics_json = json.dumps(metrics)
                    imported_versions.append(target_ver)

                active_raw = "v2" if "v2" in version_map else version_dirs[0][0]
                active_version_id = version_map[active_raw]
                proj.active_version_id = active_version_id
                session.commit()
        finally:
            engine.dispose()

    return {
        "project_id": resolved_project_id,
        "project_name": resolved_project_name,
        "active_version_id": active_version_id,
        "versions": imported_versions,
        "storage_dir": str(resolved_storage),
        "database_url": resolved_db_url,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import a Kaggle Room3D bundle (.zip or dir) into local Room3D")
    parser.add_argument("bundle_path", type=Path, help="Path to *_room3d_bundle.zip or extracted bundle directory")
    parser.add_argument("--project-id", default=None, help="Target project ID override")
    parser.add_argument("--project-name", default=None, help="Display name override")
    parser.add_argument("--storage-dir", type=Path, default=None, help="Override local storage root directory")
    parser.add_argument("--db-url", default=None, help="Override SQLAlchemy database URL")
    args = parser.parse_args(argv)
    summary = import_kaggle_bundle(
        args.bundle_path,
        project_id=args.project_id,
        project_name=args.project_name,
        storage_dir=args.storage_dir,
        db_url=args.db_url,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
