from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import BinaryIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

from .base import StorageDriver, StorageObject, validate_key

MAX_BUNDLE_BYTES = 8_000_000_000
MAX_BUNDLE_FILES = 20_000


def bundle_path(path: str) -> str:
    if ":" in path:
        raise ValueError("Bundle paths cannot contain drive or stream separators")
    clean = validate_key(path)
    if any(part.startswith(".") for part in clean.split("/")):
        raise ValueError("Bundle paths cannot include internal storage directories")
    return clean


def canonical_path(path: str) -> str:
    match = re.fullmatch(r"(depths|masks|edited_frames)/(?:depth_)?(\d{5})\.(npy|png)", path)
    return f"{match[1]}/{int(match[2]):06d}.{match[3]}" if match else path


def bundle_objects(storage: StorageDriver, prefix: str, paths: list[str]) -> dict[str, StorageObject]:
    selected = [bundle_path(path) for path in paths]
    result: dict[str, StorageObject] = {}
    for item in storage.list_objects(prefix):
        relative = item.key.removeprefix(prefix + "/")
        if any(part.startswith(".") for part in relative.split("/")):
            continue
        name = canonical_path(relative)
        if selected and not any(name == path or name.startswith(path + "/") for path in selected):
            continue
        if name not in result or relative == name:
            result[name] = item
    return dict(sorted(result.items()))


def object_payload(objects: dict[str, StorageObject]) -> list[dict[str, str | int]]:
    return [{"path": path, "size": item.size, "etag": item.etag} for path, item in objects.items()]


def object_revision(objects: dict[str, StorageObject]) -> str:
    payload = json.dumps(object_payload(objects), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_bundle(storage: StorageDriver, objects: dict[str, StorageObject], output: BinaryIO) -> None:
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for name, item in objects.items():
            with storage.open_read(item.key) as source, archive.open(name, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
    output.seek(0)


def _members(archive: ZipFile) -> list[str]:
    entries = archive.infolist()
    names: list[str] = []
    seen: set[str] = set()
    if len(entries) > MAX_BUNDLE_FILES or sum(item.file_size for item in entries) > MAX_BUNDLE_BYTES:
        raise ValueError("Bundle exceeds the file count or expanded size limit")
    for item in entries:
        name = bundle_path(item.filename.rstrip("/"))
        if item.is_dir():
            continue
        if name != item.filename or stat.S_ISLNK(item.external_attr >> 16) or item.flag_bits & 1:
            raise ValueError("Bundle contains a non-canonical, linked, or encrypted entry")
        if item.compress_type not in {ZIP_STORED, ZIP_DEFLATED} or name in seen:
            raise ValueError("Bundle contains duplicate entries or an unsupported compression method")
        names.append(name)
        seen.add(name)
    if any(any("/".join(name.split("/")[:i]) in seen for i in range(1, len(name.split("/")))) for name in names):
        raise ValueError("A bundle file cannot also be a parent directory")
    if not names:
        raise ValueError("Bundle contains no files")
    return names


def import_bundle(storage: StorageDriver, prefix: str, source: BinaryIO) -> int:
    # Validate CRCs and every path before publishing any files. Retry after a lost acknowledgement is safe.
    try:
        with ZipFile(source) as archive, tempfile.TemporaryDirectory(prefix="room3d-bundle-") as staging:
            names = _members(archive)
            for name in names:
                target = Path(staging) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as reader, target.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, 1024 * 1024)
            for name in sorted(names, key=lambda name: name == "manifest.json"):
                with (Path(staging) / name).open("rb") as stream:
                    storage.put_stream(f"{prefix}/{name}", stream)
            return len(names)
    except (BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise ValueError(f"Invalid ZIP bundle: {exc}") from exc
