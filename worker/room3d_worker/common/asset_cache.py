from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

from room3d_worker.common.assets import AssetInfo, AssetRevisionError, safe_asset_path

MAX_CACHE_BYTES = 3_000_000_000


class DiskAssetCache:
    """Revision-aware scene LRU shared by the daemon's batch and warm SAM threads."""

    _lock = threading.RLock()
    _pins: ClassVar[dict[Path, int]] = {}
    _scene_locks: ClassVar[dict[Path, threading.RLock]] = {}

    def __init__(self, root: Path, max_bytes: int = MAX_CACHE_BYTES) -> None:
        if max_bytes <= 0:
            raise ValueError("Cache size limit must be positive")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        with self._lock:
            self.cleanup()

    def entry(self, prefix: str) -> Path:
        clean = safe_asset_path(prefix)
        parts = clean.split("/")
        if len(parts) != (3 if len(parts) > 1 and parts[1] == "edits" else 2):
            raise ValueError(f"Invalid cache scope: {prefix}")
        candidate = (self.root / clean).resolve()
        if self.root not in candidate.parents:
            raise ValueError("Cache scope escapes cache root")
        return candidate

    def _entries(self) -> list[Path]:
        entries = list(self.root.glob("*/*"))
        return [entry for entry in entries if entry.is_dir() and entry.name != "edits"] + [
            entry for entry in self.root.glob("*/edits/*") if entry.is_dir()
        ]

    @contextmanager
    def _lease(self, entry: Path) -> Iterator[None]:
        with self._lock:
            self._pins[entry] = self._pins.get(entry, 0) + 1
            scene_lock = self._scene_locks.setdefault(entry, threading.RLock())
        try:
            with scene_lock:
                yield
        finally:
            with self._lock:
                self._pins[entry] -= 1
                if self._pins[entry] == 0:
                    self._pins.pop(entry)

    @staticmethod
    def _size(entry: Path) -> int:
        return sum(path.stat().st_size for path in entry.rglob("*") if path.is_file())

    def cleanup(self, incoming: int = 0, protect: Path | None = None) -> bool:
        with self._lock:
            if incoming > self.max_bytes:
                return False
            entries = self._entries()
            total = sum(self._size(entry) for entry in entries)
            for entry in sorted(entries, key=lambda path: path.stat().st_mtime_ns):
                if total + incoming <= self.max_bytes:
                    break
                if entry == protect or self._pins.get(entry, 0):
                    continue
                if self.root not in entry.resolve().parents or entry.is_symlink():
                    raise ValueError("Refusing to evict a cache path outside the cache root")
                total -= self._size(entry)
                shutil.rmtree(entry)
            return total + incoming <= self.max_bytes

    @staticmethod
    def _records(entry: Path) -> dict[str, str]:
        metadata = entry / ".cache.json"
        if not metadata.exists():
            return {}
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in payload.items()):
                raise ValueError("Invalid cache metadata schema")
            for name in payload:
                safe_asset_path(name)
            return payload
        except (ValueError, json.JSONDecodeError) as exc:
            logging.getLogger(__name__).warning("Revalidating corrupt cache metadata %s: %s", metadata, exc)
            return {}

    @staticmethod
    def _save_records(entry: Path, records: dict[str, str]) -> None:
        pending = entry / ".cache.pending"
        pending.write_text(json.dumps(records, sort_keys=True), encoding="utf-8")
        pending.replace(entry / ".cache.json")
        os.utime(entry, None)

    def copy(self, key: str, destination: Path) -> bool:
        parts = safe_asset_path(key).split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid cached asset key: {key}")
        count = 3 if parts[1] == "edits" else 2
        entry = self.entry("/".join(parts[:count]))
        with self._lock:
            source = entry / "/".join(parts[count:])
            if not source.is_file():
                return False
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copyfile(source, destination)
            os.utime(entry, None)
            return True

    def sync(
        self, prefix: str, objects: list[AssetInfo], destination: Path,
        downloader: Callable[[list[str], Path], None], requested: list[str],
    ) -> None:
        entry = self.entry(prefix)
        # Network reads hold only a scene lease, so another scene's warm SAM lane can use its cache.
        with self._lease(entry):
            records = self._records(entry)
            self._discard_missing(entry, objects, destination, requested, records)
            if entry.exists():
                self._save_records(entry, records)
            changed = [item for item in objects if records.get(item.path) != item.etag
                       or not (entry / item.path).is_file() or (entry / item.path).stat().st_size != item.size]
            if changed:
                self._fill(prefix, objects, destination, downloader, requested, changed)
            else:
                for item in objects:
                    self._copy(entry / item.path, destination / item.path)
                if entry.exists():
                    os.utime(entry, None)

    def _discard_missing(
        self, entry: Path, objects: list[AssetInfo], destination: Path, requested: list[str], records: dict[str, str],
    ) -> None:
        present = {item.path for item in objects}
        for name in list(records):
            if name not in present and any(name == path or name.startswith(path + "/") for path in requested):
                (entry / name).unlink(missing_ok=True)
                records.pop(name)
        for path in requested:
            target = destination / safe_asset_path(path)
            existing = [target] if target.is_file() else target.rglob("*")
            for file in existing:
                if file.is_file() and file.relative_to(destination).as_posix() not in present:
                    file.unlink()

    def _fill(
        self, prefix: str, objects: list[AssetInfo], destination: Path, downloader: Callable[[list[str], Path], None],
        requested: list[str], changed: list[AssetInfo],
    ) -> None:
        groups = sorted({next(path for path in requested if item.path == path or item.path.startswith(path + "/")) for item in changed})
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="room3d-cache-fill-", dir=destination.parent) as staging:
            downloaded = Path(staging)
            downloader(groups, downloaded)
            fresh = [item for item in objects if (downloaded / item.path).is_file()]
            if any((downloaded / item.path).stat().st_size != item.size for item in fresh):
                raise AssetRevisionError("Asset changed during bundle transfer; retry the task")
            if any(not (downloaded / item.path).is_file() for item in changed):
                raise ValueError("Downloaded bundle is missing an indexed asset")
            self.store(prefix, [(item.path, downloaded / item.path) for item in fresh], fresh)
            for item in objects:
                source = downloaded / item.path if (downloaded / item.path).is_file() else self.entry(prefix) / item.path
                self._copy(source, destination / item.path)

    @staticmethod
    def _copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)

    def store(self, prefix: str, files: list[tuple[str, Path]], objects: list[AssetInfo]) -> None:
        with self._lock:
            entry = self.entry(prefix)
            incoming = sum(path.stat().st_size for _, path in files) + sum(len(item.path) + len(item.etag) + 8 for item in objects) + 128
            if not self.cleanup(incoming, protect=entry):
                return
            entry.mkdir(parents=True, exist_ok=True)
            records = self._records(entry)
            etags = {item.path: item.etag for item in objects}
            for name, source in files:
                safe_asset_path(name)
                self._copy(source, entry / name)
                if name in etags:
                    records[name] = etags[name]
            self._save_records(entry, records)
            self.cleanup(protect=entry)
