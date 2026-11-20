from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

from PIL import Image


def content_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def diffusion_identity(
    frames: Mapping[int, Path], masks: Mapping[int, Path], reference: Path,
    settings: Mapping[str, object],
) -> str:
    payload = {
        "schema": 1, "settings": settings, "reference": content_digest(reference),
        "frames": {str(idx): content_digest(path) for idx, path in sorted(frames.items())},
        "masks": {str(idx): content_digest(path) for idx, path in sorted(masks.items())},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class DiffusionCheckpoints:
    """Content-addressed completed outputs; a receipt is committed after the PNG."""

    def __init__(self, root: Path, identity: str) -> None:
        self.directory = root / identity
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity

    def restore(self, idx: int, destination: Path) -> bool:
        receipt = self.directory / f"{idx:06d}.json"
        image = receipt.with_suffix(".png")
        if not receipt.is_file() or not image.is_file():
            return False
        try:
            record = json.loads(receipt.read_text(encoding="utf-8"))
            if record != {"identity": self.identity, "frame_idx": idx, "sha256": content_digest(image)}:
                return False
            with Image.open(image) as source:
                source.verify()
        except (ValueError, OSError):
            return False
        shutil.copyfile(image, destination)
        return True

    def commit(self, idx: int, source: Path) -> None:
        image = self.directory / f"{idx:06d}.png"
        pending = image.with_suffix(".png.pending")
        shutil.copyfile(source, pending)
        os.replace(pending, image)
        record = {"identity": self.identity, "frame_idx": idx, "sha256": content_digest(image)}
        receipt = image.with_suffix(".json")
        pending = receipt.with_suffix(".json.pending")
        pending.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        os.replace(pending, receipt)


class CheckpointPublisher:
    def __init__(
        self, root: Path, prefix: str, put_many: Callable[[list[tuple[str, Path]]], None],
        published_root: Path | None = None,
    ) -> None:
        self.root, self.prefix, self.put_many = root, prefix, put_many
        self.uploaded: dict[Path, str] = {}
        if published_root:
            for receipt in published_root.glob("*/*.json"):
                local = root / receipt.relative_to(published_root)
                image = receipt.with_suffix(".png")
                try:
                    record = json.loads(receipt.read_text(encoding="utf-8"))
                except ValueError as exc:
                    logging.getLogger(__name__).warning("Ignoring invalid remote checkpoint receipt %s: %s", receipt, exc)
                    continue
                valid = isinstance(record, dict) and image.is_file() and record.get("sha256") == content_digest(image)
                if valid and local.is_file() and content_digest(local) == content_digest(receipt):
                    self.uploaded[local] = content_digest(receipt)

    def __call__(self) -> None:
        for receipt in self.root.glob("*/*.json"):
            image = receipt.with_suffix(".png")
            digest = content_digest(receipt)
            if self.uploaded.get(receipt) == digest or not image.is_file():
                continue
            try:
                record = json.loads(receipt.read_text(encoding="utf-8"))
            except ValueError as exc:
                logging.getLogger(__name__).warning("Ignoring incomplete checkpoint receipt %s: %s", receipt, exc)
                continue
            if not isinstance(record, dict):
                logging.getLogger(__name__).warning("Ignoring invalid checkpoint receipt %s", receipt)
                continue
            if record.get("sha256") != content_digest(image):
                continue
            files = [(f"{self.prefix}/{path.relative_to(self.root).as_posix()}", path) for path in (image, receipt)]
            self.put_many(files)
            self.uploaded[receipt] = digest
