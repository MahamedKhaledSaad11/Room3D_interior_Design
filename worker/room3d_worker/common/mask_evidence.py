from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image


def mask_evidence(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        pixels = image.tobytes()
        area = pixels.count(255)
    return {
        "area_px": area,
        "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "visibility": "visible" if area else "unresolved",
        "rejection_reason": None if area else "empty_mask",
        "confidence": None,
        "reviewed": False,
    }
