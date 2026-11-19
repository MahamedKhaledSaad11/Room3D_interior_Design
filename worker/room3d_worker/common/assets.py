from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field, field_validator


def safe_asset_path(value: str) -> str:
    if not value or "\\" in value or "\x00" in value or ":" in value:
        raise ValueError(f"Invalid relative asset path: {value!r}")
    if any(part in {"", ".", ".."} or part.startswith(".") for part in value.split("/")):
        raise ValueError(f"Unsafe asset path: {value!r}")
    return value


class AssetRevisionError(RuntimeError):
    """The remote assets changed between inventory and archive download."""


class AssetInfo(BaseModel):
    path: str
    size: int = Field(ge=0)
    etag: str

    @field_validator("path")
    @classmethod
    def valid_path(cls, value: str) -> str:
        return safe_asset_path(value)


def asset_revision(objects: list[AssetInfo]) -> str:
    payload = json.dumps([item.model_dump() for item in sorted(objects, key=lambda item: item.path)],
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
