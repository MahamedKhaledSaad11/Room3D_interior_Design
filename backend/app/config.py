from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_DATA_DIR = _REPO_ROOT / ".local_data"


class Settings(BaseSettings):
    database_url: str = f"sqlite:///{(_LOCAL_DATA_DIR / 'room3d.db').as_posix()}"
    storage_backend: str = "local"
    local_storage_dir: str = str(_LOCAL_DATA_DIR / "storage")
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_region: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    worker_token: str = "dev-worker-token"
    jwt_secret: str = Field(default="dev-jwt-secret-change-me", min_length=16)
    auth_required: bool = False
    max_upload_bytes: int = 500 * 1024 * 1024
    task_lease_timeout_sec: int = 180
    queue_backend: Literal["db_pull", "celery_redis"] = "db_pull"
    redis_url: str | None = None
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _accept_room3d_env_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        merged = dict(data)
        for field_name in ("database_url", "local_storage_dir", "auth_required", "worker_token", "jwt_secret"):
            if field_name not in merged and field_name.upper() not in os.environ:
                prefixed = os.environ.get(f"ROOM3D_{field_name.upper()}")
                if prefixed is not None:
                    merged[field_name] = prefixed
        return merged

    @property
    def storage_root(self) -> Path:
        return Path(self.local_storage_dir).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
