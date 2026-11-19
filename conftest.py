from __future__ import annotations

import os
import tempfile
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

# App imports create tables. Keep collection away from the developer's saved projects.
_IMPORT_DATABASE = Path(tempfile.mkdtemp(prefix="room3d_pytest_")) / "collection.sqlite"
os.environ["DATABASE_URL"] = f"sqlite:///{_IMPORT_DATABASE.as_posix()}"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
_REPOSITORY = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = os.pathsep.join([str(_REPOSITORY / "backend"), str(_REPOSITORY / "worker"), os.environ.get("PYTHONPATH", "")])


@pytest.fixture(autouse=True)
def isolated_settings_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("WORKER_TOKEN", "dev-worker-token")
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    config = sys.modules.get("app.config")
    if config is not None:
        config.get_settings.cache_clear()
    yield
    if config is not None:
        config.get_settings.cache_clear()
