from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache
def task_aliases() -> dict[str, str]:
    path = Path(__file__).resolve().parents[3] / "contracts" / "enums.json"
    return dict(json.loads(path.read_text(encoding="utf-8"))["task_aliases"])


def canonical_task_type(name: str) -> str:
    return task_aliases().get(name, name)


def expand_capabilities(capabilities: list[str]) -> list[str]:
    supported = {canonical_task_type(name) for name in capabilities}
    return sorted(set(capabilities) | supported | {
        alias for alias, canonical in task_aliases().items() if canonical in supported
    })
