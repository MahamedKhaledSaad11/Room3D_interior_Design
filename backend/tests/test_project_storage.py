from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from app.storage import S3Storage


@pytest.mark.parametrize("fails", [False, True])
def test_project_delete_respects_s3_namespace_and_reports_errors(monkeypatch: pytest.MonkeyPatch, fails: bool) -> None:
    deleted: list[str] = []

    def paginate(**kwargs: object) -> list[dict[str, list[dict[str, str]]]]:
        assert kwargs["Prefix"] == "room"
        return [{"Contents": [{"Key": "room/video.mp4"}, {"Key": "room-two/video.mp4"}]}]

    def delete_objects(*, Bucket: str, Delete: dict[str, list[dict[str, str]]]) -> dict[str, object]:
        deleted.extend(item["Key"] for item in Delete["Objects"])
        return {"Errors": [{"Code": "AccessDenied"}]} if fails else {}

    client = SimpleNamespace(get_paginator=lambda name: SimpleNamespace(paginate=paginate), delete_objects=delete_objects)
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *args, **kwargs: client))
    storage = S3Storage("test-bucket")
    if fails:
        with pytest.raises(OSError, match="AccessDenied"):
            storage.delete_prefix("room")
    else:
        assert storage.delete_prefix("room") == 1
    assert deleted == ["room/video.mp4"]
