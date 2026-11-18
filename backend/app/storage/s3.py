from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from .base import StorageDriver, StorageObject, validate_key


class _CountingReader:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self.total = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size)
        self.total += len(chunk)
        return chunk


class S3Storage(StorageDriver):
    def __init__(self, bucket: str, **kwargs: object) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("S3 storage requires optional boto3 dependency") from exc
        self.client = boto3.client("s3", **kwargs)
        self.bucket = bucket

    def safe_resolve(self, key: str) -> Path:
        return Path(validate_key(key))

    def list_objects(self, prefix: str) -> list[StorageObject]:
        normalized = validate_key(prefix).rstrip("/") + "/"
        result: list[StorageObject] = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=normalized):
            for item in page.get("Contents", []):
                if not item["Key"].endswith("/"):
                    result.append(StorageObject(item["Key"], item["Size"], item["ETag"].strip('"')))
        return sorted(result, key=lambda item: item.key)

    def put_stream(self, key: str, stream: BinaryIO, chunk_size: int = 1024 * 1024) -> int:
        del chunk_size
        counter = _CountingReader(stream)
        self.client.upload_fileobj(counter, self.bucket, validate_key(key))
        return counter.total

    def open_read(self, key: str) -> BinaryIO:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=validate_key(key))["Body"]
        except self.client.exceptions.NoSuchKey as exc:
            raise FileNotFoundError(key) from exc

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=validate_key(key))
        except self.client.exceptions.ClientError:
            return False
        return True

    def delete_prefix(self, prefix: str) -> int:
        normalized = validate_key(prefix)
        count = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=normalized):
            objects = page.get("Contents", [])
            if objects:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": [{"Key": x["Key"]} for x in objects]})
                count += len(objects)
        return count
