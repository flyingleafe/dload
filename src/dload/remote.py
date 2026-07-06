"""Storage backends: the `Remote` protocol, S3/R2 implementation, and a
filesystem-backed implementation for tests.

Keys are forward-slash paths relative to the configured prefix (the
implementations prepend the prefix). All methods raise
dload.errors.RemoteError (NotFoundError for missing keys).
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import boto3
from botocore.exceptions import ClientError

from .errors import NotFoundError, RemoteError


@runtime_checkable
class Remote(Protocol):
    def get_bytes(self, key: str) -> bytes: ...

    def get_to_file(self, key: str, dest: Path) -> None:
        """Download to `dest` (parent dirs created; write to a temp sibling
        then atomic-rename so readers never observe partial files)."""
        ...

    def put_bytes(self, key: str, data: bytes) -> None: ...

    def put_file(self, key: str, src: Path) -> None: ...

    def exists(self, key: str) -> bool: ...

    def size(self, key: str) -> int: ...

    def list(self, prefix: str = "") -> Iterator[str]:
        """All keys under `prefix` (relative keys, prefix included)."""
        ...

    def delete(self, key: str) -> None:
        """Idempotent: deleting a missing key is not an error."""
        ...


def _atomic_write(dest: Path, write: Callable[[Path], object]) -> None:
    """Run `write` against a temp sibling of `dest`, then atomic-rename it
    into place so readers never observe partial files; the temp file is
    removed on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.tmp-{uuid.uuid4().hex}"
    try:
        write(tmp)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class S3Remote:
    """boto3-backed remote for any S3-compatible store (Cloudflare R2).

    - Client created lazily and per-instance config: endpoint_url, bucket,
      prefix. Credentials/region come from boto3's standard chain.
    - Uses boto3 transfer manager (upload_file/download_file) for files so
      multipart kicks in for large shards.
    - Translates ClientError 404/NoSuchKey into NotFoundError, everything
      else into RemoteError (chaining the original).
    - `list` paginates.
    - Thread-safe for concurrent gets (boto3 clients are thread-safe).
    """

    def __init__(self, endpoint_url: str, bucket: str, prefix: str = "") -> None:
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        stripped = prefix.strip("/")
        self.prefix = f"{stripped}/" if stripped else ""
        self._client = None

    @property
    def _s3(self):
        if self._client is None:
            self._client = boto3.client("s3", endpoint_url=self.endpoint_url)
        return self._client

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    @staticmethod
    def _translate(e: ClientError) -> RemoteError:
        error = e.response.get("Error", {})
        code = error.get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in ("404", "NoSuchKey", "NotFound") or status == 404:
            return NotFoundError(str(e))
        return RemoteError(str(e))

    def get_bytes(self, key: str) -> bytes:
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=self._full_key(key))
            return resp["Body"].read()
        except ClientError as e:
            raise self._translate(e) from e

    def get_to_file(self, key: str, dest: Path) -> None:
        def download(tmp: Path) -> None:
            try:
                self._s3.download_file(self.bucket, self._full_key(key), str(tmp))
            except ClientError as e:
                raise self._translate(e) from e

        _atomic_write(dest, download)

    def put_bytes(self, key: str, data: bytes) -> None:
        try:
            self._s3.put_object(Bucket=self.bucket, Key=self._full_key(key), Body=data)
        except ClientError as e:
            raise self._translate(e) from e

    def put_file(self, key: str, src: Path) -> None:
        try:
            self._s3.upload_file(str(src), self.bucket, self._full_key(key))
        except ClientError as e:
            raise self._translate(e) from e

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._full_key(key))
            return True
        except ClientError as e:
            translated = self._translate(e)
            if isinstance(translated, NotFoundError):
                return False
            raise translated from e

    def size(self, key: str) -> int:
        try:
            resp = self._s3.head_object(Bucket=self.bucket, Key=self._full_key(key))
            return resp["ContentLength"]
        except ClientError as e:
            raise self._translate(e) from e

    def list(self, prefix: str = "") -> Iterator[str]:
        full_prefix = self._full_key(prefix)
        paginator = self._s3.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                for obj in page.get("Contents", []):
                    yield obj["Key"][len(self.prefix) :]
        except ClientError as e:
            raise self._translate(e) from e

    def delete(self, key: str) -> None:
        try:
            self._s3.delete_object(Bucket=self.bucket, Key=self._full_key(key))
        except ClientError as e:
            raise self._translate(e) from e


class LocalRemote:
    """Directory-backed remote with identical semantics; for tests and for
    'remote on a network mount' setups. Also counts operations
    (self.op_counts: dict[str, int]) so tests can assert read-efficiency.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.op_counts: dict[str, int] = {}

    def _count(self, name: str) -> None:
        self.op_counts[name] = self.op_counts.get(name, 0) + 1

    def _path(self, key: str) -> Path:
        return self.root / key

    def get_bytes(self, key: str) -> bytes:
        self._count("get_bytes")
        p = self._path(key)
        if not p.is_file():
            raise NotFoundError(f"key not found: {key}")
        return p.read_bytes()

    def get_to_file(self, key: str, dest: Path) -> None:
        self._count("get_to_file")
        p = self._path(key)
        if not p.is_file():
            raise NotFoundError(f"key not found: {key}")
        _atomic_write(dest, lambda tmp: shutil.copyfile(p, tmp))

    def put_bytes(self, key: str, data: bytes) -> None:
        self._count("put_bytes")
        _atomic_write(self._path(key), lambda tmp: tmp.write_bytes(data))

    def put_file(self, key: str, src: Path) -> None:
        self._count("put_file")
        _atomic_write(self._path(key), lambda tmp: shutil.copyfile(src, tmp))

    def exists(self, key: str) -> bool:
        self._count("exists")
        return self._path(key).is_file()

    def size(self, key: str) -> int:
        self._count("size")
        p = self._path(key)
        if not p.is_file():
            raise NotFoundError(f"key not found: {key}")
        return p.stat().st_size

    def list(self, prefix: str = "") -> Iterator[str]:
        self._count("list")
        keys = [
            p.relative_to(self.root).as_posix()
            for p in sorted(self.root.rglob("*"))
            if p.is_file() and p.relative_to(self.root).as_posix().startswith(prefix)
        ]
        return iter(keys)

    def delete(self, key: str) -> None:
        self._count("delete")
        self._path(key).unlink(missing_ok=True)
