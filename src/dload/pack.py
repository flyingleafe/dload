"""Shard pack format: many samples in one file, msgpack index in the footer.

Layout (all little-endian):

    [payload region: field blobs, concatenated in write order]
    [index: msgpack bytes]
    [u64: byte length of the msgpack index]
    [8-byte magic: b"DLOADPK1"]

The index is a msgpack-encoded dict:

    {
        "keys": [str, ...],                       # sample keys, in order
        "samples": [                              # parallel to "keys"
            {field_name: [offset, length], ...},  # offsets into payload region
            ...
        ],
    }

Design notes:
- Footer placement lets `PackWriter` stream payload bytes without knowing
  the total up front, and lets a reader grab the index with a single
  ranged read of the file tail.
- Files are immutable once finished and content-addressed by sha256 of the
  complete file; `PackWriter.finish()` computes the digest while writing so
  the file never needs re-reading.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import msgpack

from .errors import PackFormatError

MAGIC = b"DLOADPK1"
_FOOTER_SIZE = 16  # u64 index length + 8-byte magic

Sample = tuple[str, dict[str, bytes]]
"""A sample: (key, {field_name: payload_bytes})."""


class PackWriter:
    """Streams samples into a pack file.

    Usage:
        w = PackWriter(tmp_path)
        w.add("utt-0001", {"audio": wav, "meta": js})
        ...
        digest, size, n = w.finish()   # sha256 hex, file size, sample count

    `tell()` returns the current payload byte count so callers can cut
    shards at a target size. After `finish()` the writer is closed.
    `abort()` closes and deletes the partial file.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._file = self._path.open("wb")
        self._hash = hashlib.sha256()
        self._offset = 0
        self._keys: list[str] = []
        self._samples: list[dict[str, list[int]]] = []
        self._closed = False

    def _write(self, data: bytes) -> None:
        self._file.write(data)
        self._hash.update(data)

    def add(self, key: str, fields: dict[str, bytes]) -> None:
        if self._closed:
            raise ValueError("writer is closed")
        entry: dict[str, list[int]] = {}
        for name, data in fields.items():
            entry[name] = [self._offset, len(data)]
            self._write(data)
            self._offset += len(data)
        self._keys.append(key)
        self._samples.append(entry)

    def tell(self) -> int:
        return self._offset

    @property
    def num_samples(self) -> int:
        return len(self._keys)

    def finish(self) -> tuple[str, int, int]:
        if self._closed:
            raise ValueError("writer is closed")
        index = {"keys": self._keys, "samples": self._samples}
        # msgpack's stub types packb() as returning `bytes | None` to cover
        # the `stream=` overload; without one it always returns bytes.
        index_bytes = cast(bytes, msgpack.packb(index, use_bin_type=True))
        self._write(index_bytes)
        self._write(struct.pack("<Q", len(index_bytes)))
        self._write(MAGIC)
        digest = self._hash.hexdigest()
        size = self._file.tell()
        self._file.close()
        self._closed = True
        return digest, size, len(self._keys)

    def abort(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True
        self._path.unlink(missing_ok=True)


class PackReader:
    """Reads a pack file; loads only the index eagerly, payload lazily (mmap
    or seek+read — implementation's choice, must be cheap for sequential
    full-shard iteration AND for single-sample access).

    Usage:
        with PackReader(path) as r:
            len(r)                      # sample count
            r.keys                      # list[str]
            r.read(i) -> Sample         # by position
            r.read_key(key) -> Sample   # by key (raises KeyError)
            for key, fields in r: ...   # in stored order

    Raises PackFormatError on bad magic / truncated / corrupt index.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._file = self._path.open("rb")
        try:
            self._keys, self._samples = self._read_index()
        except Exception:
            self._file.close()
            raise
        self._key_to_index = {k: i for i, k in enumerate(self._keys)}

    def _read_index(self) -> tuple[list[str], list[dict[str, list[int]]]]:
        size = self._file.seek(0, 2)
        if size < _FOOTER_SIZE:
            raise PackFormatError(
                f"pack file too small ({size} bytes) to contain a footer"
            )
        self._file.seek(size - _FOOTER_SIZE)
        footer = self._file.read(_FOOTER_SIZE)
        index_len_bytes, magic = footer[:8], footer[8:]
        if magic != MAGIC:
            raise PackFormatError(f"bad magic {magic!r}, expected {MAGIC!r}")
        (index_len,) = struct.unpack("<Q", index_len_bytes)
        index_start = size - _FOOTER_SIZE - index_len
        if index_len < 0 or index_start < 0:
            raise PackFormatError(
                f"corrupt index length {index_len} for file of size {size}"
            )
        self._file.seek(index_start)
        index_bytes = self._file.read(index_len)
        try:
            index = msgpack.unpackb(index_bytes, raw=False, strict_map_key=False)
        except Exception as e:
            raise PackFormatError(f"corrupt msgpack index: {e}") from e
        if (
            not isinstance(index, dict)
            or not isinstance(index.get("keys"), list)
            or not isinstance(index.get("samples"), list)
            or len(index["keys"]) != len(index["samples"])
        ):
            raise PackFormatError("malformed pack index")
        self._payload_end = index_start
        return index["keys"], index["samples"]

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> list[str]:
        return list(self._keys)

    def read(self, index: int) -> Sample:
        key = self._keys[index]
        fields = {}
        for name, (offset, length) in self._samples[index].items():
            self._file.seek(offset)
            fields[name] = self._file.read(length)
        return key, fields

    def read_key(self, key: str) -> Sample:
        try:
            index = self._key_to_index[key]
        except KeyError:
            raise KeyError(key) from None
        return self.read(index)

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self.read(i)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "PackReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
