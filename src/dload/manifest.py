"""Dataset manifests: the versioned description of a dataset.

A manifest is a JSON document; its *version id* is the sha256 hex digest of
its canonical serialization (sorted keys, no whitespace, "version" field
excluded). Manifests are immutable: any change produces a new version id.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from .errors import PackFormatError

FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ShardInfo:
    digest: str  # sha256 hex of the pack file
    size: int  # bytes
    num_samples: int


@dataclass(frozen=True, slots=True)
class Manifest:
    name: str
    created: str  # ISO-8601 UTC, e.g. "2026-07-03T12:00:00Z"
    shards: tuple[ShardInfo, ...]
    meta: dict = field(default_factory=dict)  # free-form user metadata
    recipe: str | None = None  # original download script, verbatim
    format: int = FORMAT_VERSION

    @property
    def num_samples(self) -> int:
        return sum(s.num_samples for s in self.shards)

    @property
    def total_bytes(self) -> int:
        return sum(s.size for s in self.shards)

    def _canonical_dict(self) -> dict:
        return asdict(self)

    @property
    def version(self) -> str:
        """sha256 hex of canonical JSON (excluding any version field)."""
        canonical = json.dumps(
            self._canonical_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """Human-friendly JSON (indented, sorted keys), includes "version"
        as a convenience for people reading the file; parsers ignore it."""
        data = self._canonical_dict()
        data["version"] = self.version
        return json.dumps(data, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        """Parse; tolerant of the redundant "version" key. Raises
        dload.errors.PackFormatError on unsupported "format"."""
        data = json.loads(text)
        data.pop("version", None)
        fmt = data.get("format", FORMAT_VERSION)
        if fmt != FORMAT_VERSION:
            raise PackFormatError(
                f"unsupported manifest format {fmt!r}, expected {FORMAT_VERSION}"
            )
        shards = tuple(ShardInfo(**s) for s in data["shards"])
        return cls(
            name=data["name"],
            created=data["created"],
            shards=shards,
            meta=data.get("meta", {}),
            recipe=data.get("recipe"),
            format=fmt,
        )
