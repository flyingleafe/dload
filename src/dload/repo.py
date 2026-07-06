"""Repository: the bridge between the remote store and the local cache.

Bucket layout (under the configured prefix):

    datasets/<name>/manifests/<version>.json
    datasets/<name>/refs/latest
    shards/<aa>/<digest>

Version resolution for reads: explicit argument → `dload.lock` in the
project directory → `refs/latest`.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from typing import TYPE_CHECKING

from .cache import PinnedShard, ShardCache
from .config import Config, format_size
from .errors import CacheFullError, ConfigError, NotFoundError
from .manifest import Manifest, ShardInfo
from .pack import PackWriter, Sample
from .remote import Remote, S3Remote

if TYPE_CHECKING:
    from .pipeline import Pipeline

DEFAULT_SHARD_SIZE = 128 * 1024 * 1024
_FULL_DIGEST_LEN = 64

Progress = Callable[[str], None]


def _shard_key(digest: str) -> str:
    return f"shards/{digest[:2]}/{digest}"


def _manifest_key(name: str, version: str) -> str:
    return f"datasets/{name}/manifests/{version}.json"


def _ref_key(name: str) -> str:
    return f"datasets/{name}/refs/latest"


class Repository:
    """A dataset store = remote (source of truth) + local shard cache."""

    def __init__(
        self,
        remote: Remote,
        cache: ShardCache,
        lock_path: Path | None = None,
    ) -> None:
        self.remote = remote
        self.cache = cache
        self.lock_path = lock_path or Path.cwd() / "dload.lock"

    @classmethod
    def open(cls, config: Config | None = None) -> "Repository":
        """Build a Repository from the layered machine configuration."""
        cfg = config or Config.load()
        endpoint, bucket = cfg.require_remote()
        remote = S3Remote(endpoint, bucket, cfg.prefix)
        cache = ShardCache(cfg.cache_dir, cfg.cache_budget)
        return cls(remote, cache)

    # -- ingest ------------------------------------------------------------

    def commit(
        self,
        name: str,
        samples: Iterable[Sample],
        *,
        meta: dict | None = None,
        recipe: str | None = None,
        target_shard_size: int = DEFAULT_SHARD_SIZE,
        progress: Progress | None = None,
    ) -> Manifest:
        """Pack `samples` into shards, upload anything the remote is missing,
        write a new manifest and point `refs/latest` at it.

        Shards are content-addressed, so re-committing identical data is
        cheap (nothing re-uploads) and versions share storage. Freshly
        packed shards are also dropped into the local cache (best effort)
        so training right after committing is warm.
        """
        say = progress or (lambda _msg: None)
        if "/" in name or not name:
            raise ConfigError(f"invalid dataset name {name!r}")
        shard_infos: list[ShardInfo] = []
        writer: PackWriter | None = None
        writer_path: Path | None = None

        def cut() -> None:
            nonlocal writer, writer_path
            assert writer is not None and writer_path is not None
            digest, size, count = writer.finish()
            info = ShardInfo(digest=digest, size=size, num_samples=count)
            key = _shard_key(digest)
            if self.remote.exists(key):
                say(
                    f"shard {len(shard_infos)}: {format_size(size)}, {count} samples — already in remote"
                )
            else:
                self.remote.put_file(key, writer_path)
                say(
                    f"shard {len(shard_infos)}: {format_size(size)}, {count} samples — uploaded {digest[:12]}"
                )
            self._adopt_into_cache(digest, size, writer_path)
            shard_infos.append(info)
            writer = writer_path = None

        try:
            for key, fields in samples:
                if writer is None:
                    fd, tmp = tempfile.mkstemp(prefix="pack-", dir=str(self._tmp_dir()))
                    os.close(fd)
                    writer_path = Path(tmp)
                    writer = PackWriter(writer_path)
                writer.add(key, fields)
                if writer.tell() >= target_shard_size:
                    cut()
            if writer is not None and writer.num_samples:
                cut()
        except BaseException:
            if writer is not None:
                writer.abort()
            raise

        manifest = Manifest(
            name=name,
            created=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            shards=tuple(shard_infos),
            meta=meta or {},
            recipe=recipe,
        )
        version = manifest.version
        self.remote.put_bytes(_manifest_key(name, version), manifest.to_json().encode())
        self.remote.put_bytes(_ref_key(name), version.encode())
        say(
            f"committed {name}@{version[:12]}: {manifest.num_samples} samples, "
            f"{len(shard_infos)} shards, {format_size(manifest.total_bytes)}"
        )
        return manifest

    def _tmp_dir(self) -> Path:
        d = self.cache.root / "tmp"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _adopt_into_cache(self, digest: str, size: int, src: Path) -> None:
        try:
            self.cache.ensure(
                digest, size, lambda dest: os.replace(src, dest)
            ).release()
        except CacheFullError:
            pass
        finally:
            src.unlink(missing_ok=True)

    # -- resolution & reads --------------------------------------------------

    def resolve(self, name: str, version: str | None = None) -> str:
        """Resolve to a full version id: explicit (prefixes allowed) →
        dload.lock → refs/latest."""
        if version is not None:
            if len(version) == _FULL_DIGEST_LEN:
                return version
            matches = [v for v in self._manifest_digests(name) if v.startswith(version)]
            if not matches:
                raise NotFoundError(f"{name}: no version matching {version!r}")
            if len(matches) > 1:
                raise NotFoundError(f"{name}: version prefix {version!r} is ambiguous")
            return matches[0]
        pinned = self._read_lock().get(name)
        if pinned is not None:
            return pinned
        try:
            return self.remote.get_bytes(_ref_key(name)).decode().strip()
        except NotFoundError:
            raise NotFoundError(f"dataset {name!r} not found in remote") from None

    def manifest(self, name: str, version: str | None = None) -> Manifest:
        resolved = self.resolve(name, version)
        return Manifest.from_json(
            self.remote.get_bytes(_manifest_key(name, resolved)).decode()
        )

    def dataset(self, name: str, version: str | None = None) -> "Dataset":
        return Dataset(self, self.manifest(name, version))

    def list_datasets(self) -> list[str]:
        names = {
            key.split("/")[1]
            for key in self.remote.list("datasets/")
            if key.count("/") >= 2
        }
        return sorted(names)

    def versions(self, name: str) -> list[Manifest]:
        """All manifests of a dataset, newest first."""
        digests = self._manifest_digests(name)
        if not digests:
            raise NotFoundError(f"dataset {name!r} not found in remote")
        return self._load_manifests(name, digests)

    def _load_manifests(self, name: str, digests: list[str]) -> list[Manifest]:
        manifests = [
            Manifest.from_json(self.remote.get_bytes(_manifest_key(name, d)).decode())
            for d in digests
        ]
        return sorted(manifests, key=lambda m: m.created, reverse=True)

    def _manifest_digests(self, name: str) -> list[str]:
        prefix = f"datasets/{name}/manifests/"
        return [
            key.removeprefix(prefix).removesuffix(".json")
            for key in self.remote.list(prefix)
        ]

    # -- pinning (dload.lock) --------------------------------------------------

    def pin(self, name: str, version: str | None = None) -> str:
        """Pin `name` in dload.lock to `version` (default: current
        refs/latest, deliberately bypassing an existing pin)."""
        if version is not None:
            resolved = self.resolve(name, version)
        else:
            resolved = self.remote.get_bytes(_ref_key(name)).decode().strip()
        entries = self._read_lock()
        entries[name] = resolved
        self._write_lock(entries)
        return resolved

    def unpin(self, name: str) -> None:
        entries = self._read_lock()
        if entries.pop(name, None) is not None:
            self._write_lock(entries)

    def _read_lock(self) -> dict[str, str]:
        if not self.lock_path.exists():
            return {}
        return dict(tomllib.loads(self.lock_path.read_text()).get("datasets", {}))

    def _write_lock(self, entries: dict[str, str]) -> None:
        lines = ["# Pinned dataset versions, managed by `dload pin`.", "[datasets]"]
        lines += [f'{k} = "{v}"' for k, v in sorted(entries.items())]
        self.lock_path.write_text("\n".join(lines) + "\n")

    # -- shard access ----------------------------------------------------------

    def open_shard(self, shard: ShardInfo) -> PinnedShard:
        """Pinned local handle; downloads from the remote when not cached."""
        key = _shard_key(shard.digest)
        return self.cache.ensure(
            shard.digest, shard.size, lambda dest: self.remote.get_to_file(key, dest)
        )

    def fetch(
        self, manifest: Manifest, *, workers: int = 8, progress: Progress | None = None
    ) -> None:
        """Materialize every shard of `manifest` into the local cache."""
        say = progress or (lambda _msg: None)
        budget = self.cache.budget
        if budget is not None and manifest.total_bytes > budget:
            raise CacheFullError(
                f"{manifest.name} is {format_size(manifest.total_bytes)} but the cache budget "
                f"is {format_size(budget)}; stream it instead of fetching it fully"
            )
        missing = [s for s in manifest.shards if not self.cache.contains(s.digest)]
        say(
            f"{manifest.name}: {len(manifest.shards) - len(missing)} shards cached, fetching {len(missing)}"
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, _ in enumerate(
                pool.map(lambda s: self.open_shard(s).release(), missing), 1
            ):
                say(f"fetched {i}/{len(missing)}")

    # -- destructive maintenance -------------------------------------------------

    def delete_dataset(self, name: str, version: str | None = None) -> None:
        """Delete one version, or the whole dataset when version is None.
        Shared shards are left alone — run gc() to reclaim them."""
        if version is None:
            for digest in self._manifest_digests(name):
                self.remote.delete(_manifest_key(name, digest))
            self.remote.delete(_ref_key(name))
            return
        resolved = self.resolve(name, version)
        self.remote.delete(_manifest_key(name, resolved))
        remaining = self._load_manifests(name, self._manifest_digests(name))
        if remaining:
            self.remote.put_bytes(_ref_key(name), remaining[0].version.encode())
        else:
            self.remote.delete(_ref_key(name))

    def gc(self, *, progress: Progress | None = None) -> tuple[int, int]:
        """Delete remote shards referenced by no manifest of any dataset.
        Returns (shards_deleted, bytes_freed). Do not run concurrently
        with a commit."""
        say = progress or (lambda _msg: None)
        referenced: set[str] = set()
        for name in self.list_datasets():
            # A dataset observed mid-delete may have no manifests left;
            # skip it rather than aborting the whole collection.
            manifests = self._load_manifests(name, self._manifest_digests(name))
            for m in manifests:
                referenced.update(s.digest for s in m.shards)
        deleted = freed = 0
        for key in list(self.remote.list("shards/")):
            digest = key.rsplit("/", 1)[-1]
            if digest not in referenced:
                freed += self.remote.size(key)
                self.remote.delete(key)
                self.cache.evict(digest)
                deleted += 1
                say(f"deleted orphan shard {digest[:12]}")
        return deleted, freed


class Dataset:
    """Handle to one dataset version; the entry point for streaming."""

    def __init__(self, repo: Repository, manifest: Manifest) -> None:
        self.repo = repo
        self.manifest = manifest

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version

    def __len__(self) -> int:
        return self.manifest.num_samples

    def __repr__(self) -> str:
        m = self.manifest
        return (
            f"Dataset({m.name}@{m.version[:12]}, {m.num_samples} samples, "
            f"{len(m.shards)} shards, {format_size(m.total_bytes)})"
        )

    def samples(self, *, prefetch: int = 3) -> "Pipeline":
        """A Pipeline yielding raw samples `(key, {field: bytes})`.

        `prefetch` is the sliding-window depth in shards (current shard
        included) held pinned in the local cache while streaming.
        """
        from .pipeline import Pipeline, SourceNode

        return Pipeline(SourceNode(self, prefetch=prefetch))

    def fetch(self, *, workers: int = 8, progress: Progress | None = None) -> None:
        """Materialize the whole dataset locally (fails fast if it cannot fit)."""
        self.repo.fetch(self.manifest, workers=workers, progress=progress)
