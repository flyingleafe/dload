"""Shared test helpers for test_repo.py and test_pipeline.py.

Not a test module itself (no test_* functions here), so pytest will not
collect it directly.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from dload.cache import ShardCache
from dload.pack import Sample
from dload.remote import LocalRemote
from dload.repo import Repository


def make_samples(
    n: int, size: int = 64, seed: int = 0, prefix: str = "s"
) -> list[Sample]:
    """Deterministic samples: (key, {"data": bytes}) with seeded content."""
    rng = random.Random(seed)
    return [(f"{prefix}{i:05d}", {"data": rng.randbytes(size)}) for i in range(n)]


def shard_tagged_samples(
    num_shards: int, per_shard: int, size: int = 64, seed: int = 0
) -> list[Sample]:
    """Samples whose key encodes (shard_index, position). Paired with a
    `target_shard_size` of `size * per_shard`, each shard ends up holding
    exactly the samples tagged with its index, so tests can recover which
    pack-shard a streamed sample came from just by looking at its key."""
    rng = random.Random(seed)
    out = []
    for si in range(num_shards):
        for j in range(per_shard):
            out.append((f"{si:04d}-{j:04d}", {"data": rng.randbytes(size)}))
    return out


def shard_index_of(key: str) -> int:
    return int(key.split("-", 1)[0])


def build_repo(
    tmp_path: Path,
    *,
    budget: int | None = None,
    remote: LocalRemote | None = None,
    remote_name: str = "remote",
    cache_name: str = "cache",
    lock_name: str = "dload.lock",
) -> Repository:
    """A Repository wired to a LocalRemote + ShardCache under tmp_path.
    Pass an existing `remote` to build a second Repository (e.g. a fresh
    cold cache) pointed at the same backing store."""
    remote = remote if remote is not None else LocalRemote(tmp_path / remote_name)
    cache = ShardCache(tmp_path / cache_name, budget)
    return Repository(remote, cache, lock_path=tmp_path / lock_name)


def local_remote(repo: Repository) -> LocalRemote:
    """`Repository.remote` is typed as the `Remote` protocol so production
    code can plug in any backend; every test wires up a `LocalRemote`
    though (see `build_repo`), so narrow it here once for tests that need
    LocalRemote-only introspection (`op_counts`)."""
    remote = repo.remote
    assert isinstance(remote, LocalRemote)
    return remote


class SlowLocalRemote(LocalRemote):
    """A LocalRemote whose get_to_file sleeps briefly before delegating.
    Used to prove the pipeline still yields every sample correctly when
    shard downloads are slow (prefetch-overlap correctness, not timing)."""

    def __init__(self, root: Path, delay: float = 0.01) -> None:
        super().__init__(root)
        self.delay = delay

    def get_to_file(self, key: str, dest: Path) -> None:
        time.sleep(self.delay)
        super().get_to_file(key, dest)


def freeze_repo_clock(monkeypatch, moments) -> None:
    """Monkeypatch `dload.repo`'s `datetime.now(UTC)` to yield each of
    `moments` in turn, repeating the last one once exhausted. Manifests
    embed a wall-clock `created` field in their version-defining canonical
    JSON, so without this, "commit identical data twice" tests would be
    non-deterministic whenever the two commits straddle a second boundary.
    """
    import dload.repo as repo_mod

    it = iter(moments)
    state: dict = {"last": None}

    class _FrozenDatetime:
        @staticmethod
        def now(tz=None):
            try:
                state["last"] = next(it)
            except StopIteration:
                pass
            return state["last"]

    monkeypatch.setattr(repo_mod, "datetime", _FrozenDatetime)
