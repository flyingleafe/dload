"""Local content-addressed shard cache with pinning and LRU eviction.

Layout: <root>/shards/<digest[:2]>/<digest>. Recency = file mtime (touched
on every access); eviction scans the directory, so no separate index can
drift out of sync. Pins are in-process reference counts: a pinned shard is
never evicted by this process. Downloads land in <root>/tmp/ and are
atomically renamed into place, so concurrent processes sharing a cache dir
can only race benignly (both download, one rename wins, files are
identical by content-addressing).
"""

from __future__ import annotations

import hashlib
import os
import threading
import uuid
from collections.abc import Callable
from pathlib import Path

from dload.errors import CacheFullError, IntegrityError


class ShardCache:
    def __init__(self, root: Path, budget: int | None) -> None:
        """budget=None means unlimited. Creates directories."""
        self.root = Path(root)
        self.budget = budget
        self.shards_dir = self.root / "shards"
        self.tmp_dir = self.root / "tmp"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._pins: dict[str, int] = {}
        self._inflight: dict[str, threading.Event] = {}
        self._reserved = 0

    # -- pickling ------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Drop the lock, in-flight events and pin bookkeeping — pins are
        documented as per-process, so an unpickled copy (e.g. in a
        DataLoader worker) starts with none held and none in flight."""
        state = self.__dict__.copy()
        del state["_lock"], state["_inflight"], state["_pins"], state["_reserved"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._pins = {}
        self._inflight = {}
        self._reserved = 0

    # -- internal helpers --------------------------------------------------

    def _shard_path(self, digest: str) -> Path:
        return self.shards_dir / digest[:2] / digest

    def _tmp_path(self) -> Path:
        return self.tmp_dir / f"{uuid.uuid4().hex}.tmp"

    # -- queries ---------------------------------------------------------

    def contains(self, digest: str) -> bool:
        return self._shard_path(digest).exists()

    def used_bytes(self) -> int:
        """Total size of cached shards (scan; cheap at shard granularity)."""
        return sum(size for _, size, _ in self.entries())

    def entries(self) -> list[tuple[str, int, float]]:
        """[(digest, size, mtime)] sorted oldest-first."""
        result = []
        if not self.shards_dir.exists():
            return result
        for shard_dir in self.shards_dir.iterdir():
            if not shard_dir.is_dir():
                continue
            for f in shard_dir.iterdir():
                if not f.is_file():
                    continue
                st = f.stat()
                result.append((f.name, st.st_size, st.st_mtime))
        result.sort(key=lambda e: e[2])
        return result

    # -- the core operation ----------------------------------------------

    def ensure(
        self,
        digest: str,
        size: int,
        fetch: Callable[[Path], None],
    ) -> "PinnedShard":
        """Return a pinned handle to the shard, downloading it if absent.

        If absent: evict LRU *unpinned* shards until `size` fits in the
        budget (raise CacheFullError if everything left is pinned or
        size > budget), then call fetch(tmp_path), verify sha256 of the
        result equals `digest` (raise IntegrityError, delete tmp),
        atomic-rename into place.

        Always bumps mtime and increments the pin count. Thread-safe
        (internal lock around bookkeeping; concurrent ensure() of the same
        digest must not download twice — per-digest in-flight locking).
        """
        while True:
            with self._lock:
                path = self._shard_path(digest)
                if path.exists():
                    os.utime(path, None)
                    self._pins[digest] = self._pins.get(digest, 0) + 1
                    return PinnedShard(self, digest, path)

                waiter = self._inflight.get(digest)
                if waiter is None:
                    self._reserve_locked(digest, size)
                    self._inflight[digest] = threading.Event()

            if waiter is not None:
                waiter.wait()
                continue

            # We are the fetcher.
            try:
                return self._fetch_and_install(digest, size, fetch)
            finally:
                with self._lock:
                    event = self._inflight.pop(digest, None)
                    if event is not None:
                        event.set()

    def _reserve_locked(self, digest: str, size: int) -> None:
        """Evict LRU unpinned shards (and account for other in-flight
        reservations) until `size` fits in the budget. Must be called
        while holding self._lock."""
        if self.budget is None:
            self._reserved += size
            return
        if size > self.budget:
            raise CacheFullError(
                f"shard {digest} ({size} bytes) exceeds cache budget ({self.budget} bytes)"
            )

        entries = self.entries()
        used = sum(entry_size for _, entry_size, _ in entries) + self._reserved
        if used + size <= self.budget:
            self._reserved += size
            return

        for other_digest, other_size, _ in entries:
            if used + size <= self.budget:
                break
            if self._pins.get(other_digest, 0) > 0:
                continue
            self._remove_locked(other_digest)
            used -= other_size

        if used + size > self.budget:
            raise CacheFullError(
                f"cannot fit shard {digest} ({size} bytes) in cache budget "
                f"({self.budget} bytes): remaining shards are pinned"
            )
        self._reserved += size

    def _fetch_and_install(
        self, digest: str, size: int, fetch: Callable[[Path], None]
    ) -> "PinnedShard":
        tmp_path = self._tmp_path()
        unreserved = False
        try:
            fetch(tmp_path)
            hasher = hashlib.sha256()
            with tmp_path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    hasher.update(chunk)
            actual = hasher.hexdigest()
            if actual != digest:
                raise IntegrityError(
                    f"digest mismatch for shard {digest}: got {actual}"
                )

            path = self._shard_path(digest)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Install and pin under one lock: a shard visible on disk but
            # not yet pinned would be fair game for concurrent eviction.
            with self._lock:
                self._reserved -= size
                unreserved = True
                os.replace(tmp_path, path)
                os.utime(path, None)
                self._pins[digest] = self._pins.get(digest, 0) + 1
            return PinnedShard(self, digest, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            if not unreserved:
                with self._lock:
                    self._reserved -= size
            raise

    def pin_existing(self, digest: str) -> "PinnedShard | None":
        """Pin without fetching; None if absent."""
        with self._lock:
            path = self._shard_path(digest)
            if not path.exists():
                return None
            os.utime(path, None)
            self._pins[digest] = self._pins.get(digest, 0) + 1
            return PinnedShard(self, digest, path)

    # -- maintenance -------------------------------------------------------

    def _remove_locked(self, digest: str) -> bool:
        path = self._shard_path(digest)
        if not path.exists():
            return False
        path.unlink()
        try:
            path.parent.rmdir()
        except OSError:
            pass
        return True

    def evict(self, digest: str) -> bool:
        """Remove one shard if present and unpinned. True if removed."""
        with self._lock:
            if self._pins.get(digest, 0) > 0:
                return False
            return self._remove_locked(digest)

    def clear(self) -> int:
        """Remove all unpinned shards; returns bytes freed."""
        freed = 0
        with self._lock:
            for digest, size, _ in self.entries():
                if self._pins.get(digest, 0) > 0:
                    continue
                if self._remove_locked(digest):
                    freed += size
        return freed

    # -- pin bookkeeping (used by PinnedShard) ------------------------------

    def _release(self, digest: str) -> None:
        with self._lock:
            count = self._pins.get(digest, 0)
            if count <= 1:
                self._pins.pop(digest, None)
            else:
                self._pins[digest] = count - 1


class PinnedShard:
    """Context manager handle; the shard file is guaranteed present until
    release. Re-entrant pinning of the same digest stacks refcounts.

        with cache.ensure(d, size, fetch) as path:
            reader = PackReader(path)

    .path: Path, .digest: str, .release() idempotent; __exit__ releases.
    """

    def __init__(self, cache: ShardCache, digest: str, path: Path) -> None:
        self._cache = cache
        self.digest = digest
        self.path = path
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._cache._release(self.digest)

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, *exc: object) -> None:
        self.release()
