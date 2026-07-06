import hashlib
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dload.cache import ShardCache
from dload.errors import CacheFullError, IntegrityError


def shard(content: bytes) -> tuple[str, int, Callable[[Path], None]]:
    digest = hashlib.sha256(content).hexdigest()

    def fetch(path: Path) -> None:
        path.write_bytes(content)

    return digest, len(content), fetch


def cache(tmp_path: Path, budget: int | None = None) -> ShardCache:
    return ShardCache(tmp_path, budget)


def set_mtime(path: Path, when: float) -> None:
    os.utime(path, (when, when))


# -- basic hit/miss ----------------------------------------------------------


def test_miss_fetches_and_stores(tmp_path):
    c = cache(tmp_path)
    digest, size, fetch = shard(b"hello world")
    assert not c.contains(digest)

    with c.ensure(digest, size, fetch) as path:
        assert path.read_bytes() == b"hello world"
    assert c.contains(digest)


def test_hit_does_not_refetch(tmp_path):
    c = cache(tmp_path)
    digest, size, fetch = shard(b"payload")
    with c.ensure(digest, size, fetch):
        pass

    def failing_fetch(path: Path) -> None:
        raise AssertionError("fetch should not be called on a cache hit")

    with c.ensure(digest, size, failing_fetch) as path:
        assert path.read_bytes() == b"payload"


def test_ensure_bumps_mtime_and_pins(tmp_path):
    c = cache(tmp_path)
    digest, size, fetch = shard(b"data")
    with c.ensure(digest, size, fetch) as path:
        pass
    set_mtime(path, time.time() - 1000)
    old_mtime = path.stat().st_mtime

    with c.ensure(digest, size, fetch):
        pass
    assert path.stat().st_mtime > old_mtime


def test_no_tmp_files_left_after_successful_fetch(tmp_path):
    c = cache(tmp_path)
    digest, size, fetch = shard(b"clean up nicely")
    with c.ensure(digest, size, fetch):
        pass
    assert list(c.tmp_dir.iterdir()) == []


# -- eviction ordering --------------------------------------------------------


def test_eviction_removes_oldest_first(tmp_path):
    c = cache(tmp_path, budget=100)
    now = time.time()
    digests = {}
    for i, name in enumerate(["a", "b", "c"]):
        digest, size, fetch = shard(name.encode() * 30)
        with c.ensure(digest, size, fetch) as path:
            digests[name] = digest
        set_mtime(path, now - (3 - i) * 100)  # a oldest, c newest

    d_digest, d_size, d_fetch = shard(b"d" * 30)
    with c.ensure(d_digest, d_size, d_fetch):
        pass

    assert not c.contains(digests["a"])
    assert c.contains(digests["b"])
    assert c.contains(digests["c"])
    assert c.contains(d_digest)


def test_entries_sorted_oldest_first(tmp_path):
    c = cache(tmp_path)
    now = time.time()
    order = []
    for i, name in enumerate(["x", "y", "z"]):
        digest, size, fetch = shard(name.encode())
        with c.ensure(digest, size, fetch) as path:
            pass
        set_mtime(path, now - (3 - i) * 50)
        order.append(digest)

    got = [d for d, _, _ in c.entries()]
    assert got == order


# -- pinning survives eviction pressure ---------------------------------------


def test_pinned_shard_survives_eviction_pressure(tmp_path):
    c = cache(tmp_path, budget=100)
    a_digest, a_size, a_fetch = shard(b"a" * 30)
    pinned_a = c.ensure(a_digest, a_size, a_fetch)
    set_mtime(pinned_a.path, time.time() - 1000)  # oldest, but pinned

    b_digest, b_size, b_fetch = shard(b"b" * 30)
    with c.ensure(b_digest, b_size, b_fetch):
        pass

    d_digest, d_size, d_fetch = shard(b"d" * 30)
    with c.ensure(d_digest, d_size, d_fetch):
        pass

    assert c.contains(a_digest)  # pinned, survived despite being oldest
    assert c.contains(d_digest)
    pinned_a.release()


# -- CacheFullError ------------------------------------------------------------


def test_cache_full_when_all_pinned(tmp_path):
    c = cache(tmp_path, budget=60)
    pins = []
    for name in ["a", "b"]:
        digest, size, fetch = shard(name.encode() * 30)
        pins.append(c.ensure(digest, size, fetch))

    d_digest, d_size, d_fetch = shard(b"d" * 30)
    with pytest.raises(CacheFullError):
        c.ensure(d_digest, d_size, d_fetch)
    assert not c.contains(d_digest)

    for p in pins:
        p.release()
    with c.ensure(d_digest, d_size, d_fetch):
        pass
    assert c.contains(d_digest)


def test_cache_full_when_size_exceeds_budget(tmp_path):
    c = cache(tmp_path, budget=10)
    digest, size, fetch = shard(b"x" * 100)
    with pytest.raises(CacheFullError):
        c.ensure(digest, size, fetch)
    assert not c.contains(digest)
    assert list(c.tmp_dir.iterdir()) == []


# -- integrity -----------------------------------------------------------------


def test_integrity_failure_cleans_tmp_and_raises(tmp_path):
    c = cache(tmp_path)
    digest = hashlib.sha256(b"expected content").hexdigest()

    def bad_fetch(path: Path) -> None:
        path.write_bytes(b"wrong content")

    with pytest.raises(IntegrityError):
        c.ensure(digest, 13, bad_fetch)

    assert not c.contains(digest)
    assert list(c.tmp_dir.iterdir()) == []
    assert c.used_bytes() == 0

    def good_fetch(path: Path) -> None:
        path.write_bytes(b"expected content")

    with c.ensure(digest, 17, good_fetch) as path:
        assert path.read_bytes() == b"expected content"


# -- concurrency -----------------------------------------------------------------


def test_concurrent_ensure_same_digest_fetches_once(tmp_path):
    c = cache(tmp_path)
    digest, size, _ = shard(b"shared payload")
    call_count = 0
    count_lock = threading.Lock()

    def fetch(path: Path) -> None:
        nonlocal call_count
        with count_lock:
            call_count += 1
        time.sleep(0.1)
        path.write_bytes(b"shared payload")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: c.ensure(digest, size, fetch), range(8)))

    assert call_count == 1
    paths = {r.path for r in results}
    assert paths == {c._shard_path(digest)}
    for r in results:
        r.release()


def test_concurrent_ensure_different_digests_respects_budget(tmp_path):
    c = cache(tmp_path, budget=100)

    def make_item(i: int) -> tuple[str, int, Callable[[Path], None]]:
        content = bytes([i]) * 30
        digest = hashlib.sha256(content).hexdigest()

        def fetch(path: Path) -> None:
            time.sleep(0.1)
            path.write_bytes(content)

        return digest, len(content), fetch

    items = [make_item(i) for i in range(5)]

    def do_ensure(item):
        digest, size, fetch = item
        try:
            return c.ensure(digest, size, fetch)
        except CacheFullError:
            return None

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(do_ensure, items))

    succeeded = [r for r in results if r is not None]
    assert len(succeeded) == 3
    assert c.used_bytes() <= 100
    for r in succeeded:
        r.release()


# -- clear / evict --------------------------------------------------------------


def test_evict_removes_unpinned(tmp_path):
    c = cache(tmp_path)
    digest, size, fetch = shard(b"evict me")
    with c.ensure(digest, size, fetch):
        pass
    assert c.evict(digest) is True
    assert not c.contains(digest)


def test_evict_refuses_pinned(tmp_path):
    c = cache(tmp_path)
    digest, size, fetch = shard(b"stay pinned")
    pinned = c.ensure(digest, size, fetch)
    assert c.evict(digest) is False
    assert c.contains(digest)
    pinned.release()
    assert c.evict(digest) is True


def test_evict_missing_returns_false(tmp_path):
    c = cache(tmp_path)
    assert c.evict("nonexistent" * 4) is False


def test_clear_removes_only_unpinned(tmp_path):
    c = cache(tmp_path)
    a_digest, a_size, a_fetch = shard(b"a" * 10)
    b_digest, b_size, b_fetch = shard(b"b" * 20)
    pinned_a = c.ensure(a_digest, a_size, a_fetch)
    with c.ensure(b_digest, b_size, b_fetch):
        pass

    freed = c.clear()
    assert freed == 20
    assert c.contains(a_digest)
    assert not c.contains(b_digest)
    pinned_a.release()


# -- unlimited budget ------------------------------------------------------------


def test_unlimited_budget_never_evicts(tmp_path):
    c = cache(tmp_path, budget=None)
    digests = []
    for i in range(10):
        digest, size, fetch = shard(bytes([i]) * 1000)
        with c.ensure(digest, size, fetch):
            pass
        digests.append(digest)

    assert all(c.contains(d) for d in digests)
    assert c.used_bytes() == 10000


# -- PinnedShard semantics --------------------------------------------------------


def test_pinned_shard_release_idempotent(tmp_path):
    c = cache(tmp_path)
    digest, size, fetch = shard(b"idempotent")
    pinned = c.ensure(digest, size, fetch)
    pinned.release()
    pinned.release()
    assert c.evict(digest) is True


def test_pinned_shard_context_manager_releases_on_exit(tmp_path):
    c = cache(tmp_path, budget=50)
    digest, size, fetch = shard(b"ctx" * 5)
    with c.ensure(digest, size, fetch) as path:
        assert isinstance(path, Path)
        assert c.evict(digest) is False
    assert c.evict(digest) is True


def test_reentrant_pin_stacks_refcount(tmp_path):
    c = cache(tmp_path)
    digest, size, fetch = shard(b"stacked")
    first = c.ensure(digest, size, fetch)
    second = c.ensure(digest, size, fetch)
    first.release()
    assert c.evict(digest) is False  # still pinned once (second)
    second.release()
    assert c.evict(digest) is True
