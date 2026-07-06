from __future__ import annotations

import tomllib
from datetime import UTC, datetime

import pytest

from dload.errors import CacheFullError, ConfigError, NotFoundError
from helpers import build_repo, freeze_repo_clock, local_remote, make_samples

# -- commit: shard cutting, manifest totals, refs/latest, progress -----------


def test_commit_cuts_shards_near_target_size_and_updates_manifest_and_ref(tmp_path):
    repo = build_repo(tmp_path)
    samples = make_samples(23, size=64, seed=1)
    target = 64 * 5  # 5 samples/shard nominally -> 4 full shards + 1 partial

    progress_msgs = []
    manifest = repo.commit(
        "ds", samples, target_shard_size=target, progress=progress_msgs.append
    )

    assert manifest.num_samples == 23
    assert [s.num_samples for s in manifest.shards] == [5, 5, 5, 5, 3]
    assert (
        manifest.total_bytes >= 23 * 64
    )  # payload lower bound (index/footer add a bit)
    for s in manifest.shards[:-1]:
        assert target <= s.size <= target + 1000
    assert manifest.shards[-1].size <= 3 * 64 + 1000

    assert any("shard" in m for m in progress_msgs)
    assert any("committed" in m for m in progress_msgs)

    ref = repo.remote.get_bytes("datasets/ds/refs/latest").decode().strip()
    assert ref == manifest.version


def test_commit_rejects_invalid_name(tmp_path):
    repo = build_repo(tmp_path)
    with pytest.raises(ConfigError):
        repo.commit("a/b", make_samples(1))
    with pytest.raises(ConfigError):
        repo.commit("", make_samples(1))


def test_commit_empty_samples_yields_zero_shard_manifest(tmp_path):
    repo = build_repo(tmp_path)
    manifest = repo.commit("empty", [])
    assert manifest.num_samples == 0
    assert manifest.shards == ()

    ds = repo.dataset("empty")
    assert list(ds.samples()) == []


# -- content-addressed dedup --------------------------------------------------


def test_commit_dedup_identical_data_same_version_no_reupload(tmp_path, monkeypatch):
    repo = build_repo(tmp_path)
    samples = make_samples(12, size=64, seed=2)
    target = 64 * 4  # 3 shards of 4 samples

    freeze_repo_clock(monkeypatch, [datetime(2026, 1, 1, tzinfo=UTC)])
    m1 = repo.commit("ds", samples, target_shard_size=target)
    assert len(m1.shards) == 3
    put_file_after_first = local_remote(repo).op_counts.get("put_file", 0)
    assert put_file_after_first == 3

    m2 = repo.commit("ds", samples, target_shard_size=target)
    assert m2.version == m1.version
    assert (
        local_remote(repo).op_counts.get("put_file", 0) == put_file_after_first
    )  # nothing re-uploaded


def test_commit_different_data_new_version_shares_unchanged_shards(
    tmp_path, monkeypatch
):
    repo = build_repo(tmp_path)
    samples = make_samples(12, size=64, seed=3)
    target = 64 * 4  # 3 shards: [4, 4, 4]

    freeze_repo_clock(monkeypatch, [datetime(2026, 1, 1, tzinfo=UTC)])
    m1 = repo.commit("ds", samples, target_shard_size=target)
    assert len(m1.shards) == 3

    changed = list(samples)
    changed[-1] = (changed[-1][0], {"data": b"X" * 64})  # only the last sample differs
    put_file_before = local_remote(repo).op_counts.get("put_file", 0)
    m2 = repo.commit("ds", changed, target_shard_size=target)

    assert m2.version != m1.version
    assert m1.shards[0].digest == m2.shards[0].digest
    assert m1.shards[1].digest == m2.shards[1].digest
    assert m1.shards[2].digest != m2.shards[2].digest
    assert (
        local_remote(repo).op_counts.get("put_file", 0) == put_file_before + 1
    )  # only the changed shard


def test_commit_populates_local_cache_for_immediate_read(tmp_path):
    repo = build_repo(tmp_path)
    samples = make_samples(20, size=64, seed=4)
    repo.commit("ds", samples, target_shard_size=64 * 5)

    baseline = local_remote(repo).op_counts.get("get_to_file", 0)
    ds = repo.dataset("ds")
    got = list(ds.samples())
    assert len(got) == 20
    assert (
        local_remote(repo).op_counts.get("get_to_file", 0) == baseline
    )  # served entirely from the warm cache


# -- resolve -------------------------------------------------------------------


def test_resolve_explicit_full_version(tmp_path):
    repo = build_repo(tmp_path)
    m = repo.commit("ds", make_samples(3), target_shard_size=10**9)
    assert repo.resolve("ds", m.version) == m.version


def test_resolve_unique_prefix(tmp_path):
    repo = build_repo(tmp_path)
    m = repo.commit("ds", make_samples(3), target_shard_size=10**9)
    assert repo.resolve("ds", m.version[:10]) == m.version


def test_resolve_ambiguous_prefix_raises(tmp_path, monkeypatch):
    repo = build_repo(tmp_path)
    repo.commit("ds", make_samples(3), target_shard_size=10**9)
    monkeypatch.setattr(
        repo,
        "_manifest_digests",
        lambda name: ["abcd" + "0" * 60, "abcd" + "1" * 60],
    )
    with pytest.raises(NotFoundError):
        repo.resolve("ds", "abcd")


def test_resolve_unknown_dataset_raises(tmp_path):
    repo = build_repo(tmp_path)
    with pytest.raises(NotFoundError):
        repo.resolve("nope")


def test_resolve_lock_pin_respected_and_explicit_overrides(tmp_path, monkeypatch):
    repo = build_repo(tmp_path)
    freeze_repo_clock(monkeypatch, [datetime(2026, 1, 1, tzinfo=UTC)])
    m1 = repo.commit("ds", make_samples(3, seed=1), target_shard_size=10**9)
    freeze_repo_clock(monkeypatch, [datetime(2026, 1, 2, tzinfo=UTC)])
    m2 = repo.commit("ds", make_samples(3, seed=2), target_shard_size=10**9)
    assert m1.version != m2.version

    repo.pin("ds", m1.version)
    assert repo.resolve("ds") == m1.version  # pin wins over refs/latest
    assert (
        repo.resolve("ds", m2.version) == m2.version
    )  # explicit version overrides pin


# -- pin / unpin -----------------------------------------------------------------


def test_pin_unpin_round_trip_and_lock_format(tmp_path, monkeypatch):
    repo = build_repo(tmp_path)
    freeze_repo_clock(monkeypatch, [datetime(2026, 1, 1, tzinfo=UTC)])
    m1 = repo.commit("ds", make_samples(3, seed=1), target_shard_size=10**9)
    freeze_repo_clock(monkeypatch, [datetime(2026, 1, 2, tzinfo=UTC)])
    m2 = repo.commit(
        "ds", make_samples(3, seed=2), target_shard_size=10**9
    )  # latest = m2

    resolved = repo.pin("ds", m1.version)
    assert resolved == m1.version
    assert repo._read_lock() == {"ds": m1.version}

    raw = repo.lock_path.read_text()
    parsed = tomllib.loads(raw)  # must be valid TOML
    assert parsed["datasets"]["ds"] == m1.version

    # pin() with no explicit version bypasses the existing pin, uses refs/latest
    resolved = repo.pin("ds")
    assert resolved == m2.version
    assert repo.resolve("ds") == m2.version

    repo.unpin("ds")
    assert repo._read_lock() == {}
    assert repo.resolve("ds") == m2.version  # falls back to refs/latest


# -- versions / delete / gc / fetch -----------------------------------------------


def test_versions_newest_first(tmp_path, monkeypatch):
    repo = build_repo(tmp_path)
    times = [
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
        datetime(2026, 1, 3, tzinfo=UTC),
    ]
    versions = []
    for i, t in enumerate(times):
        freeze_repo_clock(monkeypatch, [t])
        m = repo.commit("ds", make_samples(3, seed=i), target_shard_size=10**9)
        versions.append(m.version)

    vs = repo.versions("ds")
    assert [v.version for v in vs] == list(reversed(versions))


def test_delete_dataset_one_version_repoints_ref(tmp_path, monkeypatch):
    repo = build_repo(tmp_path)
    times = [
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
        datetime(2026, 1, 3, tzinfo=UTC),
    ]
    versions = []
    for i, t in enumerate(times):
        freeze_repo_clock(monkeypatch, [t])
        m = repo.commit("ds", make_samples(3, seed=i), target_shard_size=10**9)
        versions.append(m.version)
    v1, v2, v3 = versions  # v3 = latest

    repo.delete_dataset("ds", version=v1)
    assert {m.version for m in repo.versions("ds")} == {v2, v3}
    assert repo.resolve("ds") == v3  # unaffected, still latest
    with pytest.raises(NotFoundError):
        repo.manifest("ds", v1)

    repo.delete_dataset("ds", version=v3)
    assert {m.version for m in repo.versions("ds")} == {v2}
    assert repo.resolve("ds") == v2  # ref repointed to the new latest


def test_delete_whole_dataset(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("ds", make_samples(3), target_shard_size=10**9)
    assert "ds" in repo.list_datasets()

    repo.delete_dataset("ds")
    assert "ds" not in repo.list_datasets()
    with pytest.raises(NotFoundError):
        repo.versions("ds")


def test_gc_removes_only_unreferenced_shards_and_evicts_cache(tmp_path):
    repo = build_repo(tmp_path)
    shared_samples = make_samples(10, size=64, seed=5)
    target = 64 * 10  # single shard
    mA = repo.commit("A", shared_samples, target_shard_size=target)
    mB = repo.commit("B", shared_samples, target_shard_size=target)
    shared_digest = mA.shards[0].digest
    assert shared_digest == mB.shards[0].digest  # shared shard, deduped

    orphan_samples = make_samples(5, size=64, seed=99, prefix="o")
    mC = repo.commit("orphan_ds", orphan_samples, target_shard_size=target)
    orphan_digest = mC.shards[0].digest
    assert repo.cache.contains(orphan_digest)
    assert repo.cache.contains(shared_digest)

    repo.delete_dataset("orphan_ds")  # its shard is now unreferenced by any manifest

    deleted, freed = repo.gc()
    assert deleted == 1
    assert freed == mC.shards[0].size

    assert not repo.remote.exists(f"shards/{orphan_digest[:2]}/{orphan_digest}")
    assert repo.remote.exists(f"shards/{shared_digest[:2]}/{shared_digest}")
    assert not repo.cache.contains(orphan_digest)
    assert repo.cache.contains(shared_digest)  # still referenced by A and B


def test_fetch_materializes_all_shards(tmp_path):
    writer_repo = build_repo(tmp_path, budget=None)
    samples = make_samples(20, size=64, seed=6)
    manifest = writer_repo.commit("ds", samples, target_shard_size=64 * 5)

    reader_repo = build_repo(
        tmp_path,
        budget=None,
        remote=local_remote(writer_repo),
        cache_name="cache_reader",
    )
    ds = reader_repo.dataset("ds")
    assert reader_repo.cache.used_bytes() == 0

    ds.fetch()
    for s in manifest.shards:
        assert reader_repo.cache.contains(s.digest)


def test_fetch_raises_when_budget_too_small(tmp_path):
    writer_repo = build_repo(tmp_path, budget=None)
    samples = make_samples(20, size=64, seed=7)
    manifest = writer_repo.commit("ds", samples, target_shard_size=64 * 5)

    small_budget = manifest.total_bytes // 2
    reader_repo = build_repo(
        tmp_path,
        budget=small_budget,
        remote=local_remote(writer_repo),
        cache_name="cache_small",
    )
    ds = reader_repo.dataset("ds")
    with pytest.raises(CacheFullError):
        ds.fetch()
    assert reader_repo.cache.used_bytes() == 0


# -- read efficiency -------------------------------------------------------------


def test_streaming_cold_then_warm_get_to_file_counts(tmp_path):
    writer_repo = build_repo(tmp_path, budget=None)
    samples = make_samples(50, size=64, seed=8)
    manifest = writer_repo.commit("ds", samples, target_shard_size=64 * 5)
    assert len(manifest.shards) == 10

    cold_repo = build_repo(
        tmp_path, budget=None, remote=local_remote(writer_repo), cache_name="cache_cold"
    )
    baseline_get_bytes = local_remote(cold_repo).op_counts.get("get_bytes", 0)
    baseline_get_to_file = local_remote(cold_repo).op_counts.get("get_to_file", 0)

    ds = cold_repo.dataset("ds")  # resolves ref + fetches manifest -> 2 get_bytes
    got = list(ds.samples())
    assert len(got) == 50
    assert (
        local_remote(cold_repo).op_counts.get("get_to_file", 0) - baseline_get_to_file
        == 10
    )
    assert (
        local_remote(cold_repo).op_counts.get("get_bytes", 0) - baseline_get_bytes == 2
    )

    baseline2 = local_remote(cold_repo).op_counts.get("get_to_file", 0)
    got2 = list(ds.samples())
    assert len(got2) == 50
    assert (
        local_remote(cold_repo).op_counts.get("get_to_file", 0) == baseline2
    )  # warm: no more downloads
