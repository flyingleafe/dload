from __future__ import annotations

import pytest

from dload.errors import NotFoundError
from dload.pipeline import choice, concat, mix, random_stream, zip_with
from helpers import build_repo, local_remote, make_samples, shard_tagged_samples

# -- module-level transform helpers ---------------------------------------------
# fingerprint() refuses lambdas/locals (no stable qualname) and the DataLoader
# spawn-pickling contract already required module-level fns for real pipelines,
# so every transform below is defined here, at module top level.


def _prefixed(sample):
    """Derive-pipeline transform: relabels the key, leaves fields untouched."""
    key, fields = sample
    return (f"d-{key}", fields)


def _upper_key(sample):
    key, fields = sample
    return (key.upper(), fields)


def _lower_key(sample):
    key, fields = sample
    return (key.lower(), fields)


def _identity(sample):
    return sample


def _pick_left(a, b):
    return a


# -- fingerprint: determinism ----------------------------------------------------


def test_fingerprint_same_pipeline_built_twice_same_repo(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("ds", make_samples(10, seed=1), target_shard_size=10**9)

    def build():
        return repo.dataset("ds").samples().shuffle(seed=3).map(_prefixed)

    assert build().fingerprint() == build().fingerprint()


def test_fingerprint_same_across_cold_cache_second_repo(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("ds", make_samples(10, seed=2), target_shard_size=10**9)

    def build(r):
        return r.dataset("ds").samples().shuffle(seed=3).map(_prefixed)

    fp1 = build(repo).fingerprint()

    repo2 = build_repo(
        tmp_path, remote=local_remote(repo), cache_name="c2", lock_name="l2"
    )
    fp2 = build(repo2).fingerprint()
    assert fp1 == fp2


def test_fingerprint_changes_with_source_version(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("ds", make_samples(10, seed=4), target_shard_size=10**9)
    fp1 = repo.dataset("ds").samples().shuffle(seed=1).fingerprint()

    repo.commit("ds", make_samples(10, seed=5), target_shard_size=10**9)
    fp2 = repo.dataset("ds").samples().shuffle(seed=1).fingerprint()

    assert fp1 != fp2


def test_fingerprint_changes_with_shuffle_seed(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("ds", make_samples(10, seed=6), target_shard_size=10**9)
    ds = repo.dataset("ds")

    fp1 = ds.samples().shuffle(seed=1).fingerprint()
    fp2 = ds.samples().shuffle(seed=2).fingerprint()
    assert fp1 != fp2


def test_fingerprint_changes_with_take_n(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("ds", make_samples(10, seed=7), target_shard_size=10**9)
    ds = repo.dataset("ds")

    fp1 = ds.samples().shuffle(seed=1).take(3).fingerprint()
    fp2 = ds.samples().shuffle(seed=1).take(4).fingerprint()
    assert fp1 != fp2


def test_fingerprint_changes_with_mapped_function(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("ds", make_samples(10, seed=8), target_shard_size=10**9)
    ds = repo.dataset("ds")

    fp1 = ds.samples().map(_upper_key).fingerprint()
    fp2 = ds.samples().map(_lower_key).fingerprint()
    assert fp1 != fp2


def test_fingerprint_changes_with_batch_size(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("ds", make_samples(10, seed=9), target_shard_size=10**9)
    ds = repo.dataset("ds")

    fp1 = ds.samples().batch(2).fingerprint()
    fp2 = ds.samples().batch(3).fingerprint()
    assert fp1 != fp2


def test_fingerprint_changes_with_tag(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("ds", make_samples(10, seed=10), target_shard_size=10**9)
    pipe = repo.dataset("ds").samples().shuffle(seed=1)

    assert pipe.fingerprint(tag="v1") != pipe.fingerprint(tag="v2")
    assert pipe.fingerprint() != pipe.fingerprint(tag="v1")


# -- source_versions --------------------------------------------------------------


def test_source_versions_single_source(tmp_path):
    repo = build_repo(tmp_path)
    m = repo.commit("ds", make_samples(5, seed=11), target_shard_size=10**9)
    ds = repo.dataset("ds")

    pipe = ds.samples().shuffle(seed=1).map(_prefixed).take(3)
    assert pipe.source_versions() == {"ds": m.version}


def test_source_versions_multi_source_concat(tmp_path):
    repo = build_repo(tmp_path)
    mA = repo.commit(
        "dsA", make_samples(4, seed=12, prefix="a"), target_shard_size=10**9
    )
    mB = repo.commit(
        "dsB", make_samples(4, seed=13, prefix="b"), target_shard_size=10**9
    )
    dsA, dsB = repo.dataset("dsA"), repo.dataset("dsB")

    pipe = concat([dsA.samples(), dsB.samples()])
    assert pipe.source_versions() == {"dsA": mA.version, "dsB": mB.version}


def test_source_versions_multi_source_zip_with(tmp_path):
    repo = build_repo(tmp_path)
    mA = repo.commit(
        "dsA", make_samples(4, seed=14, prefix="a"), target_shard_size=10**9
    )
    mB = repo.commit(
        "dsB", make_samples(4, seed=15, prefix="b"), target_shard_size=10**9
    )
    dsA, dsB = repo.dataset("dsA"), repo.dataset("dsB")

    pipe = zip_with(_pick_left, dsA.samples(), dsB.samples())
    assert pipe.source_versions() == {"dsA": mA.version, "dsB": mB.version}


# -- fingerprint refusals -----------------------------------------------------------


def test_fingerprint_rejects_unseeded_shuffle_accepts_seeded(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("ds", make_samples(5, seed=20), target_shard_size=10**9)
    ds = repo.dataset("ds")

    with pytest.raises(ValueError):
        ds.samples().shuffle().fingerprint()
    ds.samples().shuffle(seed=0).fingerprint()  # must not raise


def test_fingerprint_rejects_unseeded_mix_accepts_seeded(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("dsA", make_samples(3, seed=21, prefix="a"), target_shard_size=10**9)
    repo.commit("dsB", make_samples(3, seed=22, prefix="b"), target_shard_size=10**9)
    dsA, dsB = repo.dataset("dsA"), repo.dataset("dsB")

    with pytest.raises(ValueError):
        mix([dsA.samples(), dsB.samples()]).fingerprint()
    mix([dsA.samples(), dsB.samples()], seed=1).fingerprint()  # must not raise


def test_fingerprint_rejects_unseeded_random_stream():
    with pytest.raises(ValueError):
        random_stream().fingerprint()
    random_stream(seed=1).fingerprint()  # must not raise


def test_fingerprint_rejects_unseeded_choice(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("dsA", make_samples(3, seed=23, prefix="a"), target_shard_size=10**9)
    repo.commit("dsB", make_samples(3, seed=24, prefix="b"), target_shard_size=10**9)
    dsA, dsB = repo.dataset("dsA"), repo.dataset("dsB")

    with pytest.raises(ValueError):
        choice([dsA.samples(), dsB.samples()]).fingerprint()
    choice([dsA.samples(), dsB.samples()], seed=1).fingerprint()  # must not raise


def test_fingerprint_rejects_unseeded_maybe(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("ds", make_samples(5, seed=25), target_shard_size=10**9)
    ds = repo.dataset("ds")

    with pytest.raises(ValueError):
        ds.samples().maybe(_identity, 0.5).fingerprint()
    ds.samples().maybe(_identity, 0.5, seed=1).fingerprint()  # must not raise


def test_fingerprint_rejects_endless_repeat_accepts_finite(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("ds", make_samples(5, seed=26), target_shard_size=10**9)
    ds = repo.dataset("ds")

    with pytest.raises(ValueError):
        ds.samples().repeat().fingerprint()
    ds.samples().repeat(2).fingerprint()  # must not raise


def test_fingerprint_rejects_lambda_transform(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("ds", make_samples(5, seed=27), target_shard_size=10**9)
    ds = repo.dataset("ds")

    with pytest.raises(ValueError):
        ds.samples().map(lambda s: s).fingerprint()


# -- derive: miss then hit -----------------------------------------------------------


def test_derive_miss_then_hit_same_version(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("src", make_samples(12, seed=30), target_shard_size=10**9)

    def build_pipe():
        return repo.dataset("src").samples().shuffle(seed=5).map(_prefixed)

    miss_msgs = []
    d1 = repo.derive("derived", build_pipe(), progress=miss_msgs.append)
    assert any("miss" in m for m in miss_msgs)

    hit_msgs = []
    d2 = repo.derive("derived", build_pipe(), progress=hit_msgs.append)
    assert any("hit" in m for m in hit_msgs)
    assert d2.version == d1.version


# -- derive: output correctness ------------------------------------------------------


def test_derive_output_matches_live_pipeline(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("src", make_samples(16, seed=31), target_shard_size=10**9)

    def build_pipe():
        return repo.dataset("src").samples().shuffle(seed=6).map(_prefixed)

    live = list(build_pipe())
    derived = repo.derive("derived_live", build_pipe())
    materialized = list(derived.samples())

    assert materialized == live


# -- derive: cross-consumer discovery, no recompute ----------------------------------


def test_derive_cross_consumer_discovery_no_recompute(tmp_path):
    repoA = build_repo(tmp_path)
    repoA.commit("src", make_samples(14, seed=32), target_shard_size=10**9)

    def build_pipe(repo):
        return repo.dataset("src").samples().shuffle(seed=7).map(_prefixed)

    expected = list(build_pipe(repoA))

    d1 = repoA.derive("derived_xc", build_pipe(repoA))
    assert list(d1.samples()) == expected

    repoB = build_repo(
        tmp_path, remote=local_remote(repoA), cache_name="cache_b", lock_name="lock_b"
    )
    # Built while "src" is still resolvable — the fingerprint bakes in its
    # version right here, before the source is destroyed below.
    pipe_for_b = build_pipe(repoB)

    repoA.delete_dataset("src")
    repoA.gc()  # actually reclaim src's now-orphaned shards from the remote
    with pytest.raises(NotFoundError):
        repoA.resolve("src")

    hit_msgs = []
    d2 = repoB.derive("derived_xc", pipe_for_b, progress=hit_msgs.append)
    assert any("hit" in m for m in hit_msgs)
    assert d2.version == d1.version

    # Streaming the derived dataset must succeed and match, even though the
    # source dataset it was built from no longer exists anywhere.
    assert list(d2.samples()) == expected


# -- derive: read efficiency ----------------------------------------------------------


def test_derive_n_gets_invariant_streaming_cold_cache(tmp_path):
    writer_repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=4, per_shard=5, size=64, seed=35)
    writer_repo.commit("src", samples, target_shard_size=64 * 5)

    pipe = writer_repo.dataset("src").samples().shuffle(seed=10).map(_prefixed)
    derived = writer_repo.derive("derived_ng", pipe, target_shard_size=64 * 5)
    num_derived_shards = len(derived.manifest.shards)
    assert derived.manifest.num_samples == len(samples)

    cold_repo = build_repo(
        tmp_path,
        budget=None,
        remote=local_remote(writer_repo),
        cache_name="cache_cold_derive",
    )
    baseline_get_bytes = local_remote(cold_repo).op_counts.get("get_bytes", 0)
    baseline_get_to_file = local_remote(cold_repo).op_counts.get("get_to_file", 0)

    ds = cold_repo.dataset("derived_ng")  # resolves ref + fetches manifest: 2 get_bytes
    got = list(ds.samples())
    assert len(got) == len(samples)
    assert (
        local_remote(cold_repo).op_counts.get("get_to_file", 0) - baseline_get_to_file
        == num_derived_shards
    )
    assert (
        local_remote(cold_repo).op_counts.get("get_bytes", 0) - baseline_get_bytes == 2
    )


# -- derive: tag distinguishes snapshots -----------------------------------------------


def test_derive_tag_produces_distinct_snapshot(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("src", make_samples(10, seed=33), target_shard_size=10**9)

    def build_pipe():
        return repo.dataset("src").samples().shuffle(seed=8).map(_prefixed)

    d_v1 = repo.derive("derived_tag", build_pipe(), tag="v1")
    d_v2 = repo.derive("derived_tag", build_pipe(), tag="v2")
    assert d_v1.version != d_v2.version

    hit_msgs_v1 = []
    d_v1_again = repo.derive(
        "derived_tag", build_pipe(), tag="v1", progress=hit_msgs_v1.append
    )
    assert d_v1_again.version == d_v1.version
    assert any("hit" in m for m in hit_msgs_v1)

    hit_msgs_v2 = []
    d_v2_again = repo.derive(
        "derived_tag", build_pipe(), tag="v2", progress=hit_msgs_v2.append
    )
    assert d_v2_again.version == d_v2.version
    assert any("hit" in m for m in hit_msgs_v2)


# -- derive: manifest meta --------------------------------------------------------------


def test_derive_manifest_meta_records_fingerprint_and_source_versions(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("src", make_samples(8, seed=34), target_shard_size=10**9)

    pipe = repo.dataset("src").samples().shuffle(seed=9).map(_prefixed)
    fp = pipe.fingerprint(tag="meta-check")
    source_versions = pipe.source_versions()

    derived = repo.derive("derived_meta", pipe, tag="meta-check")

    assert derived.manifest.meta["fingerprint"] == fp
    assert derived.manifest.meta["derived_from"] == source_versions
    assert derived.manifest.meta["tag"] == "meta-check"


def test_derive_forwards_user_meta_and_recipe(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("src", make_samples(8, seed=40), target_shard_size=10**9)

    pipe = repo.dataset("src").samples().shuffle(seed=11).map(_prefixed)
    derived = repo.derive(
        "derived_usermeta",
        pipe,
        meta={"layout": "sample-dir-v1", "fields": ["audio"]},
        recipe="resample to 16k",
    )

    # User meta rides alongside the reserved derivation keys.
    assert derived.manifest.meta["layout"] == "sample-dir-v1"
    assert derived.manifest.meta["fields"] == ["audio"]
    assert derived.manifest.recipe == "resample to 16k"
    assert derived.manifest.meta["fingerprint"] == pipe.fingerprint()
    assert derived.manifest.meta["derived_from"] == pipe.source_versions()


def test_derive_reserved_meta_keys_win_over_user_meta(tmp_path):
    repo = build_repo(tmp_path)
    repo.commit("src", make_samples(8, seed=41), target_shard_size=10**9)

    pipe = repo.dataset("src").samples().shuffle(seed=12).map(_prefixed)
    # A malicious/careless user meta cannot shadow the reserved keys.
    derived = repo.derive(
        "derived_shadow",
        pipe,
        tag="real-tag",
        meta={"fingerprint": "FAKE", "derived_from": {}, "tag": "fake-tag"},
    )

    assert derived.manifest.meta["fingerprint"] == pipe.fingerprint(tag="real-tag")
    assert derived.manifest.meta["derived_from"] == pipe.source_versions()
    assert derived.manifest.meta["tag"] == "real-tag"
