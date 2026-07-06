from __future__ import annotations

import itertools

import pytest

from dload.errors import InfeasiblePipelineError
from dload.pipeline import (
    choice,
    concat,
    from_iterable,
    mix,
    random_stream,
    seeded,
    seeded_rng,
    select,
    zip_streams,
    zip_with,
)
from helpers import (
    SlowLocalRemote,
    build_repo,
    local_remote,
    make_samples,
    shard_index_of,
    shard_tagged_samples,
)

# -- sequential order ----------------------------------------------------------


def test_sequential_order_matches_commit_order(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=5, per_shard=4, size=64, seed=1)
    repo.commit("ds", samples, target_shard_size=64 * 4)
    ds = repo.dataset("ds")

    got = [k for k, _ in ds.samples()]
    assert got == [k for k, _ in samples]


# -- windowed shuffle ------------------------------------------------------------


def test_shuffle_windowed_same_seed_epoch_identical_order(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=6, per_shard=8, size=64, seed=2)
    repo.commit("ds", samples, target_shard_size=64 * 8)
    ds = repo.dataset("ds")

    def fresh_order():
        return [k for k, _ in ds.samples(prefetch=3).shuffle(buffer_size=16, seed=7)]

    order_a = fresh_order()
    order_b = fresh_order()
    assert order_a == order_b
    assert sorted(order_a) == sorted(k for k, _ in samples)


def test_shuffle_windowed_successive_iterations_differ_but_permute(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=6, per_shard=8, size=64, seed=3)
    repo.commit("ds", samples, target_shard_size=64 * 8)
    ds = repo.dataset("ds")
    all_keys = sorted(k for k, _ in samples)

    pipe = ds.samples(prefetch=3).shuffle(buffer_size=16, seed=7)
    epoch0 = [k for k, _ in pipe]
    epoch1 = [k for k, _ in pipe]  # re-iterating the same pipeline advances the epoch

    assert sorted(epoch0) == all_keys
    assert sorted(epoch1) == all_keys
    assert epoch0 != epoch1


def test_shuffle_windowed_different_seeds_differ(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=6, per_shard=8, size=64, seed=4)
    repo.commit("ds", samples, target_shard_size=64 * 8)
    ds = repo.dataset("ds")

    order_seed1 = [k for k, _ in ds.samples(prefetch=3).shuffle(buffer_size=16, seed=1)]
    order_seed2 = [k for k, _ in ds.samples(prefetch=3).shuffle(buffer_size=16, seed=2)]
    assert order_seed1 != order_seed2


def test_shuffle_windowed_buffer_size_one_still_permutation(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=6, per_shard=8, size=64, seed=5)
    repo.commit("ds", samples, target_shard_size=64 * 8)
    ds = repo.dataset("ds")

    order = [k for k, _ in ds.samples(prefetch=3).shuffle(buffer_size=1, seed=1)]
    assert sorted(order) == sorted(k for k, _ in samples)


# -- full shuffle ----------------------------------------------------------------


def test_shuffle_full_true_permutation_and_deterministic(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=4, per_shard=5, size=64, seed=6)
    repo.commit("ds", samples, target_shard_size=64 * 5)
    ds = repo.dataset("ds")
    all_keys = sorted(k for k, _ in samples)

    def run():
        return [k for k, _ in ds.samples().shuffle(full=True, seed=99)]

    order1 = run()
    order2 = run()
    assert sorted(order1) == all_keys
    assert order1 == order2  # deterministic: same seed, both at epoch 0
    assert order1 != [k for k, _ in samples]


def test_shuffle_full_above_map_raises_infeasible(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = make_samples(5, size=64, seed=7)
    repo.commit("ds", samples, target_shard_size=10**9)
    ds = repo.dataset("ds")

    pipe = ds.samples().map(lambda x: x).shuffle(full=True)
    with pytest.raises(InfeasiblePipelineError):
        pipe.check()
    with pytest.raises(InfeasiblePipelineError):
        list(pipe)


def test_shuffle_full_infeasible_on_small_budget(tmp_path):
    writer_repo = build_repo(tmp_path, budget=None)
    samples = make_samples(20, size=1024, seed=8)
    manifest = writer_repo.commit(
        "ds", samples, target_shard_size=10**9
    )  # everything in 1 shard

    small_budget = manifest.total_bytes // 2
    reader_repo = build_repo(
        tmp_path,
        budget=small_budget,
        remote=local_remote(writer_repo),
        cache_name="cache_tiny",
    )
    ds = reader_repo.dataset("ds")
    pipe = ds.samples().shuffle(full=True, seed=1)

    report = pipe.check()
    assert not report.ok
    assert "infeasible" in report.message.lower()

    with pytest.raises(InfeasiblePipelineError):
        list(pipe)


# -- map / filter / batch / take / repeat / concat --------------------------------


def test_map_and_filter_semantics(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = make_samples(10, size=64, seed=9)
    repo.commit("ds", samples, target_shard_size=10**9)
    ds = repo.dataset("ds")

    mapped = [k for k, _ in ds.samples().map(lambda kv: (kv[0].upper(), kv[1]))]
    assert mapped == [k.upper() for k, _ in samples]

    filtered = [k for k, _ in ds.samples().filter(lambda kv: int(kv[0][1:]) % 2 == 0)]
    assert filtered == [k for k, _ in samples if int(k[1:]) % 2 == 0]


def test_batch_drop_last_and_collate(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = make_samples(10, size=64, seed=10)
    repo.commit("ds", samples, target_shard_size=10**9)
    ds = repo.dataset("ds")

    batches_keep = list(ds.samples().batch(3))
    assert [len(b) for b in batches_keep] == [3, 3, 3, 1]

    batches_drop = list(ds.samples().batch(3, drop_last=True))
    assert [len(b) for b in batches_drop] == [3, 3, 3]

    collated = list(ds.samples().batch(4, collate=lambda items: [k for k, _ in items]))
    assert collated[0] == [k for k, _ in samples[:4]]
    assert collated[-1] == [k for k, _ in samples[-2:]]  # last, partial batch


def test_take(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = make_samples(10, size=64, seed=11)
    repo.commit("ds", samples, target_shard_size=10**9)
    ds = repo.dataset("ds")

    taken = [k for k, _ in ds.samples().take(4)]
    assert taken == [k for k, _ in samples[:4]]


def test_repeat_times_and_none_with_take(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = make_samples(5, size=64, seed=12)
    repo.commit("ds", samples, target_shard_size=10**9)
    ds = repo.dataset("ds")

    repeated = [k for k, _ in ds.samples().repeat(3)]
    assert repeated == [k for k, _ in samples] * 3

    forever = [k for k, _ in ds.samples().repeat(None).take(12)]
    assert forever == ([k for k, _ in samples] * 3)[:12]


def test_concat_chains_pipelines_in_order(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    a = make_samples(4, size=64, seed=13, prefix="a")
    b = make_samples(3, size=64, seed=14, prefix="b")
    repo.commit("dsA", a, target_shard_size=10**9)
    repo.commit("dsB", b, target_shard_size=10**9)
    dsA, dsB = repo.dataset("dsA"), repo.dataset("dsB")

    combined = [k for k, _ in concat([dsA.samples(), dsB.samples()])]
    assert combined == [k for k, _ in a] + [k for k, _ in b]


# -- mix -------------------------------------------------------------------------


def test_mix_weight_ratio_approx(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    a = make_samples(50, size=64, seed=15, prefix="A")
    b = make_samples(50, size=64, seed=16, prefix="B")
    repo.commit("dsA", a, target_shard_size=10**9)
    repo.commit("dsB", b, target_shard_size=10**9)
    dsA, dsB = repo.dataset("dsA"), repo.dataset("dsB")

    pipe = mix(
        [dsA.samples().repeat(None), dsB.samples().repeat(None)],
        weights=[3, 1],
        seed=42,
    ).take(4000)
    got = [k for k, _ in pipe]
    assert len(got) == 4000
    frac_a = sum(1 for k in got if k.startswith("A")) / len(got)
    assert abs(frac_a - 0.75) < 0.08  # wide tolerance; seeded so this is deterministic


def test_mix_until_any_stops_at_first_exhaustion(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    small = make_samples(3, size=64, seed=17, prefix="S")
    big = make_samples(100, size=64, seed=18, prefix="B")
    repo.commit("dsSmall", small, target_shard_size=10**9)
    repo.commit("dsBig", big, target_shard_size=10**9)
    dsSmall, dsBig = repo.dataset("dsSmall"), repo.dataset("dsBig")

    def run():
        pipe = mix(
            [dsSmall.samples(), dsBig.samples()], weights=[20, 1], seed=1, until="any"
        )
        return [k for k, _ in pipe]

    got1 = run()
    got2 = run()
    assert got1 == got2  # deterministic given the seed
    assert 0 < len(got1) < 3 + 100  # stopped early, before fully draining both


def test_mix_until_all_drains_everything(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    a = make_samples(7, size=64, seed=19, prefix="A")
    b = make_samples(11, size=64, seed=20, prefix="B")
    repo.commit("dsA", a, target_shard_size=10**9)
    repo.commit("dsB", b, target_shard_size=10**9)
    dsA, dsB = repo.dataset("dsA"), repo.dataset("dsB")

    pipe = mix([dsA.samples(), dsB.samples()], weights=[1, 5], seed=2, until="all")
    got = sorted(k for k, _ in pipe)
    expected = sorted([k for k, _ in a] + [k for k, _ in b])
    assert got == expected


def test_mix_weights_length_mismatch_raises(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    a = make_samples(2, seed=21, prefix="A")
    b = make_samples(2, seed=22, prefix="B")
    repo.commit("dsA", a, target_shard_size=10**9)
    repo.commit("dsB", b, target_shard_size=10**9)
    dsA, dsB = repo.dataset("dsA"), repo.dataset("dsB")

    with pytest.raises(ValueError):
        mix([dsA.samples(), dsB.samples()], weights=[1, 2, 3])


# -- sliding window ----------------------------------------------------------------


def test_sliding_window_streams_fully_within_budget(tmp_path):
    writer_repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=20, per_shard=5, size=64, seed=23)
    manifest = writer_repo.commit("ds", samples, target_shard_size=64 * 5)
    assert len(manifest.shards) == 20

    max_shard_size = max(s.size for s in manifest.shards)
    budget = 4 * max_shard_size

    reader_repo = build_repo(
        tmp_path,
        budget=budget,
        remote=local_remote(writer_repo),
        cache_name="cache_window",
    )
    ds = reader_repo.dataset("ds")
    pipe = ds.samples(prefetch=4)

    assert pipe.check().ok

    got = list(pipe)
    assert len(got) == 100
    assert {k for k, _ in got} == {k for k, _ in samples}
    assert reader_repo.cache.used_bytes() <= budget

    reader_repo.cache.clear()
    assert reader_repo.cache.used_bytes() == 0  # nothing left pinned


# -- worker sharding -----------------------------------------------------------------


def test_worker_sharding_partitions_dataset_sequential_and_shuffled(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=9, per_shard=4, size=64, seed=24)
    repo.commit("ds", samples, target_shard_size=64 * 4)
    ds = repo.dataset("ds")
    all_keys = {k for k, _ in samples}
    num_workers = 3

    pipe_builders = [
        lambda: ds.samples(prefetch=2),
        # buffer_size=1 makes the sample-level shuffle buffer an identity
        # pass-through (verified separately below), so shard boundaries stay
        # intact in the output and we can still recover shard-visit order
        # from it, while the shard-order shuffle itself is exercised.
        lambda: ds.samples(prefetch=2).shuffle(buffer_size=1, seed=77),
    ]
    for build_pipe in pipe_builders:
        pipe = build_pipe()
        per_worker_keys = []
        per_worker_shard_order = []
        for w in range(num_workers):
            got = list(pipe._iterate(worker=w, num_workers=num_workers, epoch=0))
            keys = [k for k, _ in got]
            per_worker_keys.append(set(keys))
            seen = []
            for k in keys:
                si = shard_index_of(k)
                if si not in seen:
                    seen.append(si)
            per_worker_shard_order.append(seen)

        union = set().union(*per_worker_keys)
        assert union == all_keys
        for i in range(num_workers):
            for j in range(i + 1, num_workers):
                assert per_worker_keys[i].isdisjoint(per_worker_keys[j])

        # consistency: interleaving the per-worker shard-visit orders
        # round-robin must reproduce the single-worker (full) order for the
        # same epoch, proving every worker planned off the same shuffle.
        full_pipe = build_pipe()
        full_got = list(full_pipe._iterate(worker=0, num_workers=1, epoch=0))
        full_order = []
        for k, _ in full_got:
            si = shard_index_of(k)
            if si not in full_order:
                full_order.append(si)

        total_shards = sum(len(o) for o in per_worker_shard_order)
        recombined = []
        idx = [0] * num_workers
        w = 0
        while len(recombined) < total_shards:
            if idx[w] < len(per_worker_shard_order[w]):
                recombined.append(per_worker_shard_order[w][idx[w]])
                idx[w] += 1
            w = (w + 1) % num_workers
        assert recombined == full_order


# -- explicit epoch --------------------------------------------------------------


def test_explicit_epoch_reproduces_order(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=5, per_shard=6, size=64, seed=25)
    repo.commit("ds", samples, target_shard_size=64 * 6)
    ds = repo.dataset("ds")
    pipe = ds.samples(prefetch=2).shuffle(buffer_size=10, seed=5)

    order1 = [k for k, _ in pipe._iterate(worker=0, num_workers=1, epoch=3)]
    order2 = [k for k, _ in pipe._iterate(worker=0, num_workers=1, epoch=3)]
    assert order1 == order2
    assert sorted(order1) == sorted(k for k, _ in samples)


# -- slow remote / prefetch correctness --------------------------------------------


def test_slow_remote_still_yields_all_samples(tmp_path):
    slow_remote = SlowLocalRemote(tmp_path / "remote", delay=0.02)
    writer_repo = build_repo(
        tmp_path, budget=None, remote=slow_remote, cache_name="cache_writer"
    )
    samples = shard_tagged_samples(num_shards=8, per_shard=5, size=64, seed=26)
    writer_repo.commit("ds", samples, target_shard_size=64 * 5)

    reader_repo = build_repo(
        tmp_path, budget=None, remote=slow_remote, cache_name="cache_reader"
    )
    ds = reader_repo.dataset("ds")
    got = list(ds.samples(prefetch=3))
    assert sorted(k for k, _ in got) == sorted(k for k, _ in samples)


# -- generator cleanup --------------------------------------------------------------


def test_generator_close_mid_shard_releases_pins(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=6, per_shard=5, size=64, seed=27)
    repo.commit("ds", samples, target_shard_size=64 * 5)
    ds = repo.dataset("ds")

    it = iter(ds.samples(prefetch=2))
    for _ in range(3):
        next(it)
    it.close()
    del it

    repo.cache.clear()
    assert repo.cache.used_bytes() == 0  # no leaked pins from the interrupted iteration


# -- regressions from code review ---------------------------------------------------


def test_unseeded_shuffle_workers_still_partition_exactly(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=6, per_shard=4, size=64, seed=30)
    repo.commit("ds", samples, target_shard_size=64 * 4)
    ds = repo.dataset("ds")

    pipe = ds.samples().shuffle(buffer_size=8)  # no seed: auto-seed must be shared
    per_worker = [
        [k for k, _ in pipe._iterate(worker=i, num_workers=3, epoch=0)]
        for i in range(3)
    ]
    union = [k for keys in per_worker for k in keys]
    assert sorted(union) == sorted(k for k, _ in samples)
    assert len(union) == len(set(union))


def test_full_shuffle_workers_stripe_one_global_permutation(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=4, per_shard=5, size=64, seed=31)
    repo.commit("ds", samples, target_shard_size=64 * 5)
    ds = repo.dataset("ds")

    def orders(num_workers):
        pipe = ds.samples().shuffle(full=True, seed=5)
        return [
            [k for k, _ in pipe._iterate(worker=i, num_workers=num_workers, epoch=0)]
            for i in range(num_workers)
        ]

    (single,) = orders(1)
    w0, w1 = orders(2)

    union = w0 + w1
    assert sorted(union) == sorted(k for k, _ in samples)
    assert len(union) == len(set(union))
    # the two worker streams are stripes of the same global permutation
    assert w0 == single[0::2]
    assert w1 == single[1::2]


def test_shuffle_buffer_size_zero_is_passthrough(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=3, per_shard=4, size=64, seed=32)
    repo.commit("ds", samples, target_shard_size=64 * 4)
    ds = repo.dataset("ds")

    order = [k for k, _ in ds.samples().shuffle(buffer_size=0, seed=1)]
    assert sorted(order) == sorted(k for k, _ in samples)


# -- combinators: fundamental core ---------------------------------------------------


def test_close_propagates_through_transform_chain(tmp_path):
    # Early .close() must release pins even through a stack of rebased
    # combinators, not just on a bare SourceNode (which
    # test_generator_close_mid_shard_releases_pins covers).
    repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=6, per_shard=5, size=64, seed=40)
    repo.commit("ds", samples, target_shard_size=64 * 5)
    ds = repo.dataset("ds")

    pipe = ds.samples(prefetch=2).map(lambda s: s).filter(lambda s: True).batch(4)
    it = iter(pipe)
    next(it)
    it.close()
    del it

    repo.cache.clear()
    assert repo.cache.used_bytes() == 0


def test_transform_children_all_registered_in_walk(tmp_path):
    # Every node reachable via .children must land in node_ids — otherwise
    # ctx.rng_for(KeyError) or, worse, an unplanned SourceNode. Exercise a
    # TransformNode with a shard-backed branch and a RandomNode branch.
    repo = build_repo(tmp_path, budget=None)
    samples = make_samples(8, size=64, seed=41)
    repo.commit("ds", samples, target_shard_size=10**9)
    ds = repo.dataset("ds")

    pipe = zip_with(lambda s, r: (s[0], r), ds.samples(), random_stream(seed=3))
    got = list(pipe)
    assert [k for k, _ in got] == [k for k, _ in samples]
    assert all(0.0 <= r < 1.0 for _, r in got)


def test_from_iterable_fresh_per_epoch_and_shard_opt_in():
    pipe = from_iterable(lambda: range(5))
    assert list(pipe) == list(range(5))
    assert list(pipe) == list(range(5))  # factory re-called: re-iterable

    # a plain container works too
    assert list(from_iterable([3, 1, 4])) == [3, 1, 4]

    # default: replicated per worker; shard=True: interleaved slices
    replicated = from_iterable(lambda: range(6))
    per_worker = [
        list(replicated._iterate(worker=w, num_workers=2, epoch=0)) for w in range(2)
    ]
    assert per_worker[0] == per_worker[1] == list(range(6))

    sharded = from_iterable(lambda: range(6), shard=True)
    per_worker = [
        list(sharded._iterate(worker=w, num_workers=2, epoch=0)) for w in range(2)
    ]
    assert per_worker[0] == [0, 2, 4]
    assert per_worker[1] == [1, 3, 5]


def test_from_iterable_invisible_to_feasibility(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = make_samples(4, size=64, seed=42)
    repo.commit("ds", samples, target_shard_size=10**9)
    ds = repo.dataset("ds")

    # alone: nothing to plan, nothing to report
    report = from_iterable(lambda: range(3)).check()
    assert report.ok
    assert report.lines == []

    # combined: only the shard-backed source appears in the report
    combined = zip_streams(ds.samples(), from_iterable(lambda: itertools.count()))
    report = combined.check()
    assert report.ok
    assert sum("ds@" in line for line in report.lines) == 1


def test_random_stream_deterministic_per_seed_epoch_and_diverges_per_worker():
    a = list(random_stream(seed=5).take(20))
    b = list(random_stream(seed=5).take(20))
    assert a == b  # fresh pipelines, both epoch 0

    pipe = random_stream(seed=5).take(20)
    e0 = list(pipe)
    e1 = list(pipe)  # re-iteration advances the epoch
    assert e0 != e1

    assert list(random_stream(seed=6).take(20)) != a

    # Per-worker divergence is INTENTIONAL: rng_for salts with the worker
    # index. Only shard-order derivation must stay worker-free (the
    # AGENTS.md worker-sharding contract); RandomNode never enters it.
    # Do not "fix" this into worker-uniform draws.
    pipe = random_stream(seed=5).take(20)
    w0 = list(pipe._iterate(worker=0, num_workers=2, epoch=0))
    w1 = list(pipe._iterate(worker=1, num_workers=2, epoch=0))
    assert w0 != w1


def test_select_follows_index_stream_and_advances_only_chosen():
    idx = from_iterable([0, 1, 1, 0, 2])
    got = list(
        select(idx, from_iterable("AA"), from_iterable("BB"), from_iterable("CC"))
    )
    assert got == list("ABBAC")


def test_select_stops_on_index_or_selected_exhaustion():
    # index stream ends first
    assert list(select(from_iterable([0, 0]), from_iterable("ABCD"))) == ["A", "B"]
    # selected stream ends first: stop cleanly, mid-stream
    got = list(
        select(from_iterable([0, 1, 0, 1]), from_iterable("AB"), from_iterable("X"))
    )
    assert got == ["A", "X", "B"]


def test_select_validates_indices_and_arity():
    with pytest.raises(ValueError):
        select(from_iterable([0]))
    with pytest.raises(ValueError, match="out of range"):
        list(select(from_iterable([2]), from_iterable("A"), from_iterable("B")))


def test_scan_folds_state_and_composes_with_flat_map():
    nums = from_iterable(lambda: range(1, 6))
    assert list(nums.scan(lambda s, x: s + x, 0)) == [1, 3, 6, 10, 15]

    # the stateful-expansion idiom: fn returns (carry, emitted),
    # flat_map spills the emissions — here, pairwise packing
    def pack_pairs(state, x):
        carry, _ = state
        carry = carry + [x]
        if len(carry) == 2:
            return ([], [tuple(carry)])
        return (carry, [])

    packed = list(
        from_iterable(lambda: range(5))
        .scan(pack_pairs, ([], []))
        .flat_map(lambda s: s[1])
    )
    assert packed == [(0, 1), (2, 3)]


def test_flat_map_expands_and_drops():
    got = list(from_iterable(lambda: range(4)).flat_map(lambda x: [x] * x))
    assert got == [1, 2, 2, 3, 3, 3]


def test_window_tumbling_sliding_and_strided():
    src = lambda: from_iterable(lambda: range(7))  # noqa: E731
    assert list(src().window(3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(src().window(3, drop_last=True)) == [[0, 1, 2], [3, 4, 5]]
    assert list(src().window(3, 1)) == [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
    ]
    assert list(src().window(2, 3)) == [[0, 1], [3, 4], [6]]
    # shorter than one window: partial still yielded unless dropped
    assert list(from_iterable([1, 2]).window(5)) == [[1, 2]]
    assert list(from_iterable([1, 2]).window(5, drop_last=True)) == []


def test_zip_with_pairs_and_stops_at_shortest(tmp_path):
    repo = build_repo(tmp_path, budget=None)
    samples = make_samples(6, size=64, seed=43)
    repo.commit("ds", samples, target_shard_size=10**9)
    ds = repo.dataset("ds")

    pairs = list(zip_streams(ds.samples(), from_iterable(lambda: range(3))))
    assert len(pairs) == 3  # shortest stream wins
    assert [i for _, i in pairs] == [0, 1, 2]

    summed = list(
        zip_with(
            lambda a, b: a + b, from_iterable([1, 2, 3]), from_iterable([10, 20, 30])
        )
    )
    assert summed == [11, 22, 33]


def test_star_map_and_through():
    tupled = zip_streams(from_iterable("ab"), from_iterable([1, 2]))
    assert list(tupled.star_map(lambda c, n: c * n)) == ["a", "bb"]

    def dedupe(it):
        seen = set()
        for x in it:
            if x not in seen:
                seen.add(x)
                yield x

    assert list(from_iterable([1, 1, 2, 1, 3]).through(dedupe)) == [1, 2, 3]


# -- combinators: choice / maybe ------------------------------------------------------


def test_choice_static_p_ratio_and_determinism():
    pipe = choice(
        [
            from_iterable(lambda: itertools.repeat("a")),
            from_iterable(lambda: itertools.repeat("b")),
        ],
        p=0.7,
        seed=42,
    ).take(4000)
    got = list(pipe)
    frac_a = got.count("a") / len(got)
    assert abs(frac_a - 0.7) < 0.05

    again = list(
        choice(
            [
                from_iterable(lambda: itertools.repeat("a")),
                from_iterable(lambda: itertools.repeat("b")),
            ],
            p=0.7,
            seed=42,
        ).take(4000)
    )
    assert got == again  # same seed, same DAG shape, both epoch 0


def test_choice_weights_for_n_streams():
    got = list(
        choice(
            [
                from_iterable(lambda: itertools.repeat("a")),
                from_iterable(lambda: itertools.repeat("b")),
                from_iterable(lambda: itertools.repeat("c")),
            ],
            p=[1, 1, 2],
            seed=7,
        ).take(4000)
    )
    assert abs(got.count("c") / len(got) - 0.5) < 0.05


def test_choice_p_from_a_stream_tracks_schedule():
    # THE new capability: selection probability varying over iteration,
    # coming from a stream. A 0 -> 1 ramp must shift picks from stream 1
    # to stream 0.
    n = 8000
    ramp = from_iterable(lambda: (i / n for i in range(n)))
    got = list(
        choice(
            [
                from_iterable(lambda: itertools.repeat(0)),
                from_iterable(lambda: itertools.repeat(1)),
            ],
            p=ramp,
            seed=7,
        )
    )
    assert len(got) == n  # stops with the ramp (control) stream
    head, tail = got[: n // 4], got[-n // 4 :]
    assert head.count(0) / len(head) < 0.2
    assert tail.count(0) / len(tail) > 0.8


def test_choice_validates_arity_and_weights():
    a = from_iterable("a")
    b = from_iterable("b")
    c = from_iterable("c")
    with pytest.raises(ValueError):
        choice([a])
    with pytest.raises(ValueError):
        choice([a, b, c], p=0.5)  # scalar p only defined for two streams
    with pytest.raises(ValueError):
        choice([a, b], p=[1, 2, 3])
    with pytest.raises(ValueError):
        choice([a, b], p=[0, 0])


def test_maybe_applies_to_p_fraction_single_pass(tmp_path):
    # maybe must be a single pass over the upstream (zip-based, not
    # choice-based): every upstream item appears exactly once, in order.
    repo = build_repo(tmp_path, budget=None)
    samples = make_samples(30, size=64, seed=44)
    repo.commit("ds", samples, target_shard_size=10**9)
    ds = repo.dataset("ds")

    got = list(ds.samples().maybe(lambda s: (s[0].upper(), s[1]), p=0.5, seed=1))
    assert [k.lower() for k, _ in got] == [k for k, _ in samples]
    upper = sum(1 for k, _ in got if k.isupper())
    assert 0 < upper < len(got)

    # p as a stream: p=0 stream -> never applied
    never = list(
        from_iterable(lambda: range(100)).maybe(
            lambda x: -x, p=from_iterable(lambda: itertools.repeat(0.0)), seed=2
        )
    )
    assert never == list(range(100))


# -- combinators: worker interactions -------------------------------------------------


def test_zip_replicated_branch_restarts_per_worker(tmp_path):
    # Documented caveat, kept as a contract: a replicated from_iterable
    # branch restarts at position 0 in EVERY worker, so zip pairing is
    # worker-count dependent. Fine for generative/random side streams;
    # aligned 1:1 side data belongs inside the sample.
    repo = build_repo(tmp_path, budget=None)
    samples = shard_tagged_samples(num_shards=4, per_shard=3, size=64, seed=45)
    repo.commit("ds", samples, target_shard_size=64 * 3)
    ds = repo.dataset("ds")

    pipe = zip_streams(ds.samples(prefetch=2), from_iterable(lambda: itertools.count()))
    for w in range(2):
        got = list(pipe._iterate(worker=w, num_workers=2, epoch=0))
        side = [i for _, i in got]
        assert side == list(range(len(got)))  # starts at 0 in every worker


def test_seeded_dag_determinism_across_rebuilds(tmp_path):
    # Rebuilding the identical DAG must give the identical seeded stream:
    # rng_for salts with node position in the DAG, so this guards the
    # rebased operators preserving node_ids assignment (the examples/04
    # checksum property, promoted to a unit test).
    repo = build_repo(tmp_path, budget=None)
    a = make_samples(20, size=64, seed=46, prefix="A")
    b = make_samples(20, size=64, seed=47, prefix="B")
    repo.commit("dsA", a, target_shard_size=10**9)
    repo.commit("dsB", b, target_shard_size=10**9)

    def build():
        dsA, dsB = repo.dataset("dsA"), repo.dataset("dsB")
        return (
            mix(
                [
                    dsA.samples().shuffle(4, seed=0).repeat(None),
                    dsB.samples().repeat(None),
                ],
                weights=[2, 1],
                seed=3,
            )
            .map(lambda s: s[0])
            .take(60)
        )

    assert list(build()) == list(build())


# -- seeded helpers --------------------------------------------------------------------


def test_seeded_is_deterministic_and_collision_resistant():
    assert seeded("k1", "crop") == seeded("k1", "crop")
    assert seeded("k1", "crop") != seeded("k1", "snr")
    # repr-based hashing: no "a|b" vs "a","|b" joining collisions
    assert seeded("a|b", "c") != seeded("a", "b|c")
    assert seeded_rng("k1", "gain").random() == seeded_rng("k1", "gain").random()
