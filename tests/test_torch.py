"""End-to-end tests for dload.torch.as_iterable_dataset — DataLoader with
zero and multiple (real multiprocessing) workers, epoch determinism and
persistent_workers, all without touching the network.

decode()/collate() are module-level functions (not closures/lambdas) so they
pickle cleanly when the DataLoader spawns worker processes.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud

from dload.cache import ShardCache
from dload.pipeline import Pipeline
from dload.remote import LocalRemote
from dload.repo import Dataset, Repository
from dload.torch import as_iterable_dataset

N_SAMPLES = 60
SAMPLE_FLOATS = 8  # "x" field = 8 float32 = 32 bytes
TARGET_SHARD_SIZE = 170  # bytes; yields roughly a dozen small shards
BATCH_SIZE = 8

# The DataLoader's multiprocessing_context is left at the platform default
# ("fork" on Linux), which is what real training jobs use. Under fork, the
# child process shares memory (no pickling) — but the pipeline must *also*
# be picklable, since other start methods (spawn/forkserver) do serialize
# it; see test_pipeline_and_repository_are_picklable below, and the
# decode/collate functions are kept module-level (not closures) so they
# would survive that too.


# -- module-level helpers (must be top-level to stay picklable) --------------


def make_sample(i: int) -> tuple[str, dict[str, bytes]]:
    rng = np.random.default_rng(i)
    x = rng.standard_normal(SAMPLE_FLOATS).astype(np.float32).tobytes()
    return f"sample-{i:03d}", {"x": x, "idx": str(i).encode()}


def decode(sample: tuple[str, dict[str, bytes]]) -> dict:
    _key, fields = sample
    x = torch.frombuffer(bytearray(fields["x"]), dtype=torch.float32)
    info = tud.get_worker_info()
    worker = info.id if info is not None else -1
    return {"x": x, "idx": fields["idx"].decode(), "worker": worker}


def collate(items: list[dict]) -> dict:
    return {
        "x": torch.stack([it["x"] for it in items]),
        "idx": [it["idx"] for it in items],
    }


def build_dataset(tmp_path: Path, *, n: int = N_SAMPLES, name: str = "ds") -> Dataset:
    remote = LocalRemote(tmp_path / "remote")
    cache = ShardCache(tmp_path / "cache", None)
    repo = Repository(remote, cache, lock_path=tmp_path / "dload.lock")
    repo.commit(
        name, (make_sample(i) for i in range(n)), target_shard_size=TARGET_SHARD_SIZE
    )
    return repo.dataset(name)


def idx_multiset(items) -> list[int]:
    return sorted(int(it["idx"]) for it in items)


# -- num_workers=0 -------------------------------------------------------------


def test_num_workers_zero_batches_cover_dataset(tmp_path):
    dataset = build_dataset(tmp_path)
    pipe = dataset.samples().map(decode).batch(BATCH_SIZE, collate=collate)
    loader = tud.DataLoader(as_iterable_dataset(pipe), batch_size=None, num_workers=0)

    batches = list(loader)
    sizes = [batch["x"].shape[0] for batch in batches]
    assert sum(sizes) == N_SAMPLES
    assert sizes[:-1] == [BATCH_SIZE] * (len(sizes) - 1)
    assert sizes[-1] == N_SAMPLES % BATCH_SIZE or sizes[-1] == BATCH_SIZE
    for batch in batches:
        assert batch["x"].shape[1] == SAMPLE_FLOATS
        assert batch["x"].dtype == torch.float32
        assert len(batch["idx"]) == batch["x"].shape[0]

    all_idx = [i for batch in batches for i in batch["idx"]]
    assert sorted(int(i) for i in all_idx) == list(range(N_SAMPLES))


# -- num_workers=2 (real multiprocessing) ---------------------------------------


def test_num_workers_two_covers_dataset_exactly_once(tmp_path):
    dataset = build_dataset(tmp_path)
    pipe = dataset.samples().map(decode)
    loader = tud.DataLoader(
        as_iterable_dataset(pipe),
        batch_size=None,
        num_workers=2,
    )

    items = list(loader)
    assert len(items) == N_SAMPLES
    idxs = [it["idx"] for it in items]
    assert len(set(idxs)) == N_SAMPLES  # no duplicates
    assert sorted(int(i) for i in idxs) == list(range(N_SAMPLES))  # exact coverage
    assert {it["worker"] for it in items} <= {0, 1}


# -- set_epoch determinism ------------------------------------------------------


def test_set_epoch_determinism(tmp_path):
    dataset = build_dataset(tmp_path)
    pipe = dataset.samples().shuffle(buffer_size=16, seed=0).map(decode)
    iterable_dataset = as_iterable_dataset(pipe)
    loader = tud.DataLoader(
        iterable_dataset,
        batch_size=None,
        num_workers=2,
    )

    def run(epoch: int):
        iterable_dataset.set_epoch(epoch)
        items = list(loader)
        by_worker: dict[int, list[str]] = {}
        for it in items:
            by_worker.setdefault(it["worker"], []).append(it["idx"])
        return items, by_worker

    items0a, by_worker0a = run(0)
    items0b, by_worker0b = run(0)
    items1, by_worker1 = run(1)

    # exact coverage every epoch
    assert idx_multiset(items0a) == list(range(N_SAMPLES))
    assert idx_multiset(items1) == list(range(N_SAMPLES))

    # epoch 0 run twice: same global multiset and same per-worker order
    assert idx_multiset(items0a) == idx_multiset(items0b)
    assert by_worker0a == by_worker0b

    # epoch 1: different per-worker assignment/order than epoch 0
    assert by_worker0a != by_worker1


# -- persistent_workers, multiple epochs, no hang -------------------------------


def test_persistent_workers_two_epochs(tmp_path):
    dataset = build_dataset(tmp_path)
    pipe = dataset.samples().shuffle(buffer_size=16, seed=0).map(decode)
    iterable_dataset = as_iterable_dataset(pipe)
    loader = tud.DataLoader(
        iterable_dataset,
        batch_size=None,
        num_workers=2,
        persistent_workers=True,
    )

    orders = []
    for epoch in range(2):
        iterable_dataset.set_epoch(epoch)
        items = list(loader)
        assert idx_multiset(items) == list(range(N_SAMPLES))
        orders.append([it["idx"] for it in items])

    # set_epoch must reach the long-lived workers (shared-memory epoch):
    # with a seeded shuffle, epoch 1 must differ from epoch 0.
    assert orders[0] != orders[1]

    # and re-running epoch 1 must reproduce it exactly
    iterable_dataset.set_epoch(1)
    items = list(loader)
    assert [it["idx"] for it in items] == orders[1]


# -- picklability ----------------------------------------------------------------


def test_pipeline_and_repository_are_picklable(tmp_path):
    """DataLoader workers under spawn/forkserver start methods serialize the
    dataset (and therefore the Pipeline, Dataset, Repository, ShardCache and
    LocalRemote it closes over) before handing it to the child process. Fork
    (this platform's default) does not need this, but we verify it directly
    so the adapter also works under those other start methods."""
    dataset = build_dataset(tmp_path)
    pipe = dataset.samples().map(decode).batch(BATCH_SIZE, collate=collate)

    restored: Pipeline = pickle.loads(pickle.dumps(pipe))

    batches = list(restored)
    all_idx = [i for batch in batches for i in batch["idx"]]
    assert sorted(int(i) for i in all_idx) == list(range(N_SAMPLES))
