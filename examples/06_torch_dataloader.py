"""PyTorch DataLoader integration: worker-sharded, epoch-seeded streaming.

Decode/collate functions are module-level (not closures) so they survive
pickling into spawned/forked DataLoader workers. `num_workers=2` splits the
planned shard order between workers — no duplicate downloads, every sample
exactly once per epoch — and `persistent_workers=True` keeps them alive
across epochs, reseeded via `loader.dataset.set_epoch(epoch)`.
"""

# ruff: noqa: E402
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import _env

_env.setup()

import dload
from dload.torch import as_iterable_dataset

TARGET_LEN = 3 * 16_000  # 3.0 s at 16 kHz, covers every lab-tones clip
BATCH_SIZE = 16
N_EPOCHS = 2


def decode(sample: tuple[str, dict[str, bytes]]) -> torch.Tensor:
    _key, fields = sample
    wave, _sr = sf.read(io.BytesIO(fields["wav"]), dtype="float32")
    fixed = np.zeros(TARGET_LEN, dtype=np.float32)
    fixed[: min(len(wave), TARGET_LEN)] = wave[:TARGET_LEN]
    return torch.from_numpy(fixed)


def collate(items: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(items)


if __name__ == "__main__":
    repo = dload.Repository.open()
    ds = repo.dataset("lab-tones")
    print(f"dataset: {ds!r}")

    pipe = ds.samples().shuffle(seed=0).map(decode).batch(BATCH_SIZE, collate=collate)
    dataset = as_iterable_dataset(pipe)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        persistent_workers=True,
    )

    first_epoch_keys: list[torch.Tensor] = []
    for epoch in range(N_EPOCHS):
        dataset.set_epoch(epoch)
        n_samples = 0
        first_batch = None
        for batch in loader:
            if first_batch is None:
                first_batch = batch
            n_samples += batch.shape[0]
        assert n_samples == len(ds), f"expected {len(ds)} samples, got {n_samples}"
        assert first_batch is not None
        print(
            f"epoch {epoch}: {n_samples} samples (== dataset size, exact worker sharding), "
            f"first batch shape {tuple(first_batch.shape)}, dtype {first_batch.dtype}, "
            f"device {first_batch.device}"
        )
        first_epoch_keys.append(first_batch[0, :5].clone())

    differs = not torch.equal(first_epoch_keys[0], first_epoch_keys[1])
    print(f"first sample of epoch 0 vs epoch 1 differ: {differs}")
    assert differs, "set_epoch must reach persistent workers (shared-memory epoch)"
    print("per-epoch reshuffling works across persistent workers")
