"""Optional PyTorch integration. Importing this module requires torch;
the rest of dload stays torch-free.

    from dload.torch import as_iterable_dataset

    loader = torch.utils.data.DataLoader(
        as_iterable_dataset(pipe),
        batch_size=None,        # batch in the pipeline, or here — your call
        num_workers=4,
        persistent_workers=True,
    )
    for epoch in range(n):
        loader.dataset.set_epoch(epoch)   # keeps shuffling deterministic
        for batch in loader: ...

Worker sharding: each DataLoader worker receives every k-th shard of the
planned (per-epoch, seeded) shard order, so workers never download the
same shard twice and the union of workers covers the dataset exactly once
per epoch. Note that the cache budget is shared by all workers on the
machine; the feasibility check runs per worker, so keep
`num_workers × window` within budget.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Generator
from typing import Any

import torch.utils.data as tud

from .pipeline import Pipeline


class PipelineDataset(tud.IterableDataset[Any]):
    """A Pipeline as a torch IterableDataset with worker sharding.

    Picklable (spawn-safe) as long as the pipeline's map/filter/collate
    functions are module-level, not lambdas or closures.

    The epoch lives in shared memory: persistent DataLoader workers hold a
    long-lived copy of this object, so a plain attribute set from the main
    process would never reach them and `set_epoch` would silently stop
    reshuffling. Call `set_epoch` *before* starting an epoch's iteration.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipe = pipeline
        self._epoch = mp.Value("q", 0)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = epoch

    def __iter__(self) -> Generator[Any, None, None]:
        info = tud.get_worker_info()
        worker, num_workers = (info.id, info.num_workers) if info else (0, 1)
        return self._pipe._iterate(
            worker=worker, num_workers=num_workers, epoch=int(self._epoch.value)
        )


def as_iterable_dataset(pipeline: Pipeline) -> PipelineDataset:
    return PipelineDataset(pipeline)
