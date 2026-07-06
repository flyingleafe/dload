"""The ephemeral-machine story: a small cache streaming a bigger dataset.

A sliding-window prefetcher keeps only a few shards resident at a time, so
a machine with a tiny disk can stream a dataset many times its cache
budget — and the planner refuses upfront (no 3 a.m. surprise) when an
operator asks for something that fundamentally can't fit, like a true
global shuffle.
"""

# ruff: noqa: E402
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _env

_env.setup(cache_budget="24MiB")

# fsdd's real total (~20.3 MiB, see 02_public_dataset_recipe.py) turned out
# close to that 24 MiB default, which would make even a full materialization
# fit — not the point of this example. Tighten the *effective* budget below
# fsdd's total (but well above what a 2-shard prefetch window needs, ~8 MiB)
# so the sliding window is doing real work and the full-shuffle guardrail
# below actually triggers. Both overrides happen after _env.setup(), same as
# the cache dir swap.
os.environ["DLOAD_CACHE_DIR"] = str(_env.REPO_ROOT / ".dload-cache-small")
os.environ["DLOAD_CACHE_BUDGET"] = "16MiB"

import dload
from dload.config import format_size

PREFETCH = 2
PRINT_EVERY = 500
SHUFFLE_SEED = 123

repo = dload.Repository.open()
ds = repo.dataset("fsdd")
budget = repo.cache.budget
assert budget is not None
print(f"dataset: {ds!r}")
print(f"cache budget: {format_size(budget)} at {repo.cache.root}")
print(f"streaming a full shuffled epoch with prefetch={PREFETCH} shards...")
print()

pipe = ds.samples(prefetch=PREFETCH).shuffle(seed=SHUFFLE_SEED)
print("feasibility report (pipe.check()):")
print(pipe.check().message)
print()

max_used = 0
count = 0
for count, _sample in enumerate(pipe, 1):
    max_used = max(max_used, repo.cache.used_bytes())
    if count % PRINT_EVERY == 0:
        used = repo.cache.used_bytes()
        print(
            f"  [{count:5d} samples] cache used: {format_size(used)} / {format_size(budget)}"
        )

print()
print(f"epoch done: {count} samples streamed, peak cache usage {format_size(max_used)}")
print(f"budget respected throughout: {max_used <= budget}")

print()
print("guardrail: a true global shuffle needs the whole dataset resident...")
try:
    iter(ds.samples().shuffle(full=True))
except dload.InfeasiblePipelineError as exc:
    print(f"InfeasiblePipelineError (as expected):\n{exc}")
