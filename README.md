# dload

Dataset management, storage, caching and streaming for ML training.

One S3-compatible bucket (Cloudflare R2, MinIO, AWS, ...) is the single
source of truth for all your datasets — public ones you downloaded with a
script, and lab data that exists nowhere else. Any machine — a slurm node
with 3 TB of scratch, a rented H100 box, a Colab notebook with 100 GB of
disk — streams the same data through a local cache that adapts to whatever
space it has: everything fits → it stays warm forever; it doesn't → the
cache slides over the dataset like a window, downloading ahead of the
dataloader and evicting behind it.

```python
import dload

repo = dload.Repository.open()                  # env vars / dload.toml
noise = repo.dataset("lab-noise")
speech = repo.dataset("librispeech")            # multi-TB is fine

pipe = dload.mix(
    [speech.samples().shuffle(seed=0).repeat(),
     noise.samples().shuffle(seed=1).repeat()],
    weights=[0.8, 0.2], seed=2,
).map(decode_and_augment).batch(32)

for batch in pipe:                              # starts streaming instantly,
    train_step(batch)                           # cold or warm
```

No cache management in sight: iteration plans shard access, checks that the
required resident set fits the machine *before* touching storage (a
`shuffle(full=True)` over a 2 TB dataset on a 50 GB disk fails at `iter()`
with an explanation, not at 3 a.m.), then prefetches in the background.

## How data is stored

- Samples — `(key, {field_name: bytes})` — are packed into ~128 MiB
  **shards**, so a training pass over R2 costs one GET per 128 MiB instead
  of one per file (R2 bills reads, not egress).
- Shards are content-addressed (sha256): identical data is stored once,
  re-commits upload nothing new, versions share storage.
- A **manifest** (plain JSON in the bucket) lists the shards of one dataset
  version; its own sha256 is the version id. The original download script
  can be embedded in it (`recipe`), so "where did this come from" ships
  with the data.
- `dload.lock`, committed to your experiment repo, pins names to version
  ids — DVC-style reproducibility without DVC.

```
s3://bucket/datasets/<name>/manifests/<version>.json
s3://bucket/datasets/<name>/refs/latest
s3://bucket/shards/<aa>/<sha256>
```

## Setup

```bash
uv add dload-ml         # or pip install dload-ml; inside this repo: uv sync
                        # (PyPI name is dload-ml; the import is `import dload`)

# where the remote lives — env vars…
export DLOAD_BUCKET=my-data
export R2_ACCOUNT_ID=...             # or DLOAD_ENDPOINT_URL=https://...
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...

# …or a config file (machine-wide or per project)
dload init --user --bucket my-data --cache-dir /scratch/$USER/dload \
           --cache-budget unlimited          # slurm scratch: keep everything
dload init --user --cache-budget 60GB       # ephemeral box: sliding window
```

Config resolution per setting: env → `./dload.toml` (project) →
`~/.config/dload/config.toml` (user) → defaults (`~/.cache/dload`, budget
`auto` = half the free disk). Credentials always come from the standard
AWS chain.

## CLI

```bash
dload status                         # config, cache usage, remote reachability
dload commit birds --from ./birds/ --recipe download_birds.sh
dload ls                             # what's in the bucket
dload info birds                     # manifest, meta, versions
dload recipe birds                   # the download script, verbatim
dload pull birds                     # materialize fully (big-scratch machines)
dload pin birds                      # write version to dload.lock
dload cache status | dload cache clear
dload rm birds --version 3f2a && dload gc    # delete + reclaim shards
```

`dload commit --from DIR` groups files by path-sans-extension: `x.wav` +
`x.json` become one sample `x` with fields `wav` and `json`.

## Python API

Ingest anything — the writer is just an iterator of samples:

```python
def samples():
    for path in wav_files:
        yield path.stem, {
            "wav": path.read_bytes(),
            "meta": dload.codecs.json_bytes({"speaker": ...}),
        }

repo.commit("lab-noise", samples(), meta={"mic": "sm57"},
            recipe=Path("download.sh").read_text())
```

Stream with transformations — pipelines are lazy, re-iterable (each pass is
a new epoch) and deterministic under seeds:

```python
pipe = (ds.samples()                  # (key, {field: bytes})
          .shuffle(4096, seed=0)      # shard order + in-shard + buffer
          .map(decode)                # your function
          .filter(long_enough)
          .batch(32, collate=my_collate))

pipe.check()                          # feasibility report, no I/O
dload.mix([p1, p2], weights=[.7, .3]) # stochastic multiplexing
dload.concat([p1, p2])                # sequential
ds.samples().shuffle(full=True)       # true uniform permutation
                                      # (requires dataset ≤ cache budget)
```

Streams compose through a small combinator core — `scan`, `flat_map`,
`zip_with`, `select`, `through` — plus `dload.from_iterable`
(lift any iterable, e.g. a generative model, into the DAG),
`dload.random_stream` (randomness as a stream), and derived combinators
like `dload.choice(pipes, p)` where `p` can itself be a Pipeline (selection
probability varying over iteration). See the combinators section of
[TUTORIAL.md](TUTORIAL.md) and
[`examples/09_combinators.py`](examples/09_combinators.py).

Expensive, shared preprocessing (tokenization, feature extraction,
resampling) can be memoized instead of repeated: `repo.derive(name, pipe)`
runs a finite, deterministic pipeline once, commits its output as a normal
dataset, and hands that same snapshot to every other caller of the
identical pipeline — see "Derived datasets" (§8) in [TUTORIAL.md](TUTORIAL.md).

PyTorch:

```python
from dload.torch import as_iterable_dataset

loader = torch.utils.data.DataLoader(
    as_iterable_dataset(pipe), batch_size=None, num_workers=4)
for epoch in range(10):
    loader.dataset.set_epoch(epoch)   # deterministic reshuffling
    for batch in loader: ...
```

Workers split the planned shard order between them — no duplicate
downloads, exact coverage per epoch.

## Examples

Every script in [`examples/`](examples/) has been run for real against an
R2 bucket — synthetic lab audio, a public dataset ingested via a preserved
recipe, feature pipelines, mixing + augmentation, a small cache streaming a
much larger dataset, torch DataLoaders, versioning/pinning, the combinator
core end-to-end, and a full CLI walkthrough. Start with
[`examples/README.md`](examples/README.md).

## Docs

- [TUTORIAL.md](TUTORIAL.md) — the narrative walkthrough: mental model,
  ingestion, pipeline patterns, torch, versioning, all backed by tested
  examples.
- [BENCHMARKS.md](BENCHMARKS.md) — measured throughput vs. what 0.1–0.5B
  models on 1–4 H100s actually consume.
- [DESIGN.md](DESIGN.md) — architecture and format internals.
- [AGENTS.md](AGENTS.md) — operational notes for coding agents
  (`CLAUDE.md` symlinks to it).

## Design

Architecture notes live in [DESIGN.md](DESIGN.md). The short version: the
remote is dumb object storage, manifests are immutable JSON, shards are
immutable content-addressed packs with a msgpack footer index, the local
cache is a directory of shards with mtime-LRU eviction and in-process pins,
and the pipeline planner turns "what you declared" into "which shards, in
what order, how far ahead to prefetch, and does it fit this machine".
