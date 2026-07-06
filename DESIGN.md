# dload — design

Dataset management, storage, caching and streaming for ML training. Single
source of truth is an S3-compatible bucket (Cloudflare R2). Local machines
keep a content-addressed shard cache whose size budget adapts to the
environment (multi-TB scratch → keep everything; 100 GB ephemeral disk →
sliding-window LRU).

## Data model

- **Sample** = string key + named byte fields: `(key, {"audio": b"...", "meta": b"..."})`.
  Fields are opaque bytes; codecs (json/npy/audio) live at the edges.
- **Shard (pack file)** = many samples concatenated into one large object
  (default target 128 MiB) with a msgpack index in the footer. One shard =
  one R2 GET → read-efficient (R2 bills per operation, not egress).
  Shards are content-addressed by sha256 → automatic dedup across versions.
- **Manifest** = JSON document listing shard digests/sizes/sample-counts plus
  user metadata and an optional *recipe* (the original download script,
  preserved verbatim). Manifest digest = sha256 of canonical JSON = the
  dataset **version id**. DVC-style pinning falls out for free.

## Bucket layout

```
datasets/<name>/manifests/<version>.json    # immutable
datasets/<name>/refs/latest                 # text file: current version id
shards/<aa>/<digest>                        # content-addressed, shared
```

## Local layout

```
<cache_dir>/shards/<aa>/<digest>            # mirrors bucket, LRU-evicted
```

Config resolution (first hit wins): env vars (`DLOAD_*`) → `dload.toml` in
cwd or ancestors → `~/.config/dload/config.toml` → defaults. Cache budget:
`"unlimited"`, `"auto"` (half of free disk), or `"50GB"`.

`dload.lock` (TOML, committed to the experiment repo) pins dataset name →
version id; `Repository.dataset(name)` resolves through it when present.

## Streaming pipeline

`Dataset.samples()` returns a `Pipeline`; operators (`map`, `filter`,
`shuffle`, `batch`, `take`, `repeat`, and top-level `mix`/`concat`) build a
lazy DAG. `iter(pipeline)`:

1. **Plan** — walk the DAG, find dataset sources, fix a shard visit order
   (shuffled per-epoch when a `shuffle` op sits above the source).
2. **Feasibility check** — required resident set = Σ over sources of
   `prefetch_window × max_shard_size` (or the whole dataset for
   `shuffle(full=True)` global random access). If it exceeds the cache
   budget → `InfeasiblePipelineError` at iteration start, not 3 hours in.
3. **Execute** — a background prefetcher downloads shards ahead of the
   consumer (window per source), pinning them in the cache; consumed shards
   are unpinned and become LRU-evictable. Cold start streams immediately;
   warm cache is pure local reads.

Randomness = shard-order shuffle (per epoch, seeded) + a sample-level
shuffle buffer — the standard bounded-memory approach. `mix` draws from
child pipelines with given weights.

Operators are built in three layers: planner-aware nodes that cannot be
stream combinators (`SourceNode`, `ShuffleNode`, plus `RepeatNode`/`MixNode`
for structural reasons); a fundamental core — `TransformNode` (the
universal n-ary `fn(*iters) → iterator` substrate, where `.close()`
propagation is centralized), `from_iterable` (non-shard leaf, invisible to
feasibility, replicated per worker), `random_stream` (randomness as a
stream), and `scan`/`flat_map`/`zip_with`/`select`/`through`; and a derived
layer (`map`, `filter`, `batch`, `window`, `choice`, `maybe`, ...) that is
thin wrappers or pure composition — e.g. `choice` = `select` over an index
stream computed by `zip_with` from `random_stream` and a (possibly
stream-valued) probability.

## Modules

| module        | role |
|---------------|------|
| `errors.py`   | exception hierarchy |
| `codecs.py`   | json/npy/text ↔ bytes helpers |
| `pack.py`     | shard pack format: `PackWriter`, `PackReader` |
| `manifest.py` | `Manifest`/`ShardInfo` dataclasses, canonical JSON, digests |
| `config.py`   | layered config resolution, size parsing |
| `remote.py`   | `Remote` protocol; `S3Remote` (boto3), `LocalRemote` (tests) |
| `cache.py`    | `ShardCache`: content-addressed store, pins, LRU eviction |
| `repo.py`     | `Repository`: commit/list/resolve datasets, lock file; `Dataset` handle |
| `pipeline.py` | pipeline DAG, planner, feasibility check, prefetching executor |
| `torch.py`    | optional `IterableDataset` adapter with worker sharding |
| `cli.py`      | `dload` CLI (click) |

Concurrency note: pins are per-process; two processes may download the same
shard concurrently (atomic rename makes that safe), but eviction is only
aware of pins in its own process. One training job per cache dir, or a
budget with headroom, is the supported regime.
