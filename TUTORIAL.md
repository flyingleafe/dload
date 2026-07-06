# dload tutorial — from raw data to training batches

This is the narrative walkthrough. Every code fragment here is the same
pattern used by a runnable, actually-tested script in [`examples/`](examples/).

## 1. The mental model

Four ideas carry the whole library:

1. **A sample is `(key, {field_name: bytes})`.** A WAV plus its JSON
   annotation, a JPEG plus a label byte, a numpy array via `dload.codecs` —
   fields are opaque bytes, codecs live at the edges. Generic by
   construction.
2. **Samples live in shards** — ~128 MiB pack files, content-addressed by
   sha256. Reading a shard costs one GET (R2 bills reads, not egress).
   Identical bytes are stored once, ever: re-commits, shared shards between
   dataset versions, re-uploads — all dedup automatically.
3. **A manifest is a dataset version.** A small JSON file listing shard
   digests plus your metadata and (optionally) the verbatim download script
   that produced the data. The manifest's own sha256 *is* the version id.
   Pin it in `dload.lock` and your experiment repo is reproducible.
4. **The local cache is a sliding window.** Every machine has a cache dir
   and a byte budget (`unlimited` on a 3 TB scratch node, `60GB` on a rented
   box, `auto` = half the free disk). Streaming pins the shards it is about
   to read, releases them behind, and LRU-evicts when over budget. If
   everything fits, everything stays — warm restarts are pure local reads.
   You never manage this; you only pick the budget.

```
bucket (source of truth)                 your machine
  datasets/<name>/manifests/<sha>.json     <cache>/shards/<aa>/<sha256>
  datasets/<name>/refs/latest              (mtime-LRU, pinned while in use)
  shards/<aa>/<sha256>
```

## 2. Point dload at your bucket

```bash
export DLOAD_BUCKET=my-data
export R2_ACCOUNT_ID=...                        # or DLOAD_ENDPOINT_URL
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...

# or persist it per machine / per project:
dload init --user --bucket my-data --cache-dir /scratch/$USER/dload \
           --cache-budget unlimited
dload init --project --prefix myteam/          # writes ./dload.toml
```

Resolution per setting: env vars → `./dload.toml` → `~/.config/dload/config.toml`
→ defaults. `dload status` shows what won and whether the remote is reachable.

## 3. Get data in

Anything that can yield `(key, fields)` tuples can be committed — the writer
never materializes the dataset:

```python
import dload
repo = dload.Repository.open()

def samples():
    for wav in sorted(RAW_DIR.glob("*.wav")):
        yield wav.stem, {
            "wav": wav.read_bytes(),
            "meta": dload.codecs.json_bytes({"speaker": wav.stem.split("_")[1]}),
        }

repo.commit("lab-noise", samples(), meta={"mic": "sm57"})
```

For public datasets, preserve *how you got them* inside the dataset itself:

```python
def download_fsdd(dest):  # the actual download logic
    ...

repo.commit("fsdd", build_samples(), recipe=inspect.getsource(download_fsdd))
# later, anywhere:  print(repo.manifest("fsdd").recipe)   # or: dload recipe fsdd
```

From the shell, `dload commit birds --from ./birds/` groups files by
path-minus-extension: `x.wav` + `x.json` → one sample `x` with fields
`wav`/`json`. Committing is idempotent — unchanged data uploads nothing and
produces the same version id.

Two ingestion shapes worth knowing (both from real examples):

- **Continuous corpora** (text, long time series): chop into chunk samples
  (`"chunk-0042"` → 4 MB of text), reassemble windows in the pipeline
  ([t03](examples/training/t03_char_language_model.py),
  [t06](examples/training/t06_timeseries_forecast.py)).
- **Millions of tiny records** (ratings, dictionary entries): batch them into
  columnar samples (2000 ratings per sample as `.npy` fields) —
  ([t10](examples/training/t10_recsys_matrix_factorization.py),
  [t12](examples/training/t12_seq2seq_g2p.py)).

## 4. Stream it

```python
ds = repo.dataset("lab-noise")          # resolves dload.lock → refs/latest
pipe = (
    ds.samples()                        # Pipeline of (key, {field: bytes})
      .shuffle(4096, seed=0)            # see below for what this really does
      .map(decode)                      # your function, any return type
      .filter(lambda x: x.duration > 1.5)
      .batch(32, collate=my_collate)
)
for batch in pipe:                      # cold: starts streaming immediately
    train_step(batch)                   # warm: pure local reads
for batch in pipe:                      # iterating again = next epoch,
    ...                                 # freshly reshuffled
```

What happens at `iter(pipe)`, in order:

1. **Plan** — fix the shard visit order for this epoch (seeded shuffle if a
   `shuffle` sits above the source).
2. **Check** — compute the required resident set (prefetch windows, or the
   whole dataset for `shuffle(full=True)`) against your cache budget. Doesn't
   fit → `InfeasiblePipelineError` *now*, with numbers and suggestions — not
   at 3 a.m. mid-epoch. Preview anytime with `pipe.check().message`.
3. **Execute** — background threads download ahead of the consumer
   (`ds.samples(prefetch=3)` controls the window), pinning shards in the
   cache; consumed shards unpin and become evictable. That's the whole
   sliding-window machinery — invisible from your code.

**Shuffle semantics** (the one operator worth understanding): 
`.shuffle(buffer_size, seed=…)` does three things — shuffles the shard visit
order beneath it, shuffles sample order inside each shard, and runs a
`buffer_size` streaming shuffle at its own position. Bounded memory, near-
uniform mixing (measured: [BENCHMARKS.md](BENCHMARKS.md) §b03) — this is what
you want for big datasets. `.shuffle(full=True)` is a *true* uniform
permutation; it requires the dataset to fit your cache budget and must sit
directly on `ds.samples()`. Reproducibility: with a seed, order is a pure
function of (seed, epoch); every re-iteration advances the epoch.

**Combining datasets:**

```python
pipe = dload.mix(                        # stochastic weighted interleave
    [speech.samples().shuffle(seed=0).repeat(),
     noise.samples().shuffle(seed=1).repeat()],
    weights=[0.8, 0.2], seed=2,
)
dload.concat([a, b])                     # sequential chaining
```

`mix` draws sample-by-sample (measured ratio error ±0.002). For *paired*
streams — e.g. add a noise clip to every speech clip at a random SNR — use
`dload.zip_with` instead
([t09](examples/training/t09_noise_robust_audio.py)).

**Combinators.** Streams compose; you should rarely need a bespoke
generator. A minimal fundamental core —

```python
dload.from_iterable(factory)      # lift ANY iterable into the DAG (a GPU
                                  # generator, a schedule, a list); no
                                  # feasibility cost, replicated per worker
dload.random_stream(seed)         # randomness itself as a stream
pipe.scan(fn, init)               # stateful map: state = fn(state, item)
pipe.flat_map(fn)                 # one item -> many
dload.zip_with(fn, a, b, ...)     # positional pairing, stops at shortest
dload.select(idx, s0, s1, ...)    # control stream picks who yields next
pipe.through(fn)                  # escape hatch: fn(iterator) -> iterator
```

— and a convenience layer built on it: `dload.zip` (tuples),
`.star_map(fn)`, `.window(size, stride)`, `.maybe(fn, p)` (apply `fn` to a
`p`-fraction), and `dload.choice(pipes, p, seed=...)` — pick one stream per
item, where `p` may be a float, per-stream weights, **or a Pipeline**: the
selection probability can vary over the course of iteration and come from a
stream. `dload.seeded(key, "salt")` / `dload.seeded_rng(...)` give per-key
reproducible randomness for use inside your own functions.

The recipe that used to take ~40 lines of glue
([09_combinators.py](examples/09_combinators.py), asserted for real):

```python
speech = repo.dataset("speech").samples().shuffle(seed=0).map(decode_speech)
real   = repo.dataset("noise").samples().shuffle(seed=1).repeat().map(decode_noise)
gen    = dload.from_iterable(generative_model_stream)   # GPU model, in the DAG

noise = dload.choice([real, gen], p=0.5, seed=2)        # or p=schedule_pipe
pipe  = (dload.zip_with(mix_at_snr, speech, noise)
              .maybe(augment, p=0.2, seed=3)
              .batch(64, collate=collate))
```

One caveat worth knowing: pairing (`zip*`, `select`, `choice`) happens
*within* each DataLoader worker, and a `from_iterable` stream restarts in
every worker. Fine for generative/random side streams; strictly-aligned
1:1 data belongs in one sample with two fields, not two zipped datasets.

## 5. Patterns for real pipelines

These come straight from the twelve tested training examples:

- **Splits.** Deterministic hash of the key:
  `.filter(lambda s: sha1(s[0]) % 10 < 8)` — stable across machines, no split
  files ([t01](examples/training/t01_audio_classification.py)). Or store the
  split as a field at commit time ([t02](examples/training/t02_image_classification.py)).
  Time series and recsys: split by *time*, never randomly
  ([t06](examples/training/t06_timeseries_forecast.py), [t10](examples/training/t10_recsys_matrix_factorization.py)).
- **Two-pass statistics.** Stream once to collect normalization stats / vocab
  / a timestamp quantile, then build the training pipeline with them frozen.
  The second pass hits a warm cache, so pass 1 is nearly free
  ([t04](examples/training/t04_text_classification.py), [t05](examples/training/t05_tabular_regression.py)).
- **Cross-sample state stays in the DAG** — window generation over chunked
  text is `.flat_map`, sequence packing is `.scan(step, init).flat_map(...)`,
  rolling buffers are `.window(size, stride)`; `.through(fn)` covers the
  genuinely custom rest. Staying inside the DAG keeps
  `.shuffle()/.batch()/.repeat()` available downstream
  ([t03](examples/training/t03_char_language_model.py),
  [t06](examples/training/t06_timeseries_forecast.py)).
- **Models inside pipelines.** `.batch(256, collate=…).map(frozen_encoder)` —
  batch *before* the model-map to amortize inference; downstream sees only
  embeddings ([t11](examples/training/t11_model_in_pipeline.py)).
- **Multi-crop.** Decode once, cut several augmented views — measured 1.9×
  throughput for free ([bench/b03](examples/bench/b03_sampling_strategies_bench.py)).
- One memory trap: when a generator cuts small numpy slices from big arrays
  and something downstream buffers them, `.copy()` the slice — a view pins
  its whole base array.

## 6. PyTorch

```python
from dload.torch import as_iterable_dataset

dataset = as_iterable_dataset(pipe)     # keep the reference
loader = torch.utils.data.DataLoader(
    dataset, batch_size=None,           # pipeline already batches
    num_workers=4, persistent_workers=True,
)
for epoch in range(100):
    dataset.set_epoch(epoch)            # deterministic per-epoch reshuffle
    for batch in loader: ...
```

Workers split the epoch's shard order between themselves — exact coverage,
zero duplicate downloads (`set_epoch` reaches even persistent workers via
shared memory). Everything crossing into workers (map/collate functions) must
be module-level, not lambdas. Rule of thumb from the benchmarks: extra workers
pay off when per-sample CPU work is heavy (JPEG decode: +62% even on 2 cores)
and cost when it's light — measure, don't assume.

## 7. Versioning, reproducibility, hygiene

```bash
dload ls                       # what's in the bucket
dload info fsdd                # versions, sizes, meta, recipe presence
dload pin fsdd                 # freeze current version into dload.lock  → commit it
dload pin fsdd 3f2ab1          # or pin an older version by prefix
dload pull fsdd                # fully materialize on big-scratch machines
dload rm old-experiment --yes && dload gc --yes    # delete + reclaim shards
dload cache status | dload cache clear
```

Committing new data under an existing name creates a new version; unchanged
shards are shared, `refs/latest` moves, pinned projects are unaffected until
you re-pin. (`repo.pin`/`unpin`/`versions` do the same from Python.)

## 8. Will it feed my GPUs?

[BENCHMARKS.md](BENCHMARKS.md) measures full pipelines on a deliberately weak
2-core box against FLOPs-derived feed rates for 0.1–0.5B models on 1–4 H100s.
Short version: text token-packing over-delivers by ~20×; a 0.5B ViT on one
H100 is fed with headroom by two cores; audio feature pipelines need one
normal 16–32-core loader host for the hardest configs — and loader throughput
scales horizontally because worker sharding is exact. When in doubt, copy a
bench script and point it at your own dataset.
