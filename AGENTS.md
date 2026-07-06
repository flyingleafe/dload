# dload — agent notes

Dataset management/storage/streaming library for ML training. One
S3-compatible bucket is the source of truth; machines stream through a
budgeted local cache. Read `DESIGN.md` for architecture rationale,
`TUTORIAL.md` for the user-level walkthrough. This file is operational
knowledge for working on the codebase.

## Commands

```bash
uv sync                                   # install (nix devshell runs this on entry)
uv run pytest -q -m "not integration"     # fast suite (~200 tests, no network)
uv run pytest -q                          # full suite; integration tests need .env R2 creds
ruff check src tests examples && ruff format src tests examples
basedpyright                              # standard mode; must be 0 errors (pre-commit enforces)
uv run dload --help                       # the CLI
uv run python examples/01_commit_lab_data.py   # examples run from repo root
```

Pre-commit (nix git-hooks) runs ruff + ruff-format + basedpyright; a commit
fails if any of them do.

## Layout

| path | contents |
|---|---|
| `src/dload/pack.py` | shard file format (payload + msgpack footer index) |
| `src/dload/manifest.py` | `Manifest`/`ShardInfo`; canonical-JSON sha256 = version id |
| `src/dload/remote.py` | `Remote` protocol; `S3Remote` (boto3), `LocalRemote` (tests, has `op_counts`) |
| `src/dload/cache.py` | content-addressed cache: mtime-LRU, in-process pins, budget |
| `src/dload/repo.py` | `Repository` (commit/resolve/pin/gc), `Dataset` handle |
| `src/dload/pipeline.py` | node DAG, planner + feasibility check, prefetching executor |
| `src/dload/torch.py` | `PipelineDataset` adapter (imports torch; rest of lib is torch-free) |
| `src/dload/cli.py` | click CLI |
| `tests/helpers.py` | `build_repo`, seeded sample generators, `SlowLocalRemote` |
| `examples/` | 9 library-usage examples; `training/` 12 real training runs; `bench/` 4 benchmarks |

## Invariants — do not break

- **Content addressing**: shards are named by sha256 of their bytes; manifests
  by sha256 of canonical JSON (sorted keys, compact separators, `version` key
  excluded). Identical data must always dedup; any format change needs a new
  `FORMAT_VERSION` and a reader that rejects unknown versions.
- **Read efficiency**: streaming N shards must cost N object GETs (plus
  ref + manifest). Tests assert this via `LocalRemote.op_counts`. Never add
  per-sample remote reads to a hot path.
- **Pack format**: payload region, msgpack index, u64 index length, 8-byte
  magic `DLOADPK1` — footer-based so writers stream and readers grab the tail.
- **Cache concurrency**: install + pin must happen under one lock
  (`_fetch_and_install`) — a shard visible on disk but unpinned is fair game
  for concurrent eviction (this was a real race, fixed; test
  `test_sliding_window_streams_fully_within_budget` guards it). Budget
  accounting reserves in-flight sizes. Pins are per-process only; the
  supported regime is one training job per cache dir.
- **Worker sharding contract**: `Pipeline._iterate(worker, num_workers, epoch)`
  — all workers must compute the *same* shard order for an epoch, then take
  `shards[worker::num_workers]` (windowed) or stripes of one global sample
  permutation (`full=True`). This is why unseeded `ShuffleNode` draws
  `auto_seed` at construction (pickled to workers) instead of fresh entropy
  per process — fresh entropy desyncs workers and silently duplicates/drops
  shards. Seeds derive via `_derive_rng(seed, epoch, node_id)`; the worker
  index must NOT enter shard-order derivation.
- **Torch adapter epoch**: `PipelineDataset._epoch` is a `multiprocessing.Value`
  because persistent DataLoader workers hold a long-lived copy of the dataset
  object; a plain attribute never reaches them and reshuffling silently stops.
- **Feasibility before I/O**: `iter(pipeline)` plans and raises
  `InfeasiblePipelineError` before any download. `shuffle(full=True)` requires
  its child to be a `SourceNode` and the whole dataset to fit the budget.
- **Errors**: operational failures raise `DloadError` subclasses
  (`errors.py`); programmer errors (bad arguments) raise `ValueError` —
  deliberate split, don't "fix" it.
- **Combinator substrate**: `map/filter/batch/take/concat/scan/flat_map/
  window/zip_with/select/star_map/through` are all `TransformNode(children, fn)` —
  one place opens/closes child streams (`.close()` propagation lives there,
  nowhere else). `fn` must be a module-level function or `functools.partial`
  over one (DataLoader spawn pickling). Only `SourceNode`/`ShuffleNode`
  (planner-aware), `RepeatNode` (re-opens its child), and `MixNode`
  (reactive re-weighting on exhaustion) are standalone node classes — don't
  add new ones without the same justification.
- **`from_iterable` is replicated per worker** (default) and invisible to
  the feasibility check — right for generative/control/random side streams,
  wrong as a primary multi-worker source (`shard=True` opts into slicing).
  Corollary: `zip`/`zip_with`/`select` pair *within* a worker; a replicated
  branch restarts at 0 in every worker, so zip a sharded stream only against
  stateless/generative side streams — strictly-aligned 1:1 data belongs in
  one sample with two fields (test
  `test_zip_replicated_branch_restarts_per_worker` guards this contract).
- **`.maybe` must stay zip-based**: implementing it as
  `choice([pipe.map(fn), pipe])` opens the same upstream twice — desyncs
  order and double-reads shards, breaking the N-GETs invariant.

## Testing conventions

- Unit tests use `LocalRemote(tmp_path)` + `ShardCache(tmp_path, budget)`;
  no network, no mocks of dload's own classes.
- `@pytest.mark.integration` = real R2. Credentials come from `.env`
  (`R2_ACCOUNT_ID`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
  **Only ever touch the `dload-test` bucket.** Prefixes: `citest/`
  (integration tests, cleaned up), `examples/` (example datasets, persistent),
  `examples-cli/` (CLI walkthrough, self-cleaning).
- Committing identical data twice in a test needs a frozen clock
  (`tests/helpers.py:freeze_repo_clock`) because `created` is in the manifest
  digest.
- DataLoader tests: decode/collate functions must be module-level (spawn
  pickling); `ShardCache` has `__getstate__` dropping locks.

## Gotchas that have actually bitten

- **numpy view pinning**: yielding `arr[:n]` from a generator into any
  downstream buffer (shuffle buffer, batch list) pins the whole base array.
  `.copy()` small slices cut from big arrays (see `examples/bench/b02`, this
  OOM'd a 3.8 GB machine).
- **This dev box** (if you are on "hetzner"): 2 cores, 3.8 GB RAM shared with
  other services, `/tmp` nearly full. Run at most ONE heavy process (training,
  ingest, benchmark) at a time; put big files under the repo (`.downloads/`,
  `.dload-cache*`); wrap risky runs in
  `systemd-run --user --scope -p MemoryMax=1400M` so the OOM killer can't take
  the whole tmux scope down; use `PYTHONUNBUFFERED=1` for detached logs; never
  `pkill -f <pattern>` where the pattern appears in your own shell's cmdline.
- **soundfile WAV FLOAT subtype** embeds a wall-clock PEAK chunk — breaks
  content-addressed dedup of "identical" audio; use `PCM_16`/`PCM_32` for
  deterministic bytes (see `examples/_labtones.py`).
- `dload.mix()` is stochastic interleaving, not zip; for paired streams
  (signal + noise) use `dload.zip_with(fn, a, b)` (`examples/training/t09`).
- Don't run `Repository.gc()` concurrently with a `commit` — shards uploaded
  before their manifest look orphaned.

## Extending

New pipeline operators almost never need a `Node` subclass anymore — compose
the existing combinators, or wrap `TransformNode((child,), fn)` with a
module-level `fn` (see how `window`/`batch` are built in `pipeline.py`), or
use `.through(fn)` for a one-off stage. Reach for a `Node` subclass only
when the operator must be planner-aware (shard order/feasibility) or must
re-open child streams (`RepeatNode`-like); then implement `iterate(ctx)` as
a generator (must be a generator, not a raw iterator — `.close()`
propagation depends on it) and teach `_walk`/`_plan` about it only if it
affects shard order or feasibility.
