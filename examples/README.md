# dload examples

All of these run for real against the `dload-test` R2 bucket. To run them:

1. Put R2 credentials in `.env` at the repo root (`R2_ACCOUNT_ID`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
2. `uv sync` (installs numpy/soundfile/torch/pillow dev dependencies the examples use).
3. Run each script from the repo root, e.g. `uv run python examples/01_commit_lab_data.py`.

Everything here is content-addressed, so scripts are safe to re-run: identical
data re-commits without uploading anything new (`01` and `07` demonstrate this
explicitly). `02` downloads the ~16 MB FSDD zip once and caches it under
`.fsdd-download/`; later runs skip straight to extraction. Run `01` and `02`
first — `03`-`07` stream the datasets they commit.

## Library basics

| script | what it demonstrates |
|---|---|
| [`01_commit_lab_data.py`](01_commit_lab_data.py) | Committing synthetic "lab data" that only exists on this machine; content-addressed dedup on re-commit. |
| [`02_public_dataset_recipe.py`](02_public_dataset_recipe.py) | Ingesting a public dataset (FSDD) with the download script preserved as the manifest's `recipe`, round-tripped from R2. |
| [`03_stream_features.py`](03_stream_features.py) | A plain streaming pipeline: decode → feature transform → filter → batch, with cold-vs-warm-cache timing. |
| [`04_mix_augment.py`](04_mix_augment.py) | Weighted mixing of two datasets with seeded, per-sample-reproducible augmentation; proves determinism and reports the achieved mix ratio. |
| [`05_sliding_window_small_cache.py`](05_sliding_window_small_cache.py) | Streaming a dataset bigger than the cache budget on a tiny (16 MiB) cache; the feasibility guardrail rejecting `shuffle(full=True)`. |
| [`06_torch_dataloader.py`](06_torch_dataloader.py) | `dload.torch` with a multi-worker `DataLoader`: exact per-epoch worker sharding and per-epoch reshuffling that reaches persistent workers via `set_epoch`. |
| [`07_versioning_pinning.py`](07_versioning_pinning.py) | A second version of a dataset under the same name, shard-level dedup across versions, and pinning/unpinning a specific version in `dload.lock`. |
| [`08_cli_walkthrough.sh`](08_cli_walkthrough.sh) | The full `dload` CLI lifecycle — commit, ls, info, recipe, pull, pin/unpin, rm, gc, cache status/clear. |
| [`09_combinators.py`](09_combinators.py) | Combinators over streams: `choice` between a real noise corpus and a generative model lifted via `from_iterable`, selection probability from a schedule *stream*, `zip_with` pairing, `maybe` augmentation — ratio, schedule-tracking, and bit-exact determinism all asserted. |
| [`10_derived_datasets.py`](10_derived_datasets.py) | `repo.derive()`: memoizing a deterministic tokenization pipeline into a shared, content-addressed dataset — first call MISSes and materializes, a second (equivalently-built) call HITs the same snapshot instantly; `fingerprint()`/`source_versions()`, the `tag` escape hatch, and the `ValueError` from an unseeded `.shuffle()`. |

## Real training pipelines (`training/`)

Twelve end-to-end training runs on real downloaded data, each committing its
dataset to R2 (with the download recipe preserved), streaming it through a
dload pipeline, training a small torch model on CPU, and **asserting** that it
actually learned (loss drop + task-metric threshold). Every one has been run to
completion on a 2-core machine. Run order: t01 needs `02`'s FSDD; t07/t08/t11
need t02's fashion-mnist; everything else is self-contained.

| script | task / paradigm | pipeline patterns shown |
|---|---|---|
| [`t01_audio_classification.py`](training/t01_audio_classification.py) | Spoken-digit classification (supervised, audio) — 87% acc | Hash-key train/eval split via `.filter`; log-mel features in `.map`; per-epoch reshuffle |
| [`t02_image_classification.py`](training/t02_image_classification.py) | FashionMNIST CNN (supervised, vision) — 79% acc | Raw-array byte fields; split stored as a field; idempotent ingest |
| [`t03_char_language_model.py`](training/t03_char_language_model.py) | Char-level LM on Tiny Shakespeare (self-supervised, text) | Continuous corpus as chunked samples; window generation around the pipeline |
| [`t04_text_classification.py`](training/t04_text_classification.py) | SMS spam detection (supervised, NLP) — 98.9% acc | Two-pass vocab build; class-weighted loss from streamed label counts |
| [`t05_tabular_regression.py`](training/t05_tabular_regression.py) | Wine-quality regression (supervised, tabular) | `.npy` codec fields; two-pass normalization stats; warm-cache second pass |
| [`t06_timeseries_forecast.py`](training/t06_timeseries_forecast.py) | 7-day temperature forecasting (GRU) — beats persistence | Monthly segment samples; temporal (not random!) split; cross-segment window stitching |
| [`t07_image_autoencoder.py`](training/t07_image_autoencoder.py) | Conv autoencoder (unsupervised, vision) | Label-free reuse of another example's dataset; 1-NN embedding probe |
| [`t08_contrastive_ssl.py`](training/t08_contrastive_ssl.py) | SimCLR-style contrastive pretraining + linear probe | Stochastic two-view augmentation in `.map`; honest random-encoder baseline |
| [`t09_noise_robust_audio.py`](training/t09_noise_robust_audio.py) | Noise-robust digit classification — +25 pts at 5 dB SNR | Endless `.repeat()` noise stream zipped with signal stream; SNR-controlled mixing |
| [`t10_recsys_matrix_factorization.py`](training/t10_recsys_matrix_factorization.py) | MovieLens matrix factorization — beats mean baseline | Columnar batching of 100k tiny records; temporal split threshold from pass 1 |
| [`t11_model_in_pipeline.py`](training/t11_model_in_pipeline.py) | Frozen encoder as a pipeline stage | `.batch()` **then** `.map(model)` to amortize inference; embeddings-only downstream |
| [`t12_seq2seq_g2p.py`](training/t12_seq2seq_g2p.py) | Grapheme→phoneme seq2seq on CMUdict — PER 0.30 | Columnar text batching; variable-length padded collate; greedy decode eval |

## Ingestion & throughput benchmarks (`bench/`)

No training — these measure what a dload pipeline can feed, and put the
numbers against what 0.1-0.5B-parameter training on 1-4 H100s would demand
(FLOPs-based estimates, formulas printed by each script). `--quick` runs a
smoke pass; full runs want a quiet machine. Results from this repo's reference
machine live in [`BENCHMARKS.md`](../BENCHMARKS.md).

| script | pipeline benchmarked |
|---|---|
| [`b01_audio_streaming_bench.py`](bench/b01_audio_streaming_bench.py) | LibriSpeech FLAC: cold/warm streaming, decode, log-mel + SpecAugment, prefetch sweep, DataLoader workers; Conformer/Whisper-class feed-rate math |
| [`b02_text_tokenpack_bench.py`](bench/b02_text_tokenpack_bench.py) | enwik9 (1 GB): chunk streaming, byte tokenization, GPT-style 2048-token packing with document+sequence shuffling; 0.125-0.5B LM tokens/s math |
| [`b03_sampling_strategies_bench.py`](bench/b03_sampling_strategies_bench.py) | Shuffle strategies (sequential / buffered / full) with throughput **and** shuffle-quality metrics; mix-ratio fidelity and overhead; multi-crop window sampling |
| [`b04_vision_jpeg_bench.py`](bench/b04_vision_jpeg_bench.py) | Tiny-ImageNet JPEGs: decode, RandomResizedCrop-style augmentation, batching, workers; ViT-class images/s math |
