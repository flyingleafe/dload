# dload ingestion & streaming benchmarks

Measured with the scripts in [`examples/bench/`](examples/bench/) against a
real Cloudflare R2 bucket. Reference machine — deliberately the *weakest*
plausible link in a training setup:

- 2 CPU cores, 3.8 GB RAM (shared with other services), Hetzner cloud
- ~80 MB/s observed R2 download bandwidth
- benchmarks run serially on a quiet machine, one at a time, inside
  `systemd-run --scope -p MemoryMax=1400M` sandboxes

Every number below is from a full pass over the dataset (no cherry-picked
warmup windows). Rerun any of them: `uv run python examples/bench/b0X_....py`
(`--quick` for a smoke pass).

## b01 — audio streaming (LibriSpeech dev-clean, 343 MB FLAC, 2703 utterances, 5.4 h)

| stage | clips/s | throughput |
|---|---|---|
| ingest (pack + upload to R2) | — | 29.9 MB/s |
| cold sequential (R2 → cache) | 619 | 78.4 MB/s |
| warm sequential (local cache) | 4586 | 580.9 MB/s |
| + FLAC decode | 296 | 2126× realtime |
| + random 5 s crop + 80-bin log-mel | 82 | 409 audio-s/s |
| + SpecAugment + random gain | 92 | 459 audio-s/s |
| prefetch 2 / 4 / 8 (warm) | 93 / 92 / 93 | flat when warm — prefetch matters cold |
| DataLoader workers 0 → 2 | 7.2 → 5.8 batches/s | **negative** on 2 cores: IPC overhead with no spare cores |

**H100 feasibility** (Conformer/Whisper-class encoder, 5 s clips, batch 64/GPU,
990 TFLOPs bf16 @ 40% MFU, step ≈ 6·N·500 frames):

| model | 1× H100 | 4× H100 |
|---|---|---|
| 0.1B | 1320 clips/s (14× this box) | 5280 clips/s (58×) |
| 0.3B | 440 clips/s (4.8×) | 1760 clips/s (19×) |
| 0.5B | 264 clips/s (**2.9×**) | 1056 clips/s (11.5×) |

Verdict: one 16–32-core loader host feeds a 0.5B model on 1–2 H100s; bigger
GPU counts want several loader processes — dload's worker sharding
(`shards[worker::num_workers]`, exact coverage, no duplicate downloads)
scales this horizontally.

## b02 — LLM token packing (enwik9, 954 MB Wikipedia text, 239 × 4 MiB chunks)

| stage | throughput |
|---|---|
| cold sequential (R2) | 78.8 MB/s |
| warm sequential | 1228 MB/s |
| + byte tokenize (uint8 → int64) | 109.3 Mtokens/s |
| + pack into 2048-token sequences | 92.3 Mtokens/s |
| + document shuffle + 64-seq shuffle buffer | **41.3 Mtokens/s** |
| DataLoader (workers=0, [8×2048] batches) | 2032 batches/s ≈ 33 Mtokens/s |

**H100 feasibility** (decoder-only LM, tokens/s = n_gpu · 396 TFLOPs / 6N):

| model | 1× H100 | 4× H100 |
|---|---|---|
| 0.125B | 0.53 Mtok/s | 2.11 Mtok/s |
| 0.35B | 0.19 Mtok/s | 0.75 Mtok/s |
| 0.5B | 0.13 Mtok/s | 0.53 Mtok/s |

Verdict: the worst case (0.125B × 4 H100) needs **~5% of what this 2-core
box already delivers**. Byte tokenization is compute-trivial — budget
5-10× more for a real BPE tokenizer — and text loading still isn't remotely
the bottleneck. Compute is.

## b03 — sampling & augmentation strategies

Shuffle strategy on fashion-mnist (70k samples, 14 shards; uniform
references: displacement 0.333, same-shard adjacency 0.071, rank-corr 0):

| strategy | samples/s | displacement | same-shard adj. | rank corr | resident set |
|---|---|---|---|---|---|
| sequential | 149,727 | 0.000 | 1.000 | +1.000 | 3 shards (12.8 MiB) |
| shuffle(buf=256) | 65,162 | 0.432 | 0.951 | −0.566 | 3 shards |
| shuffle(buf=4096) | 62,705 | 0.430 | 0.466 | −0.553 | 3 shards |
| shuffle(full=True) | 44,967 | 0.334 | 0.076 | −0.003 | whole dataset (56.2 MiB) |

A windowed shuffle whose buffer spans ≳1 shard of samples approaches uniform
mixing at ~1.4× the throughput of a true permutation and a fraction of its
residency; `full=True` is the only option that hits the uniform references
exactly, and the planner enforces its residency requirement up front.
(Rank-corr of windowed shuffles reflects the single shard-order draw — at 14
shards one permutation can correlate; it decorrelates over epochs.)

Mixing (20,000 draws, `.repeat()`ed sources):

- 2-way fsdd 0.7 / fashion-mnist 0.3 → achieved 0.698/0.302 (±0.002), **+2.0% overhead** vs solo streams
- 3-way librispeech 0.5 / fsdd 0.3 / fashion-mnist 0.2 → achieved 0.498/0.299/0.203 (±0.003), **+38.8% overhead** — interleaving a heavyweight decode stream with lightweight ones costs cache/branch locality; weight-heavy streams dominate wall time

Random-window sampling from long audio (2703 LibriSpeech utterances):

| policy | clips/s | audio-s/s |
|---|---|---|
| fixed head crop 5 s | 298 | 1489 |
| random 5 s window (seeded per key) | 281 | 1407 |
| 2 random windows per decode (multi-crop) | 538 | 2689 |

Decode-once-crop-twice is a free 1.91× — amortize expensive decodes across
multiple augmented views.

Per-epoch reshuffling (fashion-mnist, 3 consecutive epochs): 52.8k / 45.5k /
47.5k samples/s — reshuffling costs only O(shards) planning; epochs are
stable warm-cache reads.

## b04 — vision JPEG pipeline (Tiny-ImageNet, 100k JPEGs, 191 MB, 200 classes)

| stage | images/s | note |
|---|---|---|
| ingest (stream from zip → pack → R2) | 3291 | 100k files, 6 shards, 30 s total |
| warm raw bytes | 120,362 | 223 MB/s — per-sample overhead, not bandwidth, limits small records |
| + PIL JPEG decode → RGB | 3653 | 15.0 MPix/s |
| + RandomResizedCrop + flip + normalize | 1169 | the full supervised-vision input pipeline |
| batched [256, 3, 64, 64] | 1156 | |
| DataLoader workers 0 → 2 | 1168 → **1889** | +62%: JPEG decode is GIL/CPU-bound, so extra *processes* pay off even on 2 cores (contrast b01, where they didn't) |

**H100 feasibility** (ViT at 224 px / patch 16 → 197 tokens/image, batch 256/GPU):

| model | 1× H100 | 4× H100 |
|---|---|---|
| ViT-B 0.086B | 3896 img/s (3.3× this box) | 15,583 img/s (13.3×) |
| 0.3B | 1117 img/s (**1.0×**) | 4467 img/s (3.8×) |
| 0.5B | 670 img/s (**0.6×** — fed with headroom) | 2680 img/s (2.3×) |

Verdict: this 2-core box already saturates a 0.3B ViT on one H100 and feeds a
0.5B with 1.7× headroom; small-model/many-GPU configs (ViT-B × 4) need ~13×,
i.e. one ordinary 16-32-core loader host or a few sharded loader processes.

## Reproducing

The datasets these benchmarks ingest stay in the bucket under content-addressed
shards, so re-runs skip ingest ("already ingested") and cold numbers are
produced by clearing the local cache (`repo.cache.clear()`), not re-uploading.
On a box with more cores, expect decode/augment stages to scale ~linearly with
cores and the DataLoader-workers rows to turn positive.
