"""Plain streaming + a feature transform over "lab-tones".

samples() -> map(decode wav, compute a log-magnitude STFT-ish feature with
plain numpy) -> filter(duration >= 1.5s) -> batch(16, pad-and-stack).

Run this script twice: the first run streams cold (every shard is a
download), the second run is a pure local warm-cache read of the same
shards — compare the wall-clock printed at the bottom.
"""

# ruff: noqa: E402
from __future__ import annotations

import io
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import _env

_env.setup()

import dload

FRAME_SIZE = 512
HOP = 256
MIN_DURATION = 1.5
BATCH_SIZE = 16


def log_magnitude_stft(wave: np.ndarray) -> np.ndarray:
    """(n_frames, n_freq_bins) log-magnitude spectrum via a plain-numpy
    rfft over overlapping Hann-windowed frames."""
    window = np.hanning(FRAME_SIZE)
    n_frames = max(0, 1 + (len(wave) - FRAME_SIZE) // HOP)
    frames = np.stack(
        [wave[i * HOP : i * HOP + FRAME_SIZE] * window for i in range(n_frames)]
    )
    spectrum = np.fft.rfft(frames, axis=-1)
    return np.log1p(np.abs(spectrum)).astype(np.float32)


def decode(sample: tuple[str, dict[str, bytes]]) -> dict:
    key, fields = sample
    wave, _sr = sf.read(io.BytesIO(fields["wav"]), dtype="float32")
    annotation = dload.codecs.json_from(fields["meta"])
    return {
        "key": key,
        "feature": log_magnitude_stft(wave),
        "label": annotation["label"],
        "duration": annotation["duration_s"],
    }


def collate(items: list[dict]) -> dict:
    max_t = max(item["feature"].shape[0] for item in items)
    n_freq = items[0]["feature"].shape[1]
    padded = np.zeros((len(items), max_t, n_freq), dtype=np.float32)
    for i, item in enumerate(items):
        t = item["feature"].shape[0]
        padded[i, :t] = item["feature"]
    return {
        "features": padded,
        "labels": [item["label"] for item in items],
        "durations": np.array([item["duration"] for item in items], dtype=np.float32),
    }


repo = dload.Repository.open()
ds = repo.dataset("lab-tones")
cache_before = repo.cache.used_bytes()
print(f"dataset: {ds!r}")
print(f"local cache before streaming: {cache_before / 2**20:.1f} MiB used")

pipe = (
    ds.samples()
    .map(decode)
    .filter(lambda item: item["duration"] >= MIN_DURATION)
    .batch(BATCH_SIZE, collate=collate)
)

label_histogram: Counter[str] = Counter()
batch_times: list[float] = []
n_samples = 0
t_prev = time.monotonic()
for i, batch in enumerate(pipe):
    now = time.monotonic()
    batch_times.append(now - t_prev)
    t_prev = now
    n_samples += len(batch["labels"])
    label_histogram.update(batch["labels"])
    if i == 0:
        print(
            f"first batch: features {batch['features'].shape}, dtype {batch['features'].dtype}"
        )

print()
print(
    f"epoch done: {len(batch_times)} batches, {n_samples} samples (>= {MIN_DURATION}s)"
)
print(f"label histogram: {dict(label_histogram)}")
print(f"first batch:    {batch_times[0] * 1000:.1f} ms")
if len(batch_times) > 1:
    steady = batch_times[1:]
    print(
        f"steady state:   {sum(steady) / len(steady) * 1000:.1f} ms/batch avg "
        f"over {len(steady)} batches"
    )
print(
    f"local cache after streaming: {repo.cache.used_bytes() / 2**20:.1f} MiB used "
    "(warm now — re-run this script to see steady-state-only timings from the start)"
)
