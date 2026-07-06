"""The audio-ML case: mix two datasets and augment on the fly.

Mixes "fsdd" (weight 0.7) and "lab-tones" (weight 0.3), each shuffled and
repeated forever, decodes+resamples+augments to a fixed 1.0 s waveform, and
proves the whole thing is deterministic: two independently-built pipelines
with the same seeds produce byte-identical first batches.
"""

# ruff: noqa: E402
from __future__ import annotations

import hashlib
import io
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import _env

_env.setup()

import dload

TARGET_SR = 16_000
TARGET_LEN = TARGET_SR  # exactly 1.0 s
WEIGHTS = [0.7, 0.3]
SEED = 7
BATCH_SIZE = 32
N_BATCHES = 40


def resample_linear(wave: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wave.astype(np.float32)
    duration = len(wave) / sr
    n_target = max(1, round(duration * target_sr))
    x_old = np.linspace(0.0, duration, num=len(wave), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_target, endpoint=False)
    return np.interp(x_new, x_old, wave).astype(np.float32)


def fix_length(wave: np.ndarray, target_len: int) -> np.ndarray:
    if len(wave) >= target_len:
        return wave[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[: len(wave)] = wave
    return out


def augment(wave: np.ndarray, key: str) -> np.ndarray:
    """Gain + additive noise from a rng derived from the sample's key, so
    the same sample always gets the same augmentation regardless of when
    or in what order it is streamed."""
    rng = np.random.default_rng(dload.seeded(key))
    gain = rng.uniform(0.7, 1.3)
    noise = rng.standard_normal(wave.shape[0]).astype(np.float32) * rng.uniform(0, 0.02)
    out = wave * gain + noise
    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak if peak > 1.0 else out).astype(np.float32)


def decode(
    sample: tuple[str, dict[str, bytes]], source_label: str
) -> tuple[np.ndarray, str]:
    key, fields = sample
    wave, sr = sf.read(io.BytesIO(fields["wav"]), dtype="float32")
    wave = fix_length(resample_linear(wave, sr, TARGET_SR), TARGET_LEN)
    return augment(wave, key), source_label


# functools.partial (not a lambda/closure) so these stay module-level and
# spawn-safe, same rule as the torch example.
decode_fsdd = partial(decode, source_label="fsdd")
decode_labtones = partial(decode, source_label="lab-tones")


def collate(items: list[tuple[np.ndarray, str]]) -> tuple[np.ndarray, list[str]]:
    waveforms = np.stack([wave for wave, _label in items])
    labels = [label for _wave, label in items]
    return waveforms, labels


def build_pipeline(repo: dload.Repository, seed: int, n_batches: int) -> dload.Pipeline:
    # Node ids (and therefore derived per-node rng seeds) are assigned by
    # DAG position, so every pipeline that must reproduce the same stream
    # has to be built with the *exact* same chain of operators, .take()
    # included — wrapping an extra node around a pipeline shifts every id
    # below it and changes the random draws even under the same seed.
    fsdd = repo.dataset("fsdd").samples().shuffle(seed=seed).repeat().map(decode_fsdd)
    lab = (
        repo.dataset("lab-tones")
        .samples()
        .shuffle(seed=seed + 1)
        .repeat()
        .map(decode_labtones)
    )
    mixed = dload.mix([fsdd, lab], weights=WEIGHTS, seed=seed + 2)
    return mixed.batch(BATCH_SIZE, collate=collate).take(n_batches)


def batch_checksum(batch: tuple[np.ndarray, list[str]]) -> str:
    waveforms, labels = batch
    material = waveforms.tobytes() + "|".join(labels).encode()
    return hashlib.sha256(material).hexdigest()[:16]


repo = dload.Repository.open()

print("determinism check: two fresh identically-seeded pipelines...")
first_a = next(iter(build_pipeline(repo, SEED, N_BATCHES)))
first_b = next(iter(build_pipeline(repo, SEED, N_BATCHES)))
checksum_a, checksum_b = batch_checksum(first_a), batch_checksum(first_b)
print(f"  pipeline A first batch checksum: {checksum_a}")
print(f"  pipeline B first batch checksum: {checksum_b}")
print(f"  identical: {checksum_a == checksum_b}")
assert checksum_a == checksum_b

full_pipe = build_pipeline(repo, SEED, N_BATCHES)
print()
print("feasibility report (pipe.check()):")
print(full_pipe.check().message)

print()
print(f"streaming {N_BATCHES} batches of {BATCH_SIZE}...")
source_counts: Counter[str] = Counter()
for i, (waveforms, labels) in enumerate(full_pipe):
    source_counts.update(labels)
    if i == 0:
        print(f"  first batch: {waveforms.shape} {waveforms.dtype}")
        print(
            f"  checksum:    {batch_checksum((waveforms, labels))} (matches A/B above)"
        )

total = sum(source_counts.values())
print()
print(f"achieved mix over {total} samples: {dict(source_counts)}")
for name, weight in zip(("fsdd", "lab-tones"), WEIGHTS, strict=True):
    achieved = source_counts[name] / total
    print(f"  {name}: requested {weight:.2f}, achieved {achieved:.3f}")
