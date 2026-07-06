"""Combinators over streams: the speech + real-noise + generated-noise recipe.

The scenario every hand-rolled glue script eventually grows into: mix speech
with noise that comes EITHER from a real noise corpus OR from a generative
model, at a random per-utterance SNR, with occasional augmentation — and keep
it reproducible. Here it is as pure composition:

    speech      dload pipeline      (fsdd, shuffled, decoded)
    real noise  dload pipeline      (lab-tones, shuffled, endless)
    gen noise   dload.from_iterable (a "GPU model" lifted into the DAG)

    noise = dload.choice([real, gen], p=..., seed=...)     # pick per item
    pipe  = (dload.zip_with(mix_at_snr, speech, noise)     # pair + mix
                  .maybe(augment, p=AUG_PROB, seed=...)    # 20% augmentation
                  .batch(BATCH_SIZE, collate=collate))

Three properties are asserted, not just claimed: the achieved real/generated
ratio matches `p`; `p` can itself be a STREAM (a schedule ramping 0 -> 1 over
the epoch, tracked in quarters); and rebuilding the identical pipeline
reproduces the identical batches, bit for bit.

Run 01 and 02 first (they commit lab-tones and fsdd).
"""

# ruff: noqa: E402
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import _env

_env.setup()

import dload

CROP_N = 8000  # fixed-length windows; this demo mixes arrays, not sample rates
SNR_RANGE = (0.0, 20.0)
AUG_PROB = 0.2
BATCH_SIZE = 32
N_TAKE = 1200  # speech utterances consumed per experiment below
GEN_CHUNK = 64  # the "GPU model" generates this many clips per forward pass


def fit(wave: np.ndarray) -> np.ndarray:
    """Crop or tile to exactly CROP_N samples."""
    if len(wave) < CROP_N:
        wave = np.tile(wave, CROP_N // len(wave) + 1)
    return wave[:CROP_N].astype(np.float32)


def decode_speech(sample: tuple[str, dict[str, bytes]]) -> dict:
    key, fields = sample
    wave, _sr = sf.read(io.BytesIO(fields["wav"]), dtype="float32")
    return {
        "key": key,
        "wave": fit(wave),
        "ann": dload.codecs.json_from(fields["meta"]),
    }


def decode_noise(sample: tuple[str, dict[str, bytes]]) -> dict:
    key, fields = sample
    wave, _sr = sf.read(io.BytesIO(fields["wav"]), dtype="float32")
    return {"wave": fit(wave), "source": "real"}


def generated_noise(seed: int = 7):
    """Stand-in for a generative model: emits noise in GEN_CHUNK batches —
    one 'forward pass' per chunk, not per sample (the batch-then-emit
    pattern you'd use for a real GPU model). Seeded, so each epoch of the
    lifted stream is reproducible."""
    rng = np.random.default_rng(seed)
    while True:
        chunk = rng.standard_normal((GEN_CHUNK, CROP_N)).astype(np.float32) * 0.1
        for wave in chunk:
            yield {"wave": wave, "source": "gen"}


def mix_at_snr(speech: dict, noise: dict) -> dict:
    """Pair one speech draw with one noise draw at a per-key-seeded SNR."""
    snr_db = np.random.default_rng(dload.seeded(speech["key"], "snr")).uniform(
        *SNR_RANGE
    )
    sig, noi = speech["wave"], noise["wave"]
    sig_p = float(np.mean(sig.astype(np.float64) ** 2)) + 1e-10
    noi_p = float(np.mean(noi.astype(np.float64) ** 2)) + 1e-10
    scale = np.sqrt((sig_p / 10 ** (snr_db / 10)) / noi_p)
    return {
        "key": speech["key"],
        "audio": (sig + noi * scale).astype(np.float32),
        "ann": speech["ann"],
        "noise_source": noise["source"],
        "snr_db": float(snr_db),
        "augmented": False,
    }


def augment(rec: dict) -> dict:
    gain_db = np.random.default_rng(dload.seeded(rec["key"], "gain")).uniform(-6, 6)
    return {**rec, "audio": rec["audio"] * 10 ** (gain_db / 20), "augmented": True}


def collate(items: list[dict]) -> dict:
    return {
        "audio": np.stack([r["audio"] for r in items]),
        "annotations": [r["ann"] for r in items],
        "noise_source": [r["noise_source"] for r in items],
        "augmented": np.array([r["augmented"] for r in items]),
    }


repo = dload.Repository.open()


def speech_pipe() -> dload.Pipeline:
    return repo.dataset("fsdd").samples().shuffle(seed=0).map(decode_speech)


def noise_choice(p, seed: int = 2) -> dload.Pipeline:
    real = (
        repo.dataset("lab-tones").samples().shuffle(seed=1).repeat().map(decode_noise)
    )
    gen = dload.from_iterable(generated_noise)
    return dload.choice([real, gen], p=p, seed=seed)


def build(p) -> dload.Pipeline:
    return (
        dload.zip_with(mix_at_snr, speech_pipe(), noise_choice(p))
        .maybe(augment, p=AUG_PROB, seed=3)
        .batch(BATCH_SIZE, collate=collate)
    )


# -- 1. fixed p: achieved ratio and augmentation fraction ----------------------

sources: list[str] = []
augmented: list[bool] = []
checksum_a = 0.0
for batch in build(p=0.5).take(N_TAKE // BATCH_SIZE):
    sources += batch["noise_source"]
    augmented += list(batch["augmented"])
    checksum_a += float(np.abs(batch["audio"]).sum())

real_frac = sources.count("real") / len(sources)
aug_frac = sum(augmented) / len(augmented)
print(f"fixed p=0.5 over {len(sources)} examples:")
print(f"  real-noise fraction: {real_frac:.3f}   (target 0.5)")
print(f"  augmented fraction:  {aug_frac:.3f}   (target {AUG_PROB})")
assert abs(real_frac - 0.5) < 0.06
assert abs(aug_frac - AUG_PROB) < 0.06

# -- 2. p as a STREAM: a schedule ramping real-noise share 0 -> 1 --------------

schedule = dload.from_iterable(lambda: (i / N_TAKE for i in range(N_TAKE)))
ramp_sources: list[str] = []
for batch in (
    dload.zip_with(mix_at_snr, speech_pipe(), noise_choice(p=schedule))
    .batch(BATCH_SIZE, collate=collate)
    .take(N_TAKE // BATCH_SIZE)
):
    ramp_sources += batch["noise_source"]

q = len(ramp_sources) // 4
quarters = [ramp_sources[i * q : (i + 1) * q].count("real") / q for i in range(4)]
print(
    f"p from a schedule stream (0 -> 1 ramp), real fraction per quarter: "
    f"{[f'{f:.2f}' for f in quarters]}"
)
assert quarters[0] < 0.30 and quarters[3] > 0.70
assert quarters == sorted(quarters), "ramp must increase monotonically by quarter"

# -- 3. determinism: rebuild the identical DAG, get identical batches ----------

checksum_b = 0.0
for batch in build(p=0.5).take(N_TAKE // BATCH_SIZE):
    checksum_b += float(np.abs(batch["audio"]).sum())

print(f"determinism: checksum_a={checksum_a:.6f} checksum_b={checksum_b:.6f}")
assert checksum_a == checksum_b, "identical DAG + seeds must reproduce exactly"

print()
print(
    "the whole recipe is ~6 lines of composition: choice() picks a noise\n"
    "stream per item (p fixed OR a schedule stream), zip_with() pairs it\n"
    "with speech, maybe() gates augmentation, batch() collates — and the\n"
    "planner/cache/worker machinery still applies, because from_iterable()\n"
    "lifted the generative model INTO the DAG instead of dropping the\n"
    "pipeline out into hand-rolled generator glue."
)
