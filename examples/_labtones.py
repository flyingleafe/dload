"""Shared synthetic-audio generator for 01_commit_lab_data.py and
07_versioning_pinning.py — kept in one place so both scripts pack *exactly*
the same bytes for the same (seed, index), which is what makes the
content-addressed dedup demos in both scripts work.

Each clip is a mixture of a tone, a linear chirp and white noise (picked by
label), 16 kHz mono, 1-3 s, encoded as a float64 WAV. An annotation JSON
(label, snr_db, duration_s) rides alongside it.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

import dload

SAMPLE_RATE = 16_000
LABELS = ("tone", "chirp", "noise", "mix")


def synth_clip(index: int, seed: int) -> tuple[bytes, dict]:
    """Deterministic clip for (seed, index): (wav_bytes, annotation)."""
    rng = np.random.default_rng((seed, index))
    label = LABELS[rng.integers(len(LABELS))]
    duration = float(rng.uniform(1.0, 3.0))
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE

    if label == "tone":
        freq = rng.uniform(220, 1800)
        sig = np.sin(2 * np.pi * freq * t)
    elif label == "chirp":
        f0, f1 = rng.uniform(150, 500), rng.uniform(2000, 6000)
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
        sig = np.sin(phase)
    elif label == "noise":
        sig = rng.standard_normal(n)
    else:  # mix
        freq = rng.uniform(220, 1800)
        sig = 0.6 * np.sin(2 * np.pi * freq * t) + 0.4 * rng.standard_normal(n)
    sig = sig / (np.max(np.abs(sig)) + 1e-9)

    snr_db = float(rng.uniform(-5, 20))
    noise = rng.standard_normal(n)
    sig_power = np.mean(sig**2)
    noise_power = np.mean(noise**2) + 1e-12
    noise *= np.sqrt((sig_power / (10 ** (snr_db / 10))) / noise_power)

    out = sig + noise
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0:
        out = out / peak

    # PCM_32, not FLOAT/DOUBLE: libsndfile stamps float WAVs with a PEAK
    # chunk that embeds a wall-clock timestamp, which would make "identical
    # seed -> identical bytes" false across separate runs of this script.
    buf = io.BytesIO()
    sf.write(buf, out.astype(np.float64), SAMPLE_RATE, format="WAV", subtype="PCM_32")
    annotation = {
        "label": label,
        "snr_db": round(snr_db, 2),
        "duration_s": round(duration, 4),
    }
    return buf.getvalue(), annotation


def clip_samples(n: int, seed: int, key_prefix: str = "clip") -> list[tuple[str, dict]]:
    """n deterministic (key, {"wav": ..., "meta": ...}) samples."""
    out = []
    for i in range(n):
        wav, annotation = synth_clip(i, seed)
        key = f"{key_prefix}-{i:04d}"
        out.append((key, {"wav": wav, "meta": dload.codecs.json_bytes(annotation)}))
    return out
