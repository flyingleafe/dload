"""Commit "random lab data that lives only on my machine".

Synthesizes ~200 seeded synthetic audio clips (sine/chirp/noise mixtures,
16 kHz mono, 1-3 s) with JSON annotations, and commits them to R2 as the
"lab-tones" dataset. Because generation is seeded, this script is
idempotent: run it twice and the second run uploads nothing — shards are
content-addressed, so the packer recognizes it already has every one of
them and the commit prints "already in remote" instead of "uploaded".
"""

# ruff: noqa: E402
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _env

_env.setup()

import _labtones
import dload
from dload.config import format_size

N_CLIPS = 200
SEED = 42
TARGET_SHARD_SIZE = 4 * 1024 * 1024

print(f"synthesizing {N_CLIPS} seeded audio clips (seed={SEED})...")
t0 = time.monotonic()
samples = _labtones.clip_samples(N_CLIPS, SEED)
total_wav_bytes = sum(len(fields["wav"]) for _, fields in samples)
print(
    f"generated in {time.monotonic() - t0:.2f}s, "
    f"{format_size(total_wav_bytes)} of raw WAV across {len(samples)} clips"
)

repo = dload.Repository.open()
print(
    f"committing to R2 as 'lab-tones' (target shard size {format_size(TARGET_SHARD_SIZE)})..."
)
t0 = time.monotonic()
manifest = repo.commit(
    "lab-tones",
    samples,
    meta={
        "source": "synthetic-lab",
        "n_samples": str(N_CLIPS),
        "sample_rate": "16000",
        "seed": str(SEED),
    },
    recipe=Path(__file__).read_text(),
    target_shard_size=TARGET_SHARD_SIZE,
    progress=print,
)
elapsed = time.monotonic() - t0

print()
print("manifest summary:")
print(f"  name:      {manifest.name}")
print(f"  version:   {manifest.version}")
print(f"  samples:   {manifest.num_samples}")
print(f"  shards:    {len(manifest.shards)}")
print(f"  size:      {format_size(manifest.total_bytes)}")
print(f"  meta:      {manifest.meta}")
print(f"  commit took {elapsed:.2f}s")
print()
print(
    "re-running this script regenerates the exact same bytes (seeded RNG) — "
    "every shard digest already exists in the remote, so `repo.commit` will "
    "print 'already in remote' for each one and upload nothing new."
)
