"""Derived datasets: memoizing a deterministic preprocessing pipeline.

`repo.derive(name, pipeline)` materializes a finite, deterministic pipeline
ONCE and shares the result: the pipeline's `fingerprint()` (source dataset
versions + DAG shape + transform functions by module/qualname + every seed)
keys a "derivation ref" in the remote. The first caller MISSes, runs the
pipeline, and commits its output as a normal content-addressed dataset. Every
later caller — this process re-running, or a different machine building the
identical pipeline — HITs the ref and streams the memoized snapshot straight
away, no recomputation.

The scenario: "lab-docs", a tiny synthetic text corpus, tokenized against a
fixed vocabulary. Tokenizing is pure CPU work with a stable output — exactly
what you don't want every worker/every epoch/every machine redoing, and
exactly what a lambda-based ad hoc script can't safely cache (no stable
identity to key a cache on). Here the pipeline re-encodes into the encoded
domain itself (token-id arrays as `.npy` bytes), so the memoized dataset is
storable and streams like any other.
"""

# ruff: noqa: E402
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _env

_env.setup()

import numpy as np

import dload
from dload.config import format_size

N_DOCS = 300
DOC_SEED = 123
MIN_WORDS = 4
SHUFFLE_SEED = 7
N_TAKE = 200
TARGET_SHARD_SIZE = 8 * 1024

VOCAB = (
    "the cat sat on mat dog ran fast under tree sun rose over hill bird flew "
    "high into sky fish swam deep river clouds gathered rain fell softly "
    "children played in park music drifted late night stars shone above "
    "quiet town wind blew through open fields horses galloped across plains"
).split()
VOCAB_INDEX = {w: i for i, w in enumerate(VOCAB)}


def synth_documents(n: int, seed: int) -> list[tuple[str, dict[str, bytes]]]:
    """Deterministic (seeded stdlib `random`) synthetic "documents": random
    word salads drawn from a small fixed vocabulary."""
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        n_words = rng.randint(1, 25)
        words = [rng.choice(VOCAB) for _ in range(n_words)]
        samples.append((f"doc-{i:04d}", {"text": " ".join(words).encode("utf-8")}))
    return samples


def has_min_words(sample: tuple[str, dict[str, bytes]]) -> bool:
    """Module-level filter predicate (must be module-level: pickling +
    fingerprinting both reject lambdas/closures)."""
    _key, fields = sample
    return len(fields["text"].decode("utf-8").split()) >= MIN_WORDS


def tokenize(sample: tuple[str, dict[str, bytes]]) -> tuple[str, dict[str, bytes]]:
    """Re-encode into the encoded domain: text bytes -> token-id array bytes
    (`.npy`), against the fixed module-level VOCAB_INDEX. This is the
    preprocessing step we want memoized instead of redone by every consumer."""
    key, fields = sample
    words = fields["text"].decode("utf-8").split()
    ids = np.array([VOCAB_INDEX[w] for w in words], dtype=np.int32)
    return key, {"tokens": dload.codecs.npy_bytes(ids)}


def build_pipeline(repo: dload.Repository) -> dload.Pipeline:
    """Rebuilt from scratch each time, standing in for "a different training
    run / a different machine constructing the identical recipe"."""
    return (
        repo.dataset("lab-docs")
        .samples()
        .shuffle(seed=SHUFFLE_SEED)
        .filter(has_min_words)
        .map(tokenize)
        .take(N_TAKE)
    )


repo = dload.Repository.open()

print(f"synthesizing {N_DOCS} seeded documents (seed={DOC_SEED})...")
docs = synth_documents(N_DOCS, DOC_SEED)
total_bytes = sum(len(fields["text"]) for _, fields in docs)
print(f"generated {len(docs)} documents, {format_size(total_bytes)} of text")

repo.commit(
    "lab-docs",
    docs,
    meta={"source": "synthetic-lab", "n_docs": str(N_DOCS), "seed": str(DOC_SEED)},
    recipe=Path(__file__).read_text(),
    target_shard_size=TARGET_SHARD_SIZE,
    progress=print,
)

# -- 1. first call: MISS, the pipeline actually runs -------------------------

print()
print("deriving 'lab-docs-tokenized' (run 1 — simulates the first machine)...")
t0 = time.monotonic()
derived_1 = repo.derive("lab-docs-tokenized", build_pipeline(repo), progress=print)
elapsed_1 = time.monotonic() - t0
print(f"run 1 took {elapsed_1:.2f}s -> {derived_1!r}")

# -- 2. second call, equivalently-built pipeline: HIT, no recompute ----------

print()
print("deriving 'lab-docs-tokenized' (run 2 — simulates a second machine)...")
t0 = time.monotonic()
derived_2 = repo.derive("lab-docs-tokenized", build_pipeline(repo), progress=print)
elapsed_2 = time.monotonic() - t0
print(f"run 2 took {elapsed_2:.2f}s -> {derived_2!r}")

assert derived_1.version == derived_2.version, (
    "same fingerprint must hit the same snapshot"
)
print(
    f"same version both times ({derived_1.version[:12]}); "
    f"run 2 was {elapsed_1 / max(elapsed_2, 1e-9):.0f}x faster than run 1"
)

# -- 3. stream a few memoized samples ----------------------------------------

print()
print("streaming the memoized (tokenized) samples:")
for key, fields in derived_1.samples().take(3):
    ids = dload.codecs.npy_from(fields["tokens"])
    print(f"  {key}: {len(ids)} tokens, dtype={ids.dtype}, first 8={ids[:8].tolist()}")

# -- 4. identity: fingerprint + source_versions ------------------------------

print()
pipe = build_pipeline(repo)
fp = pipe.fingerprint()
print(f"pipe.fingerprint() = {fp}")
print(f"pipe.source_versions() = {pipe.source_versions()}")

# -- 5. the tag escape hatch: force a fresh snapshot -------------------------

print()
print("deriving again with tag='v2' (simulates a transform code change that")
print("kept the same function name — tag forces a new identity)...")
derived_v2 = repo.derive(
    "lab-docs-tokenized", build_pipeline(repo), tag="v2", progress=print
)
print(f"tag='v2' -> {derived_v2!r}")
assert derived_v2.version != derived_1.version, "a tag must mint a distinct snapshot"

# -- 6. determinism is enforced: unseeded shuffle refuses to fingerprint ----

print()
print("attempting to derive from a pipeline with an UNSEEDED shuffle...")
nondeterministic = (
    repo.dataset("lab-docs").samples().shuffle().filter(has_min_words).map(tokenize)
)
try:
    repo.derive("lab-docs-tokenized-bad", nondeterministic, progress=print)
    raise AssertionError("expected a ValueError")
except ValueError as e:
    print(f"raised ValueError, as required: {e}")

print()
print(
    "the whole feature is one call: `repo.derive(name, pipeline)` — the\n"
    "fingerprint IS the cache key, so any machine building the identical\n"
    "recipe (same source versions, same DAG, same seeds, same named\n"
    "transforms) converges on the same memoized snapshot without a single\n"
    "byte of coordination beyond the shared remote."
)
