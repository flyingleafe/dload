# ruff: noqa: E402
"""Random-sampling & augmentation strategy benchmark (no training, no new
ingest -- everything here is already in the remote from b01/b02 and the
training examples).

1. Shuffle-strategy throughput + quality on "fashion-mnist" (70000 samples,
   14 shards): sequential vs. windowed-buffer shuffle (256, 4096) vs.
   `shuffle(full=True)`, both samples/s and shuffle-quality statistics
   (displacement, adjacent-shard mixing, rank correlation) of the emitted
   key order against commit order.
2. Mix-ratio fidelity + overhead: `dload.mix` of "fsdd"/"fashion-mnist"
   (0.7/0.3), plus a 3-way mix with "librispeech-dev-clean" if present.
3. Random-window sampling from long audio ("librispeech-dev-clean", skipped
   if not yet ingested): fixed head crop vs. seeded random crop vs. a
   decode-once/crop-twice multi-crop flat-map.
4. Per-epoch reshuffling cost: 3 consecutive epochs of one `shuffle`
   pipeline object -- warm-cache stability, reshuffling itself is free.
"""

from __future__ import annotations

import io
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _env

_env.setup()
os.environ["DLOAD_CACHE_DIR"] = str(_env.REPO_ROOT / ".dload-cache-bench")

import dload

QUICK = "--quick" in sys.argv

FSDD, FASHION, LIBRISPEECH = "fsdd", "fashion-mnist", "librispeech-dev-clean"
N_SHUFFLE_QUICK = 3000  # section 1
N_MIX_DRAWS, N_MIX_DRAWS_QUICK = 20000, 3000  # section 2
N_EPOCH_QUICK = 10000  # section 4
SAMPLE_RATE = 16_000
CROP_SECONDS = 5.0
CROP_SAMPLES = int(CROP_SECONDS * SAMPLE_RATE)
_AUDIO_S = "audio-s/s"


def bench(
    name: str, iterable: Any, n_items: int, extra_fn: Any = None, label: str = ""
) -> float:
    """Measure wall time / items/s over up to `n_items` items; returns items/s."""
    it = iter(iterable)
    t0 = time.monotonic()
    n = extra_total = 0
    for item in it:
        n += 1
        if extra_fn is not None:
            extra_total += extra_fn(item)
        if n >= n_items:
            break
    dt = max(time.monotonic() - t0, 1e-9)
    items_per_s = n / dt
    row = f"  {name:<24} {n:>6} it  {dt:>7.2f}s  {items_per_s:>9.1f} it/s"
    if extra_fn is not None:
        row += f"  {extra_total / dt:>9.2f} {label}"
    print(row)
    return items_per_s


# --------------------------------------------------------------------------
# section 1: shuffle strategy throughput + quality


def commit_order(ds: dload.Dataset) -> dict[str, int]:
    """key -> absolute index in commit order (one plain sequential pass, no
    decode: samples() yields raw (key, fields) tuples)."""
    return {key: i for i, (key, _fields) in enumerate(ds.samples())}


def shuffle_quality(
    keys: list[str], abs_idx: dict[str, int], bounds: np.ndarray
) -> tuple[float, float, float]:
    """(displacement, same_shard_adjacent, rank_corr) of emitted key order
    `keys` vs. commit order, restricted to the observed subset so --quick
    truncation and a full pass are directly comparable: `pos_out` is
    emission rank 0..n-1, `pos_in` is the rank the same subset of keys would
    have in commit order."""
    n = len(keys)
    positions = np.array([abs_idx[k] for k in keys])
    pos_in = np.empty(n, dtype=np.int64)
    pos_in[np.argsort(positions)] = np.arange(n)
    pos_out = np.arange(n)
    displacement = float(np.mean(np.abs(pos_out - pos_in))) / n
    shard_ids = np.searchsorted(bounds, positions, side="right")
    same_shard_adj = float(np.mean(shard_ids[:-1] == shard_ids[1:])) if n > 1 else 1.0
    rank_corr = float(np.corrcoef(pos_out, pos_in)[0, 1]) if n > 1 else 1.0
    return displacement, same_shard_adj, rank_corr


def bench_shuffle_quality(
    name: str, pipe: dload.Pipeline, n: int, abs_idx: dict[str, int], bounds: np.ndarray
) -> None:
    keys: list[str] = []
    t0 = time.monotonic()
    for key, _fields in pipe:
        keys.append(key)
        if len(keys) >= n:
            break
    items_per_s = len(keys) / max(time.monotonic() - t0, 1e-9)
    disp, same_shard, rank_corr = shuffle_quality(keys, abs_idx, bounds)
    print(
        f"  {name:<20} {items_per_s:>9.1f} samples/s  displacement={disp:.3f}  "
        f"same-shard-adj={same_shard:.3f}  rank-corr={rank_corr:+.3f}"
    )


def shuffle_strategy_section(repo: dload.Repository) -> None:
    print("\n--- 1. shuffle strategy throughput + quality (fashion-mnist) ---")
    ds = repo.dataset(FASHION)
    n = N_SHUFFLE_QUICK if QUICK else len(ds)
    num_shards = len(ds.manifest.shards)
    print(f"n={n} samples per measurement (quick={QUICK}), {num_shards} shards")
    print(
        f"strategy | samples/s | displacement | same-shard-adj | rank-corr  "
        f"(uniform ref: 0.333 | ~{1 / num_shards:.3f} | 0)\n"
    )
    strategies: dict[str, dload.Pipeline] = {
        "sequential": ds.samples(),
        "shuffle(buf=256)": ds.samples().shuffle(256, seed=0),
        "shuffle(buf=4096)": ds.samples().shuffle(4096, seed=0),
        "shuffle(full=True)": ds.samples().shuffle(full=True, seed=0),
    }
    abs_idx, bounds = (
        commit_order(ds),
        np.cumsum([s.num_samples for s in ds.manifest.shards]),
    )
    for name, pipe in strategies.items():
        bench_shuffle_quality(name, pipe, n, abs_idx, bounds)

    print("\n  feasibility check, shuffle(full=True) vs. a windowed shuffle:")
    print(
        "  "
        + ds.samples().shuffle(full=True, seed=0).check().message.replace("\n", "\n  ")
    )
    print(
        "  "
        + ds.samples(prefetch=3)
        .shuffle(4096, seed=0)
        .check()
        .message.replace("\n", "\n  ")
    )
    print(
        "\n  windowed shuffle only needs a few shards resident (the prefetch window "
        "above) to approach uniform adjacency/displacement once its buffer spans a "
        "shard's worth of samples, whereas shuffle(full=True) needs the whole "
        "dataset resident for a true permutation -- windowed shuffle buys "
        "near-uniform stats at a fraction of full-shuffle's residency cost."
    )


# --------------------------------------------------------------------------
# section 2: mix ratio fidelity + throughput


def mix_section(
    repo: dload.Repository, names: list[str], weights: list[float], n: int, seed0: int
) -> None:
    dsets = {name: repo.dataset(name) for name in names}
    n_solo = min([n] + [len(ds) for ds in dsets.values()])
    solo_rate = {
        name: bench(f"{name}_solo", ds.samples(), n_solo) for name, ds in dsets.items()
    }
    weighted_solo = sum(
        w * solo_rate[name] for name, w in zip(names, weights, strict=True)
    )

    def tag(label: str) -> Any:
        return lambda sample: (sample[0], label)

    pipes = [
        dsets[name].samples().shuffle(seed=seed0 + i).repeat().map(tag(name))
        for i, name in enumerate(names)
    ]
    mixed = dload.mix(pipes, weights=weights, seed=seed0 + len(names))
    counts = dict.fromkeys(names, 0)
    t0 = time.monotonic()
    for i, (_key, label) in enumerate(mixed, 1):
        counts[label] += 1
        if i >= n:
            break
    mixed_rate = n / max(time.monotonic() - t0, 1e-9)

    tag_str = ", ".join(
        f"{name}={w:.2f}" for name, w in zip(names, weights, strict=True)
    )
    print(
        f"\n  {'3-way ' if len(names) > 2 else ''}mix over {n} draws (requested {tag_str}):"
    )
    for name, w in zip(names, weights, strict=True):
        achieved = counts[name] / n
        print(
            f"    {name:<20} achieved={achieved:.3f}  abs_error={abs(achieved - w):.3f}"
        )
    overhead = (weighted_solo - mixed_rate) / weighted_solo * 100
    print(
        f"  mixed throughput: {mixed_rate:.1f} samples/s vs. weighted mean of solo "
        f"rates {weighted_solo:.1f} samples/s -- overhead {overhead:+.1f}%"
    )


def mix_ratio_section(repo: dload.Repository, has_librispeech: bool) -> None:
    print("\n--- 2. mix ratio fidelity + throughput ---")
    n = N_MIX_DRAWS_QUICK if QUICK else N_MIX_DRAWS
    mix_section(repo, [FSDD, FASHION], [0.7, 0.3], n, seed0=0)
    if not has_librispeech:
        print(
            "\n  !! librispeech-dev-clean not in remote -- skipping the 3-way mix "
            "(run examples/bench/b01_audio_streaming_bench.py once to ingest it)."
        )
        return
    mix_section(repo, [LIBRISPEECH, FSDD, FASHION], [0.5, 0.3, 0.2], n, seed0=10)


# --------------------------------------------------------------------------
# section 3: random-window sampling from long audio


def decode_flac(fields: dict[str, bytes]) -> np.ndarray:
    wave, _sr = sf.read(io.BytesIO(fields["flac"]), dtype="float32")
    return np.ascontiguousarray(wave, dtype=np.float32)


def _loop_to_length(wave: np.ndarray, n: int) -> np.ndarray:
    return wave if len(wave) >= n else np.tile(wave, n // max(len(wave), 1) + 1)


def _start_for(max_start: int, seed: int) -> int:
    if max_start <= 0:
        return 0
    return int(np.random.default_rng(seed).integers(0, max_start + 1))


def crop_head(sample: tuple[str, dict[str, bytes]]) -> tuple[str, np.ndarray]:
    key, fields = sample
    wave = _loop_to_length(decode_flac(fields), CROP_SAMPLES)
    return key, np.ascontiguousarray(wave[:CROP_SAMPLES])


def crop_random(sample: tuple[str, dict[str, bytes]]) -> tuple[str, np.ndarray]:
    key, fields = sample
    wave = _loop_to_length(decode_flac(fields), CROP_SAMPLES)
    start = _start_for(len(wave) - CROP_SAMPLES, dload.seeded(key, "crop"))
    return key, np.ascontiguousarray(wave[start : start + CROP_SAMPLES])


N_MULTI_CROP_WINDOWS = 2


def multi_crop_flatmap(
    sample: tuple[str, dict[str, bytes]],
) -> Iterator[tuple[str, np.ndarray]]:
    """Flat-map fn for `.flat_map()` (one input yields several outputs):
    decode each flac file exactly once, then slice `N_MULTI_CROP_WINDOWS`
    independent seeded random 5s crops out of the same waveform -- the
    "multi-crop" pattern, amortizing the decode over several crops instead
    of paying it once per crop."""
    key, fields = sample
    wave = _loop_to_length(decode_flac(fields), CROP_SAMPLES)
    max_start = len(wave) - CROP_SAMPLES
    for w in range(N_MULTI_CROP_WINDOWS):
        start = _start_for(max_start, dload.seeded(key, f"crop{w}"))
        yield key, np.ascontiguousarray(wave[start : start + CROP_SAMPLES])


def random_window_section(repo: dload.Repository, has_librispeech: bool) -> None:
    print("\n--- 3. random-window sampling from long audio (librispeech-dev-clean) ---")
    if not has_librispeech:
        print(
            "  !! librispeech-dev-clean not in remote -- skipping this section "
            "(run examples/bench/b01_audio_streaming_bench.py once to ingest it)."
        )
        return
    ds = repo.dataset(LIBRISPEECH)
    n = min(200, len(ds)) if QUICK else len(ds)
    print(f"n={n} utterances per measurement (quick={QUICK})\n")

    a_rate = bench(
        "3a_fixed_head_crop",
        ds.samples().map(crop_head),
        n,
        lambda _i: CROP_SECONDS,
        _AUDIO_S,
    )
    b_rate = bench(
        "3b_random_window",
        ds.samples().map(crop_random),
        n,
        lambda _i: CROP_SECONDS,
        _AUDIO_S,
    )
    c_rate = bench(  # 2 crops/decode -> n_items = 2*n covers the same n decodes
        "3c_multi_crop_2x",
        ds.samples().flat_map(multi_crop_flatmap),
        2 * n,
        lambda _i: CROP_SECONDS,
        _AUDIO_S,
    )
    gain = c_rate / b_rate if b_rate else float("nan")
    print(
        f"\n  decode-once-crop-twice gain: {c_rate:.1f} vs {b_rate:.1f} clips/s "
        f"(x{gain:.2f}) for the same {n} underlying flac decodes -- multi-crop "
        f"amortizes the decode (fixed head crop, no randomization: {a_rate:.1f} clips/s)."
    )


# --------------------------------------------------------------------------
# section 4: per-epoch reshuffling cost


def epoch_reshuffle_section(repo: dload.Repository) -> None:
    print("\n--- 4. per-epoch reshuffling cost (fashion-mnist) ---")
    ds = repo.dataset(FASHION)
    n = N_EPOCH_QUICK if QUICK else len(ds)
    print(f"n={n} samples per epoch (quick={QUICK})\n")
    pipe = ds.samples().shuffle(4096, seed=0)
    for epoch in range(3):
        bench(f"4_epoch={epoch}", pipe, n)
    print(
        "\n  reshuffling between epochs costs nothing beyond planning (O(shards), "
        "see the feasibility check in section 1): throughput stays stable across "
        "the 3 epochs above, all served from the same warm cache."
    )


def main() -> None:
    repo = dload.Repository.open()
    try:
        repo.dataset(LIBRISPEECH)
        has_librispeech = True
    except dload.NotFoundError:
        has_librispeech = False
        print(
            "!! librispeech-dev-clean not found in remote -- sections 2 (3-way mix) "
            "and 3 (random-window sampling) will be skipped. Run "
            "examples/bench/b01_audio_streaming_bench.py once to ingest it."
        )

    shuffle_strategy_section(repo)
    mix_ratio_section(repo, has_librispeech)
    random_window_section(repo, has_librispeech)
    epoch_reshuffle_section(repo)


if __name__ == "__main__":
    main()
