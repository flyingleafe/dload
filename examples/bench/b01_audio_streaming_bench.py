# ruff: noqa: E402
"""Heavyweight audio ingestion + streaming benchmark (no model training).

Ingests LibriSpeech dev-clean (~337 MB, ~2703 flac utterances, ~5.4h speech)
once, then measures each layer of the streaming pipeline in isolation: raw
R2 download throughput, warm-cache disk throughput, flac decode throughput,
feature extraction (80-bin log-mel over a seeded random 5s crop), a full
SpecAugment-style augmentation pipeline, the prefetch window's effect, and
DataLoader worker scaling. Flac bytes are stored *compressed* in the shards
and decoded only inside the pipeline (`sf.read` in `.map`) -- the standard
dload tradeoff of ~1/3 the shard bytes and R2 egress against a decode step
every epoch (same choice `03_stream_features.py` makes for "lab-tones").
Ends with a back-of-envelope H100 feasibility check.
"""

from __future__ import annotations

import inspect
import io
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import _env

_env.setup()
os.environ["DLOAD_CACHE_DIR"] = str(_env.REPO_ROOT / ".dload-cache-bench")

import dload
from _common import ensure_dataset, http_download  # pyright: ignore[reportMissingImports]
from dload.torch import as_iterable_dataset

QUICK = "--quick" in sys.argv
QUICK_N = 200

LIBRISPEECH_URL = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
DATASET_NAME = "librispeech-dev-clean"
TARGET_SHARD_SIZE = 32 * 1024 * 1024

SAMPLE_RATE = 16_000
CROP_SECONDS = 5.0
CROP_SAMPLES = int(CROP_SECONDS * SAMPLE_RATE)
FRAME_SIZE = 400  # 25 ms
HOP = 160  # 10 ms -> 100 frames/s
N_MELS = 80
N_FRAMES = 1 + (CROP_SAMPLES - FRAME_SIZE) // HOP  # ~498, ~= the 500 used below


def download_librispeech():
    """Download LibriSpeech dev-clean (cached in .downloads/, ~337 MB) and
    yield one sample per utterance: flac bytes kept compressed as-is (the
    standard store-compressed/decode-in-pipeline tradeoff) plus a json meta
    blob {speaker, chapter, text}."""
    tar_path = http_download(LIBRISPEECH_URL, "dev-clean.tar.gz")
    transcripts: dict[str, str] = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if member.name.endswith(".trans.txt"):
                f = tf.extractfile(member)
                assert f is not None
                for line in f.read().decode("utf-8").splitlines():
                    utt_id, _, text = line.partition(" ")
                    transcripts[utt_id] = text
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.name.endswith(".flac"):
                continue
            utt_id = Path(member.name).stem
            speaker, chapter, _rest = utt_id.split("-")
            f = tf.extractfile(member)
            assert f is not None
            meta = dload.codecs.json_bytes(
                {
                    "speaker": speaker,
                    "chapter": chapter,
                    "text": transcripts.get(utt_id, ""),
                }
            )
            yield utt_id, {"flac": f.read(), "meta": meta}


def ingest(repo: dload.Repository) -> dload.Dataset:
    try:
        ds = repo.dataset(DATASET_NAME)
        print(f"dataset {DATASET_NAME!r}: {ds!r} (already ingested)")
        return ds
    except dload.NotFoundError:
        pass
    print(f"dataset {DATASET_NAME!r} not in remote yet -- downloading + committing")
    t0 = time.monotonic()
    ds = ensure_dataset(
        repo,
        DATASET_NAME,
        download_librispeech,
        recipe=inspect.getsource(download_librispeech),
        meta={"source": "openslr.org/resources/12 (LibriSpeech dev-clean)"},
        target_shard_size=TARGET_SHARD_SIZE,
    )
    dt = time.monotonic() - t0
    mb = ds.manifest.total_bytes / 2**20
    print(f"ingest done: {dt:.1f}s, {mb:.1f} MB, {mb / dt:.2f} MB/s")
    return ds


# decode / features / augment below are module-level so they survive
# pickling into spawned DataLoader workers (see 06_torch_dataloader.py).


def _build_mel_filterbank(n_fft_bins: int, n_mels: int, sr: int) -> np.ndarray:
    """(n_mels, n_fft_bins) triangular mel filterbank, plain numpy."""
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)  # noqa: E731
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)  # noqa: E731
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2)
    bins = np.floor(mel_to_hz(mel_points) * (n_fft_bins - 1) / (sr / 2)).astype(int)
    fb = np.zeros((n_mels, n_fft_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, max(center, left + 1)):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, max(right, center + 1)):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


MEL_FB = _build_mel_filterbank(FRAME_SIZE // 2 + 1, N_MELS, SAMPLE_RATE)


def decode(sample: tuple[str, dict[str, bytes]]) -> tuple[str, np.ndarray]:
    key, fields = sample
    wave, _sr = sf.read(io.BytesIO(fields["flac"]), dtype="float32")
    return key, np.ascontiguousarray(wave, dtype=np.float32)


def _seeded_rng(*parts: str) -> np.random.Generator:
    return np.random.default_rng(dload.seeded(*parts))


def extract_logmel(wave: np.ndarray) -> np.ndarray:
    """(N_MELS, n_frames) log-mel: frame -> Hann -> rfft -> mel -> log."""
    window = np.hanning(FRAME_SIZE).astype(np.float32)
    n_frames = 1 + (len(wave) - FRAME_SIZE) // HOP
    frames = np.stack(
        [wave[i * HOP : i * HOP + FRAME_SIZE] * window for i in range(n_frames)]
    )
    magnitude = np.abs(np.fft.rfft(frames, axis=-1)).astype(np.float32)
    return np.log1p(magnitude @ MEL_FB.T).astype(np.float32).T


def features(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray]:
    """Random (seeded-per-key, reproducible) 5s crop -> log-mel; short clips loop."""
    key, wave = item
    if len(wave) < CROP_SAMPLES:
        wave = np.tile(wave, CROP_SAMPLES // max(len(wave), 1) + 1)
    max_start = len(wave) - CROP_SAMPLES
    start = (
        int(_seeded_rng(key, "crop").integers(0, max_start + 1)) if max_start > 0 else 0
    )
    crop = np.ascontiguousarray(wave[start : start + CROP_SAMPLES], dtype=np.float32)
    return key, extract_logmel(crop)


def augment(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray]:
    """SpecAugment-style: 2 time masks + 2 freq masks + random gain."""
    key, feat = item
    feat = feat.copy()
    rng = _seeded_rng(key, "aug")
    feat += rng.uniform(-2.0, 2.0)  # random gain, log domain
    n_mels, n_frames = feat.shape
    for _ in range(2):
        w = int(rng.integers(1, max(n_frames // 10, 2)))
        t0 = int(rng.integers(0, max(n_frames - w, 1)))
        feat[:, t0 : t0 + w] = 0.0
    for _ in range(2):
        w = int(rng.integers(1, max(n_mels // 10, 2)))
        f0 = int(rng.integers(0, max(n_mels - w, 1)))
        feat[f0 : f0 + w, :] = 0.0
    return key, feat


def collate_feat(items: list[tuple[str, np.ndarray]]) -> np.ndarray:
    return np.stack([f for _key, f in items])


def bench(
    name: str,
    iterable: Any,
    n_items: int,
    bytes_per_item_fn: Any = None,
    extra_fn: Any = None,
    extra_label: str = "",
) -> float:
    """Measure wall time / items/s / approx MB/s over up to `n_items` items;
    returns items/s."""
    it = iter(iterable)
    t0 = time.monotonic()
    n = nbytes = extra_total = 0
    for item in it:
        n += 1
        if bytes_per_item_fn is not None:
            nbytes += bytes_per_item_fn(item)
        if extra_fn is not None:
            extra_total += extra_fn(item)
        if n >= n_items:
            break
    dt = max(time.monotonic() - t0, 1e-9)
    items_per_s = n / dt
    mb_per_s = (nbytes / 2**20) / dt if bytes_per_item_fn is not None else None
    extra_per_s = extra_total / dt if extra_fn is not None else None
    row = f"  {name:<22} {n:>6} it  {dt:>7.2f}s  {items_per_s:>9.1f} it/s"
    if mb_per_s is not None:
        row += f"  {mb_per_s:>9.2f} MB/s"
    if extra_per_s is not None:
        row += f"  {extra_per_s:>9.2f} {extra_label}"
    print(row)
    return items_per_s


def flac_bytes(sample: tuple[str, dict[str, bytes]]) -> int:
    return len(sample[1]["flac"])


_AUDIO_S = "audio-s/s"


def run_benchmarks(repo: dload.Repository, ds: dload.Dataset) -> float:
    n = QUICK_N if QUICK else len(ds)
    print(f"\nbenchmarking with n={n} samples per measurement (quick={QUICK})\n")

    repo.cache.clear()  # 1-2: cold vs warm sequential raw flac bytes (R2 vs disk)
    bench("1_cold_sequential", ds.samples(), n, bytes_per_item_fn=flac_bytes)
    bench("2_warm_sequential", ds.samples(), n, bytes_per_item_fn=flac_bytes)

    bench(  # 3: flac -> float32 waveform
        "3_warm_decode",
        ds.samples().map(decode),
        n,
        extra_fn=lambda item: len(item[1]) / SAMPLE_RATE,
        extra_label="audio-s/s (realtime x)",
    )

    feat_clips_s = bench(  # 4: + 5s crop + 80-bin log-mel
        "4_warm_decode_features",
        ds.samples().map(decode).map(features),
        n,
        extra_fn=lambda _item: CROP_SECONDS,
        extra_label=_AUDIO_S,
    )

    aug_clips_s = bench(  # 5: + SpecAugment-style masking + gain
        "5_warm_augmented",
        ds.samples().map(decode).map(features).map(augment),
        n,
        extra_fn=lambda _item: CROP_SECONDS,
        extra_label=_AUDIO_S,
    )

    for p in (2, 4, 8):  # 6: prefetch sweep, warm_decode_features, shuffled
        pipe = ds.samples(prefetch=p).shuffle(seed=0).map(decode).map(features)
        bench(
            f"6_prefetch={p}",
            pipe,
            n,
            extra_fn=lambda _item: CROP_SECONDS,
            extra_label=_AUDIO_S,
        )

    base_pipe = ds.samples().map(decode).map(features).batch(16, collate=collate_feat)
    n_batches = max(1, n // 16)
    for nw in (0, 2):  # 7: torch DataLoader worker scaling, batch=16
        loader = torch.utils.data.DataLoader(
            as_iterable_dataset(base_pipe), batch_size=None, num_workers=nw
        )
        bench(f"7_num_workers={nw}", loader, n_batches, extra_label="batches/s")

    print(
        f"\n(clips/s: {feat_clips_s:.1f} bare features, {aug_clips_s:.1f} with augmentation)"
    )
    return aug_clips_s


def h100_feasibility(measured_clips_per_s: float) -> None:
    """FLOPs-based estimate of clips/s an H100 (or a few) needs fed to it to
    stay busy training a Conformer/Whisper-class encoder, vs. what this
    box's warm_augmented feature pipeline actually delivers."""
    tokens_per_clip = 500  # 5s * 100 frames/s (this pipeline's real N_FRAMES ~= 498)
    batch_per_gpu = 64
    peak_flops = 990e12  # H100 SXM, bf16 dense
    mfu = 0.40
    effective_flops = peak_flops * mfu

    print("\n" + "=" * 78)
    print("H100 FEASIBILITY: pretraining a Conformer/Whisper-class audio encoder")
    print("=" * 78)
    print(
        f"assumptions: clip={CROP_SECONDS:.0f}s, tokens/clip={tokens_per_clip} (100 frames/s), "
        f"batch/gpu={batch_per_gpu}\n             H100 bf16 peak={peak_flops:.2e} FLOPs/s, MFU={mfu:.0%}"
    )
    print(
        "formulas: flops/step/gpu = 6*N*tokens_per_clip*batch_per_gpu (fwd+bwd, dense)\n"
        "          step_time = flops/step/gpu / (peak_flops*MFU)\n"
        "          required_clips/s = batch_per_gpu*n_gpus / step_time\n"
    )
    print(
        f"  {'N params':>9} {'n_gpus':>7} {'step_ms':>9} {'req clips/s':>12} {'x this box':>11}"
    )
    worst_mult, worst_desc = 0.0, ""
    for n_params in (0.1e9, 0.3e9, 0.5e9):
        step_time = 6 * n_params * tokens_per_clip * batch_per_gpu / effective_flops
        for n_gpus in (1, 2, 3, 4):
            required = batch_per_gpu * n_gpus / step_time
            mult = required / measured_clips_per_s
            print(
                f"  {n_params / 1e9:>7.1f}B {n_gpus:>7} {step_time * 1000:>8.1f}ms"
                f" {required:>12.1f} {mult:>10.1f}x"
            )
            if mult > worst_mult:
                worst_mult, worst_desc = (
                    mult,
                    f"{n_params / 1e9:.1f}B params x {n_gpus} GPUs",
                )

    print(
        f"\nthis box's measured warm_augmented throughput: {measured_clips_per_s:.1f} clips/s"
    )
    print(
        f"worst case ({worst_desc}) needs {worst_mult:.0f}x that -- e.g. ~{worst_mult:.0f} cores "
        f"of loader like this one (has 2), or a handful of typical 16-32 core dataloader hosts.\n"
        "dload's SourceNode splits shards[worker::num_workers] with exact per-epoch coverage and "
        "no duplicate downloads (see torch.py), so this scales close to linearly with independent "
        "loader processes/hosts -- horizontal, not single-box, is the fix for small-model / "
        "large-GPU-count data starvation."
    )


def main() -> None:
    repo = dload.Repository.open()
    ds = ingest(repo)
    clips_per_s = run_benchmarks(repo, ds)
    h100_feasibility(clips_per_s)


if __name__ == "__main__":
    main()
