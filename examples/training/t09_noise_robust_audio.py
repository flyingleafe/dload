"""Noise-robust spoken-digit classification: streaming mix as augmentation.

`dload.mix()` stochastically interleaves whole pipelines, which is great
for blending corpora — but it isn't the only way two R2-hosted datasets can
combine. Here we pair a finite SIGNAL stream against an endless NOISE
stream with `dload.zip_with`, to synthesize noisy training examples on the
fly:

    noise_pipe  = lab_tones.samples().shuffle(seed=1).repeat().map(decode_noise)
    signal_pipe = fsdd.samples().filter(is_train).shuffle(seed=0).map(decode_signal)
    mixed_pipe  = signal_pipe.zip_with(mix_pair, noise_pipe)

Both datasets stream straight from object storage the whole time — nothing
is pre-downloaded or pre-mixed. "fsdd" (8 kHz spoken digits) is the signal;
"lab-tones" (16 kHz tones/noises, decoded then decimated `[::2]` to 8 kHz)
is the noise. The SNR is drawn per-key in [0, 20] dB so the same clip
always gets the same noise level, epoch after epoch.

To show the augmentation actually buys something, the same small CNN is
trained twice (2 epochs each): model A on clean features only, model B on
noisy-mixed features. Both are evaluated on a clean fsdd eval split and a
fixed-SNR (5 dB) noisy eval split built from it — expect A ~= B on clean,
B clearly ahead on noisy.
"""

# ruff: noqa: E402
from __future__ import annotations

import hashlib
import io
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _env

_env.setup()

import dload
from _common import assert_improved

FRAME_SIZE = 256
HOP = 128
N_MELS = 40
N_FRAMES = 64
SAMPLE_RATE = 8000
BATCH_SIZE = 32
N_EPOCHS = 2
TRAIN_SEED = 0
TRAIN_TAKE = 1500  # keep the 5-min runtime budget even on a slow link
NOISE_SEED_TRAIN = 1
NOISE_SEED_EVAL = 2
TRAIN_SNR_RANGE = (0.0, 20.0)
EVAL_SNR_DB = 5.0

Signal = tuple[str, np.ndarray, int]  # (key, waveform, digit)


def _build_mel_filterbank(n_fft_bins: int, n_mels: int, sr: int) -> np.ndarray:
    """(n_mels, n_fft_bins) triangular mel filterbank, plain numpy (no librosa)."""
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)  # noqa: E731
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)  # noqa: E731
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2)
    bin_points = np.floor(mel_to_hz(mel_points) * (n_fft_bins - 1) / (sr / 2)).astype(
        int
    )
    fb = np.zeros((n_mels, n_fft_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(left, max(center, left + 1)):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, max(right, center + 1)):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


MEL_FB = _build_mel_filterbank(FRAME_SIZE // 2 + 1, N_MELS, SAMPLE_RATE)


def extract_features(wave: np.ndarray) -> np.ndarray:
    """(N_MELS, N_FRAMES) log-mel-ish features: frame -> Hann window -> rfft
    -> triangular mel pooling -> log -> fixed-length pad/crop on the time axis."""
    if len(wave) < FRAME_SIZE:
        wave = np.pad(wave, (0, FRAME_SIZE - len(wave)))
    window = np.hanning(FRAME_SIZE).astype(np.float32)
    n_frames = 1 + (len(wave) - FRAME_SIZE) // HOP
    frames = np.stack(
        [wave[i * HOP : i * HOP + FRAME_SIZE] * window for i in range(n_frames)]
    )
    magnitude = np.abs(np.fft.rfft(frames, axis=-1)).astype(np.float32)
    log_mel = np.log1p(magnitude @ MEL_FB.T).astype(np.float32)  # (n_frames, N_MELS)
    if log_mel.shape[0] >= N_FRAMES:
        log_mel = log_mel[:N_FRAMES]
    else:
        pad = np.zeros((N_FRAMES - log_mel.shape[0], N_MELS), dtype=np.float32)
        log_mel = np.concatenate([log_mel, pad], axis=0)
    return log_mel.T  # (N_MELS, N_FRAMES)


def split_bucket(sample: tuple[str, dict[str, bytes]]) -> int:
    """Deterministic 0-9 bucket from the sample key (same scheme as t01)."""
    key, _fields = sample
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10


def is_train(sample: tuple[str, dict[str, bytes]]) -> bool:
    return split_bucket(sample) < 8


def is_eval(sample: tuple[str, dict[str, bytes]]) -> bool:
    return not is_train(sample)


def decode_signal(sample: tuple[str, dict[str, bytes]]) -> Signal:
    key, fields = sample
    wave, _sr = sf.read(io.BytesIO(fields["wav"]), dtype="float32")
    digit = dload.codecs.json_from(fields["meta"])["digit"]
    return key, wave, digit


def decode_noise(sample: tuple[str, dict[str, bytes]]) -> np.ndarray:
    """lab-tones is 16 kHz; decimate by 2 (plain slicing) to match fsdd's 8 kHz."""
    _key, fields = sample
    wave, _sr = sf.read(io.BytesIO(fields["wav"]), dtype="float32")
    return np.ascontiguousarray(wave[::2], dtype=np.float32)


def _fit_length(wave: np.ndarray, length: int) -> np.ndarray:
    if len(wave) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(wave) >= length:
        return wave[:length]
    reps = length // len(wave) + 1
    return np.tile(wave, reps)[:length]


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` so that signal + noise lands at `snr_db`, then add."""
    noise = _fit_length(noise, len(signal))
    sig_power = float(np.mean(signal.astype(np.float64) ** 2)) + 1e-10
    noise_power = float(np.mean(noise.astype(np.float64) ** 2)) + 1e-10
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    return (signal + noise * scale).astype(np.float32)


def random_snr_db(key: str) -> float:
    """Per-key SNR in TRAIN_SNR_RANGE: the same clip always gets the same
    noise level, epoch after epoch, without any shared mutable rng."""
    rng = np.random.default_rng(dload.seeded(key, "snr"))
    return float(rng.uniform(*TRAIN_SNR_RANGE))


def to_clean(sig: Signal) -> tuple[np.ndarray, int]:
    _key, wave, digit = sig
    return extract_features(wave), digit


def mix_pair(
    sig: Signal, noise_wave: np.ndarray, *, snr_db_fn: Callable[[str], float]
) -> tuple[np.ndarray, int]:
    """`dload.zip_with(partial(mix_pair, snr_db_fn=...), signal_pipe,
    noise_pipe)` pairs a finite signal stream positionally against an
    endless noise stream, one node inside the pipeline DAG."""
    key, wave, digit = sig
    mixed = mix_at_snr(wave, noise_wave, snr_db_fn(key))
    return extract_features(mixed), digit


def collate(items: list[tuple[np.ndarray, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    features = np.stack([feat for feat, _digit in items])[:, None, :, :]  # (B,1,40,64)
    labels = np.array([digit for _feat, digit in items], dtype=np.int64)
    return torch.from_numpy(features), torch.from_numpy(labels)


class DigitCNN(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16, 20, 32)
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 10, 16)
        )
        self.classifier = nn.Linear(32 * 10 * 16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.flatten(1))


def train(
    model: DigitCNN,
    make_batches: Callable[[], dload.Pipeline],
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in make_batches():  # fresh reshuffled epoch each call
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        epoch_loss = total_loss / n_batches
        losses.append(epoch_loss)
        print(f"    epoch {epoch}: loss {epoch_loss:.4f} ({n_batches} batches)")
    return losses


def evaluate(model: DigitCNN, features: list[tuple[np.ndarray, int]]) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dload.from_iterable(features).batch(BATCH_SIZE, collate=collate):
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += y.shape[0]
    return correct / total


def build_train_signal_pipe(repo: dload.Repository) -> dload.Pipeline:
    return (
        repo.dataset("fsdd")
        .samples()
        .filter(is_train)
        .shuffle(seed=TRAIN_SEED)
        .map(decode_signal)
        .take(TRAIN_TAKE)
    )


def build_noise_pipe(repo: dload.Repository, seed: int) -> dload.Pipeline:
    return (
        repo.dataset("lab-tones")
        .samples()
        .shuffle(seed=seed)
        .repeat()
        .map(decode_noise)
    )


def main() -> None:
    torch.manual_seed(0)
    t0 = time.monotonic()

    repo = dload.Repository.open()
    fsdd = repo.dataset("fsdd")
    lab_tones = repo.dataset("lab-tones")
    print(f"datasets: {fsdd!r}, {lab_tones!r}")

    print("model A: training on clean features only")
    model_a = DigitCNN()
    losses_a = train(
        model_a,
        lambda: (
            build_train_signal_pipe(repo)
            .map(to_clean)
            .batch(BATCH_SIZE, collate=collate)
        ),
    )

    print("model B: training on noisy-mixed features (fsdd x lab-tones)")
    model_b = DigitCNN()
    losses_b = train(
        model_b,
        lambda: (
            build_train_signal_pipe(repo)
            .zip_with(
                partial(mix_pair, snr_db_fn=random_snr_db),
                build_noise_pipe(repo, NOISE_SEED_TRAIN),
            )
            .batch(BATCH_SIZE, collate=collate)
        ),
    )

    print("building eval sets from the fsdd eval split (clean + noisy @5dB)...")
    eval_signal_pipe = fsdd.samples().filter(is_eval).map(decode_signal)
    eval_clean = list(eval_signal_pipe.map(to_clean))

    eval_signal_pipe2 = fsdd.samples().filter(is_eval).map(decode_signal)
    eval_noise_pipe = build_noise_pipe(repo, NOISE_SEED_EVAL)
    eval_noisy = list(
        eval_signal_pipe2.zip_with(
            partial(mix_pair, snr_db_fn=lambda _key: EVAL_SNR_DB), eval_noise_pipe
        )
    )
    print(f"  eval set: {len(eval_clean)} clean / {len(eval_noisy)} noisy clips")

    results = {
        ("A", "clean"): evaluate(model_a, eval_clean),
        ("A", "noisy"): evaluate(model_a, eval_noisy),
        ("B", "clean"): evaluate(model_b, eval_clean),
        ("B", "noisy"): evaluate(model_b, eval_noisy),
    }

    print()
    print("accuracy table (rows: model, cols: eval set)")
    print(f"{'':>8}{'clean':>10}{'noisy@5dB':>12}")
    for model_name in ("A", "B"):
        row = "".join(
            f"{results[(model_name, ev)]:>10.3f}   " for ev in ("clean", "noisy")
        )
        label = "A (clean-trained)" if model_name == "A" else "B (noisy-trained)"
        print(f"{label:>18}: {row}")
    print(f"total runtime: {time.monotonic() - t0:.1f}s")

    assert_improved(losses_a[0], losses_a[-1])
    assert_improved(losses_b[0], losses_b[-1])
    assert results[("A", "clean")] > 0.7, (
        f"model A clean accuracy too low: {results[('A', 'clean')]:.3f}"
    )
    assert results[("B", "clean")] > 0.7, (
        f"model B clean accuracy too low: {results[('B', 'clean')]:.3f}"
    )
    assert results[("B", "noisy")] > 0.5, (
        f"model B noisy accuracy too low: {results[('B', 'noisy')]:.3f}"
    )
    assert results[("B", "noisy")] > results[("A", "noisy")] + 0.1, (
        f"noise-mixed training did not clearly help: "
        f"A={results[('A', 'noisy')]:.3f} B={results[('B', 'noisy')]:.3f}"
    )
    print(
        "PASS: mixing an endless noise stream into a finite signal stream via "
        "dload.zip_with, purely from object storage, buys real robustness — "
        "clean accuracy is preserved, noisy accuracy is not."
    )


if __name__ == "__main__":
    main()
