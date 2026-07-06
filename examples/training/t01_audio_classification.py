"""Supervised audio classification: spoken-digit recognition on FSDD.

Trains a tiny CNN to classify which digit (0-9) is spoken in an 8 kHz WAV
clip, using log-mel-ish features computed with plain numpy. The dload
pattern showcased here is a **deterministic hash-split + per-epoch
reshuffling straight from object storage**: train/eval membership is
decided by `sha1(key) % 10` (no held-out file list to keep in sync), and
the train pipeline is a single `dload.Pipeline` object that gets
`for`-looped over once per epoch — dload reshuffles it fresh (new shard
order, new in-buffer order) each time, entirely driven by the R2-backed
dataset "fsdd".
"""

# ruff: noqa: E402
from __future__ import annotations

import hashlib
import io
import sys
import time
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
N_EPOCHS = 3
TRAIN_SEED = 0


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
    """Deterministic 0-9 bucket from the sample key: no held-out file list
    to keep in sync, and it's stable across re-commits of the dataset."""
    key, _fields = sample
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10


def is_train(sample: tuple[str, dict[str, bytes]]) -> bool:
    return split_bucket(sample) < 8


def decode(sample: tuple[str, dict[str, bytes]]) -> tuple[np.ndarray, int]:
    _key, fields = sample
    wave, _sr = sf.read(io.BytesIO(fields["wav"]), dtype="float32")
    digit = dload.codecs.json_from(fields["meta"])["digit"]
    return extract_features(wave), digit


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


def main() -> None:
    torch.manual_seed(0)
    t0 = time.monotonic()

    repo = dload.Repository.open()
    ds = repo.dataset("fsdd")
    print(f"dataset: {ds!r}")

    train_pipe = (
        ds.samples()
        .filter(is_train)
        .shuffle(seed=TRAIN_SEED)
        .map(decode)
        .batch(BATCH_SIZE, collate=collate)
    )
    is_eval = lambda s: not is_train(s)  # noqa: E731
    eval_pipe = (
        ds.samples().filter(is_eval).map(decode).batch(BATCH_SIZE, collate=collate)
    )

    n_train = sum(1 for _ in filter(is_train, ds.samples()))
    print(
        f"split: {n_train} train / {len(ds) - n_train} eval (hash-bucketed, deterministic)"
    )

    model = DigitCNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: DigitCNN, {n_params} params")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses: list[float] = []
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in train_pipe:  # fresh reshuffled epoch each time round
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        epoch_loss = total_loss / n_batches
        losses.append(epoch_loss)
        print(f"epoch {epoch}: loss {epoch_loss:.4f} ({n_batches} batches)")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in eval_pipe:
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += y.shape[0]
    accuracy = correct / total
    print(f"eval accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"total runtime: {time.monotonic() - t0:.1f}s")

    assert_improved(losses[0], losses[-1])
    assert accuracy > 0.5, f"eval accuracy too low: {accuracy:.3f}"
    print(
        "PASS: dload streams a deterministic hash-split with per-epoch "
        "reshuffling straight from R2, and the model actually learns."
    )


if __name__ == "__main__":
    main()
