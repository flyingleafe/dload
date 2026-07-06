"""SUPERVISED TABULAR REGRESSION on the UCI Wine Quality dataset.

Ingest: red + white wine CSVs (~6.5k rows, 11 physicochemical features,
integer quality 0-10), one sample per row, features stored as `.npy` bytes
via `dload.codecs.npy_bytes` (numeric fields, vs. the usual audio/text).

Patterns: pipelines are lazy and re-iterable, so a first pass streams the
train split just to accumulate per-feature mean/std with plain numpy, and a
second pass re-streams the *same* shards (now warm in the local cache) to
train — the wall-clock gap shows the cache at work. Train/eval is a
`.filter` over a stable hash of the sample key, not a stored column.
"""

# ruff: noqa: E402
from __future__ import annotations

import csv
import hashlib
import inspect
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import _env

_env.setup()

import dload
from _common import assert_improved, ensure_dataset, http_download

WINE_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]  # fmt: skip
N_FEATURES = len(FEATURE_NAMES)
BATCH_SIZE = 256
N_EPOCHS = 6
LR = 2e-2


def download_and_parse_wine_quality():
    """Fetch the UCI Wine Quality zip and yield one sample per row of each
    color's CSV: 11 float features (.npy bytes), quality (1 byte, 0-10),
    color (b"red"/b"white"). Handles the CSVs living at any depth in the
    zip (some UCI mirrors nest them in a directory)."""
    zip_path = http_download(WINE_URL, "wine-quality.zip")
    with zipfile.ZipFile(zip_path) as zf:
        by_basename = {Path(n).name: n for n in zf.namelist()}
        for color in ("red", "white"):
            member = by_basename[f"winequality-{color}.csv"]
            text = zf.read(member).decode("utf-8")
            rows = csv.reader(io.StringIO(text), delimiter=";")
            next(rows)  # header
            for i, row in enumerate(rows):
                values = [float(x) for x in row]
                features = np.array(values[:N_FEATURES], dtype=np.float32)
                quality = int(values[N_FEATURES])
                yield (
                    f"{color}-{i:04d}",
                    {
                        "features": dload.codecs.npy_bytes(features),
                        "quality": bytes([quality]),
                        "color": color.encode(),
                    },
                )


def split_bucket(key: str) -> int:
    """Stable 0-99 bucket from a sha256 of the key, for a deterministic
    hash-based train/eval split that needs no stored column."""
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100


def is_train(sample: tuple[str, dict[str, bytes]]) -> bool:
    return split_bucket(sample[0]) < 80


def is_eval(sample: tuple[str, dict[str, bytes]]) -> bool:
    return not is_train(sample)


def make_normalize(mean: np.ndarray, std: np.ndarray):
    def normalize(sample: tuple[str, dict[str, bytes]]) -> tuple[np.ndarray, float]:
        _key, fields = sample
        features = dload.codecs.npy_from(fields["features"]).astype(np.float32)
        features = (features - mean) / std
        return features, float(fields["quality"][0])

    return normalize


def collate(items: list[tuple[np.ndarray, float]]) -> tuple[torch.Tensor, torch.Tensor]:
    features = np.stack([f for f, _q in items])
    targets = np.array([q for _f, q in items], dtype=np.float32)
    return torch.from_numpy(features), torch.from_numpy(targets)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


torch.manual_seed(0)
t_start = time.monotonic()

repo = dload.Repository.open()
ds = ensure_dataset(
    repo, "wine-quality", download_and_parse_wine_quality,
    recipe=inspect.getsource(download_and_parse_wine_quality),
    meta={"source": WINE_URL, "feature_names": FEATURE_NAMES},
)  # fmt: skip
print(f"dataset: {ds!r}")

# -- pass 1: stream the train split once to accumulate feature/target stats.
t0 = time.monotonic()
n = 0
feat_sum, feat_sumsq = np.zeros(N_FEATURES), np.zeros(N_FEATURES)
q_sum = q_sumsq = 0.0
for _key, fields in ds.samples().filter(is_train):
    f = dload.codecs.npy_from(fields["features"]).astype(np.float64)
    feat_sum += f
    feat_sumsq += f * f
    q = float(fields["quality"][0])
    q_sum += q
    q_sumsq += q * q
    n += 1
pass1_s = time.monotonic() - t0
mean = (feat_sum / n).astype(np.float32)
std = np.sqrt(np.maximum(feat_sumsq / n - (feat_sum / n) ** 2, 1e-8)).astype(np.float32)
train_mean_quality = q_sum / n
print(f"pass 1 (stats over {n} train rows): {pass1_s:.2f}s -- cold shards downloaded")

# -- pass 2: re-stream the same shards, now warm, to train.
normalize = make_normalize(mean, std)
train_pipe = (
    ds.samples()
    .filter(is_train)
    .shuffle(4096, seed=0)
    .map(normalize)
    .batch(BATCH_SIZE, collate=collate)
)
eval_pipe = (
    ds.samples().filter(is_eval).map(normalize).batch(BATCH_SIZE, collate=collate)
)

t1 = time.monotonic()
first_batch = next(iter(train_pipe))
print(
    f"pass 2 first batch (warm cache): {(time.monotonic() - t1) * 1000:.1f} ms "
    f"(vs {pass1_s * 1000:.0f} ms for the whole of pass 1)"
)
print(
    f"  features {tuple(first_batch[0].shape)}, targets {tuple(first_batch[1].shape)}"
)

model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=LR)
epoch_losses: list[float] = []
for epoch in range(N_EPOCHS):
    total, seen = 0.0, 0
    for features, targets in train_pipe:
        opt.zero_grad()
        preds = model(features)
        loss = nn.functional.mse_loss(preds, targets)
        loss.backward()
        opt.step()
        total += loss.item() * len(targets)
        seen += len(targets)
    epoch_losses.append(total / seen)
    print(f"epoch {epoch}: train MSE {epoch_losses[-1]:.4f}")

assert_improved(epoch_losses[0], epoch_losses[-1], name="train MSE")

# -- eval: model RMSE/MAE vs the train-mean baseline.
model.eval()
se = ae = se_baseline = 0.0
n_eval = 0
with torch.no_grad():
    for features, targets in eval_pipe:
        preds = model(features)
        se += ((preds - targets) ** 2).sum().item()
        ae += (preds - targets).abs().sum().item()
        se_baseline += ((targets - train_mean_quality) ** 2).sum().item()
        n_eval += len(targets)

rmse = (se / n_eval) ** 0.5
mae = ae / n_eval
baseline_rmse = (se_baseline / n_eval) ** 0.5
print(
    f"eval ({n_eval} rows): RMSE {rmse:.4f}, MAE {mae:.4f}, baseline RMSE {baseline_rmse:.4f}"
)

assert rmse < 0.92 * baseline_rmse, (
    f"model RMSE {rmse:.4f} not < 0.92x baseline {baseline_rmse:.4f}"
)
print(f"model beats the train-mean baseline: {rmse:.4f} < {0.92 * baseline_rmse:.4f}")
print(f"total runtime: {time.monotonic() - t_start:.1f}s")
