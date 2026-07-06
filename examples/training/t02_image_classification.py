"""Supervised image classification on real FashionMNIST.

Ingests the four official IDX gzip files, parses them with plain numpy (no
torchvision), and commits ONE dataset whose samples carry raw uint8 arrays
as opaque binary fields — the "binary field encoding of arrays" pattern.
Training and eval pipelines are then split out of the *same* dataset purely
by filtering on a metadata field (`split`), showcasing split-by-metadata
filtering instead of separate train/test datasets.
"""

# ruff: noqa: E402
from __future__ import annotations

import gzip
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import _env

_env.setup()

from _common import assert_improved, ensure_dataset, http_download

import dload

BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]  # fmt: skip
BATCH_SIZE = 128
N_EPOCHS = 2
TRAIN_SUBSAMPLE = 12_000
EVAL_SUBSAMPLE = 2_000


def download_and_parse() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Download the 4 official FashionMNIST IDX gz files (cached) and parse
    them with plain numpy: gunzip, then np.frombuffer past the IDX header
    (16 bytes for the image files, 8 for the label files)."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, img_file, lbl_file in [
        ("train", "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
        ("test", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
    ]:
        img_path = http_download(BASE_URL + img_file, img_file)
        lbl_path = http_download(BASE_URL + lbl_file, lbl_file)
        images = np.frombuffer(
            gzip.open(img_path, "rb").read(), dtype=np.uint8, offset=16
        )
        labels = np.frombuffer(
            gzip.open(lbl_path, "rb").read(), dtype=np.uint8, offset=8
        )
        out[split] = (images.reshape(-1, 28, 28), labels)
    return out


def samples_from(data: dict[str, tuple[np.ndarray, np.ndarray]]):
    """Lazily yield (key, fields) samples — one image's bytes materialize at
    a time, never the whole 55 MB corpus as a list of dicts."""
    for split, (images, labels) in data.items():
        for i in range(len(labels)):
            yield (
                f"{split}-{i:05d}",
                {
                    "img": images[i].tobytes(),
                    "label": bytes([int(labels[i])]),
                    "split": split.encode(),
                },
            )


def is_train(sample: tuple[str, dict[str, bytes]]) -> bool:
    return sample[1]["split"] == b"train"


def is_test(sample: tuple[str, dict[str, bytes]]) -> bool:
    return sample[1]["split"] == b"test"


def decode(sample: tuple[str, dict[str, bytes]]) -> tuple[torch.Tensor, int]:
    _key, fields = sample
    img = (
        np.frombuffer(fields["img"], dtype=np.uint8)
        .reshape(1, 28, 28)
        .astype(np.float32)
        / 255.0
    )
    return torch.from_numpy(img), fields["label"][0]


def collate(items: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.stack([img for img, _ in items])
    labels = torch.tensor([label for _, label in items], dtype=torch.long)
    return imgs, labels


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )  # fmt: skip
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    t0 = time.monotonic()

    repo = dload.Repository.open()
    ds = ensure_dataset(
        repo,
        "fashion-mnist",
        lambda: samples_from(download_and_parse()),
        recipe=inspect.getsource(download_and_parse),
        meta={"source": "zalandoresearch/fashion-mnist", "classes": CLASSES},
        target_shard_size=4 * 1024 * 1024,
    )
    print(f"dataset: {ds!r}")

    n_params = sum(p.numel() for p in SmallCNN().parameters())
    print(f"model: SmallCNN, {n_params} params")

    train_pipe = (
        ds.samples()
        .filter(is_train)
        .shuffle(seed=0)
        .take(TRAIN_SUBSAMPLE)
        .map(decode)
        .batch(BATCH_SIZE, collate=collate)
    )
    eval_pipe = (
        ds.samples()
        .filter(is_test)
        .take(EVAL_SUBSAMPLE)
        .map(decode)
        .batch(BATCH_SIZE, collate=collate)
    )

    model = SmallCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epoch_losses: list[float] = []
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss, n_batches = 0.0, 0
        for imgs, labels in train_pipe:
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        epoch_losses.append(avg_loss)
        print(f"epoch {epoch}: {n_batches} batches, avg loss {avg_loss:.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in eval_pipe:
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    accuracy = correct / total
    print(f"eval accuracy: {accuracy:.4f} ({correct}/{total})")

    assert accuracy > 0.7, f"accuracy too low: {accuracy:.4f}"
    print(f"accuracy {accuracy:.4f} > 0.7 (chance 0.1)")
    assert_improved(epoch_losses[0], epoch_losses[-1])

    print(f"total runtime: {time.monotonic() - t0:.1f}s")
