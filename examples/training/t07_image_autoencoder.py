"""Unsupervised representation learning: a convolutional autoencoder on
FashionMNIST that never sees a label during training.

Reuses the "fashion-mnist" dataset committed by t02_image_classification.py
— same raw uint8 "img"/"label"/"split" bytes, streamed with a different
pipeline (no label decoding at all) for a completely different task. This
is the "one dataset, many tasks" pattern: a supervised classifier and an
unsupervised autoencoder both stream the identical shards from the remote,
no re-ingestion, just `repo.dataset("fashion-mnist")`.

The encoder's 32-dim bottleneck is evaluated label-free-at-training-time by
a leave-one-out 1-nearest-neighbor probe: labels are only used at *eval*
to check whether nearby embeddings share a class, never to train the model.
"""

# ruff: noqa: E402
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import _env

_env.setup()

from _common import assert_improved

import dload

BATCH_SIZE = 128
N_EPOCHS = 2
TRAIN_SUBSAMPLE = 10_000
EVAL_SUBSAMPLE = 1_000
BOTTLENECK = 32


def decode_img(sample: tuple[str, dict[str, bytes]]) -> torch.Tensor:
    _key, fields = sample
    img = (
        np.frombuffer(fields["img"], dtype=np.uint8)
        .reshape(1, 28, 28)
        .astype(np.float32)
        / 255.0
    )
    return torch.from_numpy(img)


def collate_img(items: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(items)


def decode_img_label(sample: tuple[str, dict[str, bytes]]) -> tuple[torch.Tensor, int]:
    return decode_img(sample), sample[1]["label"][0]


def collate_img_label(
    items: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.stack([img for img, _ in items])
    labels = torch.tensor([label for _, label in items], dtype=torch.long)
    return imgs, labels


def is_train(sample: tuple[str, dict[str, bytes]]) -> bool:
    return sample[1]["split"] == b"train"


def is_test(sample: tuple[str, dict[str, bytes]]) -> bool:
    return sample[1]["split"] == b"test"


class ConvAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),  # 28 -> 14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 14 -> 7
        )  # fmt: skip
        self.to_bottleneck = nn.Linear(32 * 7 * 7, BOTTLENECK)
        self.from_bottleneck = nn.Linear(BOTTLENECK, 32 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 7 -> 14
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),  # 14 -> 28
        )  # fmt: skip

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).flatten(1)
        return self.to_bottleneck(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_bottleneck(z).view(-1, 32, 7, 7)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def one_nn_accuracy(features: torch.Tensor, labels: torch.Tensor) -> float:
    """Leave-one-out 1-nearest-neighbor accuracy in `features` space."""
    dists = torch.cdist(features, features)
    dists.fill_diagonal_(float("inf"))
    nearest = labels[dists.argmin(dim=1)]
    return (nearest == labels).float().mean().item()


def to_ascii(img: torch.Tensor) -> str:
    """28x28 -> 14x14 average-pooled ASCII art, ' .:*#' by intensity."""
    chars = " .:*#"
    small = torch.nn.functional.avg_pool2d(img.view(1, 1, 28, 28), 2).view(14, 14)
    return "\n".join(
        "".join(chars[min(int(v.item() * len(chars)), len(chars) - 1)] for v in row)
        for row in small
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    t0 = time.monotonic()

    repo = dload.Repository.open()
    ds = repo.dataset("fashion-mnist")
    print(f"dataset: {ds!r}")

    model = ConvAutoencoder()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: ConvAutoencoder, {n_params} params, bottleneck={BOTTLENECK}")

    train_pipe = (
        ds.samples()
        .filter(is_train)
        .shuffle(seed=0)
        .take(TRAIN_SUBSAMPLE)
        .map(decode_img)
        .batch(BATCH_SIZE, collate=collate_img)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epoch_losses: list[float] = []
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss, n_batches = 0.0, 0
        for imgs in train_pipe:
            optimizer.zero_grad()
            recon, _z = model(imgs)
            loss = loss_fn(recon, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        epoch_losses.append(avg_loss)
        print(f"epoch {epoch}: {n_batches} batches, avg recon MSE {avg_loss:.4f}")

    assert_improved(epoch_losses[0], epoch_losses[-1], name="recon MSE")

    # Label-free-at-training probe: embed 1000 held-out test images, then
    # check whether the *labels* (used only now, at eval) cluster in
    # embedding space via leave-one-out 1-NN.
    eval_pipe = (
        ds.samples()
        .filter(is_test)
        .take(EVAL_SUBSAMPLE)
        .map(decode_img_label)
        .batch(BATCH_SIZE, collate=collate_img_label)
    )

    model.eval()
    all_imgs, all_z, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels in eval_pipe:
            _recon, z = model(imgs)
            all_imgs.append(imgs)
            all_z.append(z)
            all_labels.append(labels)
    imgs = torch.cat(all_imgs)
    embeddings = torch.cat(all_z)
    labels = torch.cat(all_labels)

    raw_pixels = imgs.flatten(1)
    raw_acc = one_nn_accuracy(raw_pixels, labels)
    embed_acc = one_nn_accuracy(embeddings, labels)
    print(
        f"raw-pixel 1-NN accuracy:  {raw_acc:.4f} ({len(labels)} samples, chance 0.1)"
    )
    print(
        f"embedding 1-NN accuracy: {embed_acc:.4f} ({len(labels)} samples, chance 0.1)"
    )

    assert embed_acc > 0.6, f"embedding 1-NN accuracy too low: {embed_acc:.4f}"
    print(f"embedding 1-NN accuracy {embed_acc:.4f} > 0.6")

    with torch.no_grad():
        recon0, _ = model(imgs[:1])
    print("before (original):")
    print(to_ascii(imgs[0]))
    print("after (reconstruction):")
    print(to_ascii(recon0[0]))

    print(f"total runtime: {time.monotonic() - t0:.1f}s")
