"""MODEL-IN-THE-PIPELINE: a frozen neural encoder running as a dload `.map`
transform, streaming *embeddings* instead of raw pixels to everything below
it — the pattern for pipelining data through any other model (a feature
extractor, a teacher network, a text embedder) instead of your own hand
decoder.

Act 1 quick-pretrains a small conv autoencoder (copied from
t07_image_autoencoder.py) for a couple of epochs and freezes its encoder —
standing in for "some big pretrained model you already have lying around".
Act 2
wires that frozen encoder INTO a feature pipeline:
`.batch(256, collate=img_collate).map(encode_batch)` — batching *before*
the model-map amortizes inference into one forward pass per 256 images,
exactly how you'd wrap a GPU feature extractor or teacher model; every
consumer downstream only ever sees 32-dim embeddings, never a pixel. Act 3
trains a cheap linear head on those streamed embeddings and compares it
against the same head trained directly on raw pixels, to see how much
signal survives a 24x compression down to the bottleneck.
"""

# ruff: noqa: E402
from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import _env

_env.setup()

from _common import assert_improved

import dload

AE_BATCH_SIZE = 128
AE_TRAIN_SUBSAMPLE = 8_000
AE_EPOCHS = 2
BOTTLENECK = 32

FEATURE_BATCH_SIZE = 256
FEAT_TRAIN_SUBSAMPLE = 4_000
FEAT_TEST_SUBSAMPLE = 1_500
HEAD_EPOCHS = 3
HEAD_LR = 3e-2

Sample = tuple[str, dict[str, bytes]]


def is_train(sample: Sample) -> bool:
    return sample[1]["split"] == b"train"


def is_test(sample: Sample) -> bool:
    return sample[1]["split"] == b"test"


def decode_img(sample: Sample) -> torch.Tensor:
    img = (
        np.frombuffer(sample[1]["img"], dtype=np.uint8)
        .reshape(1, 28, 28)
        .astype(np.float32)
        / 255.0
    )
    return torch.from_numpy(img)


def collate_img(items: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(items)


def img_collate(items: list[Sample]) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode + stack a raw-sample batch into (imgs, labels) — this is the
    collate that feeds the model-map stage, so it stops at pixels."""
    imgs = torch.stack([decode_img(item) for item in items])
    labels = torch.tensor([item[1]["label"][0] for item in items], dtype=torch.long)
    return imgs, labels


def raw_pixel_collate(items: list[Sample]) -> tuple[torch.Tensor, torch.Tensor]:
    imgs, labels = img_collate(items)
    return imgs.flatten(1), labels


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


def make_encode_batch(
    frozen_encoder: ConvAutoencoder,
) -> Callable[[tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]:
    """The model-in-pipeline stage: closes over the frozen encoder and turns
    a (imgs, labels) batch into a (features, labels) batch, one forward pass
    per batch, no gradients."""

    def encode_batch(
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        imgs, labels = batch
        with torch.no_grad():
            features = frozen_encoder.encode(imgs)
        return features, labels

    return encode_batch


def train_linear_head(
    pipe: dload.Pipeline, in_dim: int, n_epochs: int
) -> tuple[nn.Linear, list[float]]:
    head = nn.Linear(in_dim, 10)
    optimizer = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    loss_fn = nn.CrossEntropyLoss()
    epoch_losses: list[float] = []
    for _epoch in range(n_epochs):
        total_loss, n_batches = 0.0, 0
        for features, labels in pipe:
            optimizer.zero_grad()
            loss = loss_fn(head(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        epoch_losses.append(total_loss / n_batches)
    return head, epoch_losses


def eval_linear_head(head: nn.Linear, pipe: dload.Pipeline) -> float:
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in pipe:
            preds = head(features).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total


if __name__ == "__main__":
    torch.manual_seed(0)
    t0 = time.monotonic()

    repo = dload.Repository.open()
    ds = repo.dataset("fashion-mnist")
    print(f"dataset: {ds!r}")

    # --- Act 1: quick-pretrain the "big model you already have" -----------
    print(f"\n=== act 1: pretraining conv autoencoder (bottleneck={BOTTLENECK}) ===")
    autoencoder = ConvAutoencoder()
    ae_train_pipe = (
        ds.samples()
        .filter(is_train)
        .shuffle(seed=0)
        .take(AE_TRAIN_SUBSAMPLE)
        .map(decode_img)
        .batch(AE_BATCH_SIZE, collate=collate_img)
    )
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    ae_losses: list[float] = []
    for epoch in range(AE_EPOCHS):
        autoencoder.train()
        total_loss, n_batches = 0.0, 0
        for imgs in ae_train_pipe:
            optimizer.zero_grad()
            recon, _z = autoencoder(imgs)
            loss = loss_fn(recon, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        ae_losses.append(avg_loss)
        print(f"  epoch {epoch}: {n_batches} batches, avg recon MSE {avg_loss:.4f}")

    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad_(False)
    print("encoder frozen (eval mode, requires_grad=False) — now just a function")

    # --- Act 2: the frozen encoder becomes a pipeline stage ----------------
    print("\n=== act 2: encoder as a .map stage — pipeline yields embeddings ===")
    encode_batch = make_encode_batch(autoencoder)
    train_feat_pipe = (
        ds.samples()
        .filter(is_train)
        .shuffle(seed=0)
        .take(FEAT_TRAIN_SUBSAMPLE)
        .batch(FEATURE_BATCH_SIZE, collate=img_collate)
        .map(encode_batch)
    )
    test_feat_pipe = (
        ds.samples()
        .filter(is_test)
        .take(FEAT_TEST_SUBSAMPLE)
        .batch(FEATURE_BATCH_SIZE, collate=img_collate)
        .map(encode_batch)
    )
    sample_features, sample_labels = next(iter(train_feat_pipe))
    print(
        f"  train feature pipeline yields batches of {tuple(sample_features.shape)} "
        f"features + {tuple(sample_labels.shape)} labels — no pixels downstream"
    )

    # --- Act 3: cheap downstream head on streamed embeddings ---------------
    print("\n=== act 3: linear head (32 -> 10) on streamed embeddings ===")
    embed_head, embed_losses = train_linear_head(
        train_feat_pipe, in_dim=BOTTLENECK, n_epochs=HEAD_EPOCHS
    )
    for epoch, loss in enumerate(embed_losses):
        print(f"  epoch {epoch}: avg CE loss {loss:.4f}")
    embed_acc = eval_linear_head(embed_head, test_feat_pipe)
    assert_improved(embed_losses[0], embed_losses[-1], name="embedding-head loss")

    print("\n--- baseline: linear head (784 -> 10) directly on raw pixels ---")
    raw_train_pipe = (
        ds.samples()
        .filter(is_train)
        .shuffle(seed=0)
        .take(FEAT_TRAIN_SUBSAMPLE)
        .batch(FEATURE_BATCH_SIZE, collate=raw_pixel_collate)
    )
    raw_test_pipe = (
        ds.samples()
        .filter(is_test)
        .take(FEAT_TEST_SUBSAMPLE)
        .batch(FEATURE_BATCH_SIZE, collate=raw_pixel_collate)
    )
    raw_head, raw_losses = train_linear_head(
        raw_train_pipe, in_dim=28 * 28, n_epochs=HEAD_EPOCHS
    )
    raw_acc = eval_linear_head(raw_head, raw_test_pipe)

    print(
        f"\nembedding head ({BOTTLENECK}-d, {FEAT_TRAIN_SUBSAMPLE} samples): "
        f"accuracy {embed_acc:.4f} (chance 0.1)"
    )
    print(
        f"raw-pixel head (784-d, {FEAT_TRAIN_SUBSAMPLE} samples):    "
        f"accuracy {raw_acc:.4f} (chance 0.1)"
    )
    print(
        f"a {BOTTLENECK}-d representation ({784 // BOTTLENECK}x smaller than raw pixels) "
        f"carries most of the classification signal — the point isn't beating the "
        f"raw-pixel head, it's that {embed_acc:.4f} vs {raw_acc:.4f} came from a "
        f"representation 24x more compact."
    )

    assert embed_acc > 0.6, f"embedding head accuracy too low: {embed_acc:.4f}"
    print(f"embedding head accuracy {embed_acc:.4f} > 0.6")

    print(f"\ntotal runtime: {time.monotonic() - t0:.1f}s")
