"""SUPERVISED TEXT CLASSIFICATION on the real UCI SMS Spam Collection.

Ingest: 5574 labelled texts (ham/spam) from the UCI SMS Spam Collection,
one sample per line, committed via `_common.ensure_dataset`.

Pipeline patterns showcased: the **two-pass pattern for text** — pass 1
streams the train split once with plain-regex tokenization to accumulate
token frequencies (building a frozen top-4000 vocabulary) and label counts,
then pass 2 re-streams the same (now warm) shards to train a bag-of-words
model against that frozen vocabulary. Class imbalance (~13% spam) is
handled with a weighted cross-entropy loss whose weights come straight out
of the streamed label counts from pass 1 — no separate stats pass needed.
"""

# ruff: noqa: E402
from __future__ import annotations

import hashlib
import inspect
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import _env

_env.setup()

import dload
from _common import assert_improved, ensure_dataset, http_download

SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
N_MESSAGES, N_HAM, N_SPAM = 5574, 4827, 747  # known UCI stats, for meta
VOCAB_SIZE = 4000
HIDDEN = 64
BATCH_SIZE = 64
N_EPOCHS = 3
TOKEN_RE = re.compile(r"\w+")

Sample = tuple[str, dict[str, bytes]]


def download_and_parse():
    """Fetch the UCI SMS Spam Collection zip (cached) and yield one sample
    per line of SMSSpamCollection: tab-separated label<TAB>text."""
    zip_path = http_download(SMS_URL, "sms-spam-collection.zip")
    with zipfile.ZipFile(zip_path) as zf:
        text = zf.read("SMSSpamCollection").decode("utf-8")
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        label, _, msg = line.partition("\t")
        yield (
            f"sms-{i:04d}",
            {"text": msg.encode("utf-8"), "label": label.encode("utf-8")},
        )


def split_bucket(key: str) -> int:
    """Stable 0-9 bucket from a sha256 of the key: deterministic 80/20
    train/eval split with no held-out file list to keep in sync."""
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10


def is_train(sample: Sample) -> bool:
    return split_bucket(sample[0]) < 8


def is_eval(sample: Sample) -> bool:
    return not is_train(sample)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def make_featurize(vocab_index: dict[str, int]):
    def featurize(sample: Sample) -> tuple[np.ndarray, int]:
        _key, fields = sample
        vec = np.zeros(VOCAB_SIZE, dtype=np.float32)
        for tok in tokenize(fields["text"].decode("utf-8")):
            idx = vocab_index.get(tok)
            if idx is not None:
                vec[idx] = 1.0  # multi-hot: presence, not count
        label = 1 if fields["label"] == b"spam" else 0
        return vec, label

    return featurize


def collate(items: list[tuple[np.ndarray, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.stack([v for v, _l in items])
    y = np.array([label for _v, label in items], dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


class BoWClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(VOCAB_SIZE, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 2)
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    torch.manual_seed(0)
    t0 = time.monotonic()

    repo = dload.Repository.open()
    ds = ensure_dataset(
        repo,
        "sms-spam",
        download_and_parse,
        recipe=inspect.getsource(download_and_parse),
        meta={
            "source": SMS_URL,
            "n_messages": N_MESSAGES,
            "n_ham": N_HAM,
            "n_spam": N_SPAM,
            "spam_fraction": round(N_SPAM / N_MESSAGES, 4),
        },
        target_shard_size=1024 * 1024,
    )
    print(f"dataset: {ds!r}")

    # -- pass 1: stream the train split once for token + label counts.
    token_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for _key, fields in ds.samples().filter(is_train):
        token_counts.update(tokenize(fields["text"].decode("utf-8")))
        label_counts[fields["label"].decode("utf-8")] += 1
    n_train = sum(label_counts.values())
    vocab = [tok for tok, _n in token_counts.most_common(VOCAB_SIZE)]
    vocab_index = {tok: i for i, tok in enumerate(vocab)}
    print(
        f"pass 1: {n_train} train msgs, {len(token_counts)} distinct tokens "
        f"-> vocab of {len(vocab)}; labels {dict(label_counts)}"
    )
    spam_fraction = label_counts["spam"] / n_train
    print(f"class balance (train): {spam_fraction:.1%} spam")

    n_ham, n_spam = label_counts["ham"], label_counts["spam"]
    class_weights = torch.tensor(
        [n_train / (2 * n_ham), n_train / (2 * n_spam)], dtype=torch.float32
    )
    print(f"class weights (ham, spam): {class_weights.tolist()}")

    # -- pass 2: re-stream the same (now warm) shards, frozen vocab, to train.
    featurize = make_featurize(vocab_index)
    train_pipe = (
        ds.samples().filter(is_train).shuffle(seed=0).map(featurize)
        .batch(BATCH_SIZE, collate=collate)
    )  # fmt: skip
    eval_pipe = (
        ds.samples().filter(is_eval).map(featurize).batch(BATCH_SIZE, collate=collate)
    )

    model = BoWClassifier()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: BoWClassifier, {n_params} params")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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

    # -- eval: overall accuracy + spam recall (what matters under imbalance).
    model.eval()
    correct = total = spam_correct = spam_total = ham_total = 0
    with torch.no_grad():
        for x, y in eval_pipe:
            preds = model(x).argmax(dim=1)
            correct += int((preds == y).sum())
            total += y.shape[0]
            spam_mask = y == 1
            spam_total += int(spam_mask.sum())
            spam_correct += int((preds[spam_mask] == 1).sum())
            ham_total += int((y == 0).sum())

    accuracy = correct / total
    spam_recall = spam_correct / spam_total
    baseline = ham_total / total
    print(f"eval ({total} msgs): majority-class baseline {baseline:.3f}")
    print(f"eval accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"spam recall: {spam_recall:.3f} ({spam_correct}/{spam_total})")
    print(f"total runtime: {time.monotonic() - t0:.1f}s")

    assert_improved(losses[0], losses[-1])
    assert accuracy > 0.93, f"accuracy too low: {accuracy:.3f}"
    assert spam_recall > 0.7, f"spam recall too low: {spam_recall:.3f}"
    print(
        "PASS: two-pass vocab-then-train over a streamed dataset, with "
        "class-weighted loss from streamed label counts, beats both the "
        "majority-class baseline and the imbalance."
    )


if __name__ == "__main__":
    main()
