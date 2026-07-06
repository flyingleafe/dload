"""COLLABORATIVE FILTERING on real MovieLens 100k ratings.

Ingest: 100k (user, item, rating, timestamp) rows are the textbook case for
**columnar batching** — one sample per rating would mean 100k tiny pack
entries, with per-sample framing overhead dwarfing the ~16 bytes of actual
payload. Instead we pack 2000 ratings per sample as parallel `.npy` arrays
(`users`/`items`/`ratings`/`timestamps`, via `dload.codecs.npy_bytes`), 50
samples total, plus one extra sample carrying the item id -> title lookup
as JSON (`dload.codecs.json_bytes`) for the human-readable demo at the end.

Patterns: recommenders must not leak future ratings into the past, so the
split is **temporal**, not a hash of the key. A first pass streams every
rating batch to find the 80th-percentile timestamp; a `.map` on the real
training/eval pipelines then masks each columnar batch to `ts < q80`
(train) or `ts >= q80` (eval), skipping the "item-titles" sample via
`.filter`. Sample-batch order is shuffled by `.shuffle(seed=0)`; row order
*within* a batch is shuffled separately with a seeded numpy rng (dload only
shuffles whole samples). Minibatches of 1024 ratings are then assembled
from the stream of masked columnar batches in plain python, since dload's
`.batch` groups whole samples, not the rows packed inside one.
"""

# ruff: noqa: E402
from __future__ import annotations

import hashlib
import inspect
import sys
import time
import zipfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import _env

_env.setup()

import dload
from _common import assert_improved, ensure_dataset, http_download

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
RATINGS_PER_SAMPLE = 2000
N_USERS = 944  # user ids run 1..943; index 0 unused
N_ITEMS = 1683  # item ids run 1..1682; index 0 unused
EMB_DIM = 32
MINIBATCH_SIZE = 1024
N_EPOCHS = 5
LR = 5e-3
WEIGHT_DECAY = 1e-5

Sample = tuple[str, dict[str, bytes]]


def download_and_parse_movielens() -> Iterator[Sample]:
    """Fetch the MovieLens 100k zip and yield 50 columnar samples of 2000
    ratings each (`users`/`items`/`ratings`/`timestamps` as `.npy` arrays),
    plus one "item-titles" sample with the id -> title lookup as JSON. Note:
    MovieLens data is distributed by GroupLens for research use only."""
    zip_path = http_download(ML100K_URL, "ml-100k.zip")
    with zipfile.ZipFile(zip_path) as zf:
        data_text = zf.read("ml-100k/u.data").decode("utf-8")
        item_text = zf.read("ml-100k/u.item").decode("latin-1")

    titles = [
        [int(line.split("|")[0]), line.split("|")[1]]
        for line in item_text.splitlines()
        if line.strip()
    ]
    yield "item-titles", {"titles": dload.codecs.json_bytes(titles)}

    users, items, ratings, timestamps = [], [], [], []
    batch_idx = 0
    for line in data_text.splitlines():
        if not line.strip():
            continue
        u, i, r, t = line.split("\t")
        users.append(int(u))
        items.append(int(i))
        ratings.append(float(r))
        timestamps.append(int(t))
        if len(users) == RATINGS_PER_SAMPLE:
            yield (
                f"ratings-{batch_idx:03d}",
                {
                    "users": dload.codecs.npy_bytes(np.array(users, dtype=np.int32)),
                    "items": dload.codecs.npy_bytes(np.array(items, dtype=np.int32)),
                    "ratings": dload.codecs.npy_bytes(
                        np.array(ratings, dtype=np.float32)
                    ),
                    "timestamps": dload.codecs.npy_bytes(
                        np.array(timestamps, dtype=np.int32)
                    ),
                },
            )
            batch_idx += 1
            users, items, ratings, timestamps = [], [], [], []
    assert not users, f"{len(users)} leftover ratings don't fill a full batch"


def is_ratings_batch(sample: Sample) -> bool:
    return sample[0] != "item-titles"


def row_shuffle_seed(key: str) -> int:
    """Stable per-batch seed for the in-batch row shuffle, independent of
    dload's own sample-batch shuffle."""
    return int(hashlib.sha256(f"row-shuffle:{key}".encode()).hexdigest(), 16) % (2**32)


RowBatch = tuple[np.ndarray, np.ndarray, np.ndarray]


def make_row_processor(q80: float, *, train: bool) -> Callable[[Sample], RowBatch]:
    def process(sample: Sample) -> RowBatch:
        key, fields = sample
        users = dload.codecs.npy_from(fields["users"])
        items = dload.codecs.npy_from(fields["items"])
        ratings = dload.codecs.npy_from(fields["ratings"]).astype(np.float32)
        timestamps = dload.codecs.npy_from(fields["timestamps"])
        mask = timestamps < q80 if train else timestamps >= q80
        users, items, ratings = users[mask], items[mask], ratings[mask]
        if train:
            perm = np.random.default_rng(row_shuffle_seed(key)).permutation(len(users))
            users, items, ratings = users[perm], items[perm], ratings[perm]
        return users, items, ratings

    return process


def minibatches(pipe: Iterable[RowBatch], batch_size: int) -> Iterator[RowBatch]:
    """Assemble fixed-size minibatches of ratings from a stream of variable-
    size (post-mask) columnar batches -- plain python, no dload `.batch`."""
    buf_u: list[np.ndarray] = []
    buf_i: list[np.ndarray] = []
    buf_r: list[np.ndarray] = []
    n = 0
    for users, items, ratings in pipe:
        buf_u.append(users)
        buf_i.append(items)
        buf_r.append(ratings)
        n += len(users)
        while n >= batch_size:
            u, i, r = (
                np.concatenate(buf_u),
                np.concatenate(buf_i),
                np.concatenate(buf_r),
            )
            yield u[:batch_size], i[:batch_size], r[:batch_size]
            buf_u, buf_i, buf_r = [u[batch_size:]], [i[batch_size:]], [r[batch_size:]]
            n -= batch_size
    if n:
        yield np.concatenate(buf_u), np.concatenate(buf_i), np.concatenate(buf_r)


class MatrixFactorization(nn.Module):
    def __init__(self, global_mean: float) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(N_USERS, EMB_DIM)
        self.item_emb = nn.Embedding(N_ITEMS, EMB_DIM)
        self.user_bias = nn.Embedding(N_USERS, 1)
        self.item_bias = nn.Embedding(N_ITEMS, 1)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        self.register_buffer(
            "global_mean", torch.tensor(global_mean, dtype=torch.float32)
        )

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        pu, qi = self.user_emb(users), self.item_emb(items)
        bu, bi = self.user_bias(users).squeeze(-1), self.item_bias(items).squeeze(-1)
        return self.global_mean + bu + bi + (pu * qi).sum(-1)


torch.manual_seed(0)
t_start = time.monotonic()

repo = dload.Repository.open()
ds = ensure_dataset(
    repo, "movielens-100k", download_and_parse_movielens,
    recipe=inspect.getsource(download_and_parse_movielens),
    meta={
        "source": ML100K_URL,
        "num_ratings": 100_000,
        "num_users": 943,
        "num_items": 1682,
        "num_rating_samples": 50,
        "note": "MovieLens data is distributed by GroupLens for research use only.",
    },
)  # fmt: skip
print(f"dataset: {ds!r}")

# -- pass 1: stream every rating batch once to find the temporal split point
# (80th percentile timestamp) and the train-set global mean rating.
t0 = time.monotonic()
u_chunks, i_chunks, r_chunks, t_chunks = [], [], [], []
for _key, fields in ds.samples().filter(is_ratings_batch):
    u_chunks.append(dload.codecs.npy_from(fields["users"]))
    i_chunks.append(dload.codecs.npy_from(fields["items"]))
    r_chunks.append(dload.codecs.npy_from(fields["ratings"]))
    t_chunks.append(dload.codecs.npy_from(fields["timestamps"]))
all_users = np.concatenate(u_chunks)
all_items = np.concatenate(i_chunks)
all_ratings = np.concatenate(r_chunks)
all_timestamps = np.concatenate(t_chunks)
pass1_s = time.monotonic() - t0

q80 = float(np.quantile(all_timestamps, 0.8))
train_mask = all_timestamps < q80
eval_mask = ~train_mask
global_mean = float(all_ratings[train_mask].mean())
n_train, n_eval = int(train_mask.sum()), int(eval_mask.sum())
print(
    f"pass 1 ({len(all_timestamps)} ratings): {pass1_s:.2f}s -- "
    f"q80 timestamp {int(q80)}, {n_train} train / {n_eval} eval, "
    f"train mean rating {global_mean:.3f}"
)

train_seen: dict[int, set[int]] = {}
for u, i in zip(all_users[train_mask], all_items[train_mask]):
    train_seen.setdefault(int(u), set()).add(int(i))

_titles_key, titles_fields = next(
    iter(ds.samples().filter(lambda s: s[0] == "item-titles"))
)
id_to_title = {
    int(i): title for i, title in dload.codecs.json_from(titles_fields["titles"])
}

# -- pass 2: re-stream the same (now warm) shards, masked to each split.
train_pipe = (
    ds.samples()
    .filter(is_ratings_batch)
    .shuffle(seed=0)
    .map(make_row_processor(q80, train=True))
)
eval_pipe = (
    ds.samples().filter(is_ratings_batch).map(make_row_processor(q80, train=False))
)

model = MatrixFactorization(global_mean)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

epoch_losses: list[float] = []
for epoch in range(N_EPOCHS):
    total, seen = 0.0, 0
    for users, items, ratings in minibatches(train_pipe, MINIBATCH_SIZE):
        u = torch.from_numpy(users.astype(np.int64))
        i = torch.from_numpy(items.astype(np.int64))
        r = torch.from_numpy(ratings)
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(u, i), r)
        loss.backward()
        opt.step()
        total += loss.item() * len(r)
        seen += len(r)
    epoch_losses.append(total / seen)
    print(f"epoch {epoch}: train MSE {epoch_losses[-1]:.4f} ({seen} ratings)")

assert_improved(epoch_losses[0], epoch_losses[-1], name="train MSE")

# -- eval: model RMSE on the temporal holdout vs. the global-train-mean baseline.
model.eval()
se = se_baseline = 0.0
n_eval_seen = 0
with torch.no_grad():
    for users, items, ratings in minibatches(eval_pipe, MINIBATCH_SIZE):
        u = torch.from_numpy(users.astype(np.int64))
        i = torch.from_numpy(items.astype(np.int64))
        r = torch.from_numpy(ratings)
        preds = model(u, i)
        se += ((preds - r) ** 2).sum().item()
        se_baseline += ((r - global_mean) ** 2).sum().item()
        n_eval_seen += len(r)

rmse = (se / n_eval_seen) ** 0.5
baseline_rmse = (se_baseline / n_eval_seen) ** 0.5
print(
    f"eval ({n_eval_seen} ratings): RMSE {rmse:.4f}, baseline (train-mean) RMSE {baseline_rmse:.4f}"
)

assert rmse < 0.95 * baseline_rmse, (
    f"model RMSE {rmse:.4f} not < 0.95x baseline {baseline_rmse:.4f}"
)
print(f"model beats the global-mean baseline: {rmse:.4f} < {0.95 * baseline_rmse:.4f}")

# -- fun proof: top-5 unseen movies for the most active eval user.
eval_users, eval_counts = np.unique(all_users[eval_mask], return_counts=True)
active_user = int(eval_users[np.argmax(eval_counts)])
seen = train_seen.get(active_user, set())
candidates = [i for i in range(1, N_ITEMS) if i not in seen]
with torch.no_grad():
    scores = model(
        torch.full((len(candidates),), active_user, dtype=torch.long),
        torch.tensor(candidates, dtype=torch.long),
    ).tolist()
top5 = sorted(zip(candidates, scores), key=lambda x: -x[1])[:5]
print(f"top-5 predicted unseen movies for user {active_user}:")
for item_id, score in top5:
    print(f"  {id_to_title.get(item_id, '?')!r} (predicted {score:.2f})")

print(f"total runtime: {time.monotonic() - t_start:.1f}s")
