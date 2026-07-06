"""TIME-SERIES FORECASTING on real data: Daily Minimum Temperatures
(Melbourne, Australia, 1981-1990, ~3650 daily values).

Ingest: the series is committed as ~120 MONTHLY segment samples, keyed
`f"{yyyy}-{mm}"` — segmenting a continuous series into keyed samples so
subranges stream cheaply (a decade fits; minute-resolution sensor data
wouldn't, and this is how you'd shard it).

A time series must not be split randomly: train/eval is a TEMPORAL split
via `.filter` on the (zero-padded, string-sortable) month key. Downstream of
that filter the month segments must stay in *sequential* order for window
construction, so no shuffle sits above the source: `.flat_map` flattens the
monthly segments into a single ordered stream of daily values (a rolling
buffer across month boundaries falls out of `.window`'s own carry-over
buffer), and `.window(size, stride)` yields the (30-day input, 7-day
target) pairs at stride 3. Only then does `.shuffle` randomize the
(already-materialized, so windowing is never repeated) windows, seeded so
each re-iteration of the pipeline reshuffles deterministically per epoch.
"""

# ruff: noqa: E402
from __future__ import annotations

import csv
import inspect
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _env

_env.setup()

import dload
from _common import assert_improved, ensure_dataset, http_download

CSV_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
INPUT_LEN, TARGET_LEN, STRIDE = 30, 7, 3
BUFFER_SIZE, BATCH_SIZE, N_EPOCHS, TRAIN_SEED = 500, 32, 4, 0
_NUM_RE = re.compile(r"-?\d+\.?\d*")

Sample = tuple[str, dict[str, bytes]]
Window = tuple[np.ndarray, np.ndarray]


def download_and_parse() -> list[tuple[str, float]]:
    """Download the CSV (cached) and parse (date, temp) rows, tolerant of
    stray junk on the temperature field (e.g. a literal `"?0.2"`) via a
    numeric-substring regex instead of a strict `float()` cast."""
    path = http_download(CSV_URL, "daily-min-temperatures.csv")
    records: list[tuple[str, float]] = []
    with path.open(newline="") as f:
        rows = csv.reader(f)
        next(rows)  # header
        for row in rows:
            if not row:
                continue
            match = _NUM_RE.search(row[1])
            if match is None:
                continue
            records.append((row[0].strip().strip("\"'"), float(match.group())))
    return records


def samples_from(records: list[tuple[str, float]]) -> Iterator[Sample]:
    """Group daily records into monthly segments, chronological order."""
    months: dict[str, list[tuple[str, float]]] = {}
    for date, temp in records:
        months.setdefault(date[:7], []).append((date, temp))
    for key in sorted(months):
        dates = [d for d, _t in months[key]]
        temps = np.array([t for _d, t in months[key]], dtype=np.float32)
        fields = {
            "dates": dload.codecs.json_bytes(dates),
            "temps": dload.codecs.npy_bytes(temps),
        }
        yield key, fields


def is_train(sample: Sample) -> bool:
    return sample[0] < "1989-01"


is_eval = lambda sample: not is_train(sample)  # noqa: E731


def _split_window(days: list[float]) -> Window:
    return (
        np.array(days[:INPUT_LEN], dtype=np.float32),
        np.array(days[INPUT_LEN:], dtype=np.float32),
    )


def windowed(samples: dload.Pipeline) -> dload.Pipeline:
    """Flatten monthly segments into a single ordered stream of daily temps
    (`.flat_map`), then slide a `window_len`-wide, stride-3 window over it
    (`.window`) — the carry-over buffer inside `.window` is exactly the
    rolling buffer a hand-rolled generator would keep across month
    boundaries. `drop_last=True` matches the old generator, which only ever
    emitted full windows and silently dropped the trailing remainder."""
    window_len = INPUT_LEN + TARGET_LEN
    return (
        samples.flat_map(lambda s: dload.codecs.npy_from(s[1]["temps"]).tolist())
        .window(window_len, stride=STRIDE, drop_last=True)
        .map(_split_window)
    )


def batches(
    windows: list[Window], mean: float, std: float
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    for i in range(0, len(windows) - BATCH_SIZE + 1, BATCH_SIZE):
        chunk = windows[i : i + BATCH_SIZE]
        x = (np.stack([w[0] for w in chunk]) - mean) / std
        y = (np.stack([w[1] for w in chunk]) - mean) / std
        yield (
            torch.from_numpy(x[..., None].astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )


class TempGRU(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, TARGET_LEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _out, hidden = self.gru(x)
        return self.head(hidden[-1])


def main() -> None:
    torch.manual_seed(0)
    t0 = time.monotonic()

    repo = dload.Repository.open()
    ds = ensure_dataset(
        repo,
        "daily-min-temps",
        lambda: samples_from(download_and_parse()),
        recipe=inspect.getsource(download_and_parse),
        meta={
            "source": CSV_URL,
            "units": "degrees Celsius",
            "span": "1981-01-01 to 1990-12-31",
        },
    )
    print(f"dataset: {ds!r}")

    # -- pass 1: stream the train split once to accumulate mean/std.
    t1 = time.monotonic()
    n = total = total_sq = 0
    for _key, fields in ds.samples().filter(is_train):
        temps = dload.codecs.npy_from(fields["temps"]).astype(np.float64)
        total += temps.sum()
        total_sq += (temps * temps).sum()
        n += len(temps)
    mean = total / n
    std = float(np.sqrt(max(total_sq / n - mean * mean, 1e-8)))
    print(f"pass 1 (stats over {n} train days): {time.monotonic() - t1:.2f}s -- cold")

    # -- pass 2: re-stream the same (now warm) shards, sequentially, into
    # 30->7 day windows — materialized ONCE, since recomputing them every
    # epoch would re-walk the whole cache for no reason.
    t2 = time.monotonic()
    train_windows = list(windowed(ds.samples().filter(is_train)))
    print(
        f"pass 2 ({len(train_windows)} train windows): "
        f"{(time.monotonic() - t2) * 1000:.0f} ms -- warm cache"
    )
    # A seeded shuffle pipeline over the in-memory list: each re-iteration
    # advances its epoch counter automatically, so this reshuffles
    # deterministically per epoch without manual `TRAIN_SEED + epoch` bookkeeping.
    shuffled_train = dload.from_iterable(train_windows).shuffle(
        BUFFER_SIZE, seed=TRAIN_SEED
    )

    model = TempGRU()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: TempGRU, {n_params} params")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses: list[float] = []
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss, n_batches = 0.0, 0
        epoch_windows = list(shuffled_train)
        for x, y in batches(epoch_windows, mean, std):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        epoch_loss = total_loss / n_batches
        losses.append(epoch_loss)
        print(f"epoch {epoch}: loss {epoch_loss:.4f} ({n_batches} batches)")

    assert_improved(losses[0], losses[-1])

    # -- eval on the temporal holdout: model vs. two naive baselines, RMSE
    # in degrees C (denormalized).
    model.eval()
    eval_windows = list(windowed(ds.samples().filter(is_eval)))
    se_model = se_persistence = se_mean30 = n_eval = 0.0
    with torch.no_grad():
        for i in range(0, len(eval_windows), BATCH_SIZE):
            chunk = eval_windows[i : i + BATCH_SIZE]
            x = np.stack([w[0] for w in chunk])
            y = np.stack([w[1] for w in chunk])
            x_norm = torch.from_numpy(((x - mean) / std)[..., None].astype(np.float32))
            pred = model(x_norm).numpy() * std + mean
            se_model += float(((pred - y) ** 2).sum())
            persistence = np.repeat(x[:, -1:], TARGET_LEN, axis=1)
            se_persistence += float(((persistence - y) ** 2).sum())
            mean30 = np.repeat(x.mean(axis=1, keepdims=True), TARGET_LEN, axis=1)
            se_mean30 += float(((mean30 - y) ** 2).sum())
            n_eval += y.size

    rmse_model = (se_model / n_eval) ** 0.5
    rmse_persistence = (se_persistence / n_eval) ** 0.5
    rmse_mean30 = (se_mean30 / n_eval) ** 0.5
    print(f"eval ({len(eval_windows)} windows, {int(n_eval)} day-predictions):")
    print(f"  model RMSE:            {rmse_model:.3f} C")
    print(f"  persistence baseline:  {rmse_persistence:.3f} C")
    print(f"  mean-last-30 baseline: {rmse_mean30:.3f} C")

    assert rmse_model < rmse_persistence, (
        f"model RMSE {rmse_model:.3f} not < persistence {rmse_persistence:.3f}"
    )
    print("model beats the persistence baseline on the temporal holdout")
    print(f"total runtime: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
