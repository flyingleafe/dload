"""Self-supervised character-level language modeling on Tiny Shakespeare.

Ingest: the ~1.1 MB corpus is chopped into contiguous ~4 KB chunks and
committed as ordinary samples — a continuous text corpus stored as chunked
samples, the trick for any streamed corpus too big to be one sample.

Pipeline pattern: turning one chunk into many (input, target) windows is a
1->N expansion, `.flat_map(make_windows)`. `.shuffle(seed=...)` only mixes
whole chunks, so consecutive windows from one chunk would still land next
to each other in the stream; a second `.shuffle(1000, seed=...)` downstream
of the flat_map fixes that at window granularity. `.repeat(None)` loops the
chunk order forever (re-shuffling shard order each pass), so the whole
epoch machinery collapses into one combinator chain that the training loop
just zips against `range(N_STEPS)`.
"""

# ruff: noqa: E402
from __future__ import annotations

import inspect
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import _env

_env.setup()

import dload
from _common import assert_improved, ensure_dataset, http_download

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CHUNK_CHARS = 4096
CONTEXT_LEN = 64
STRIDE = 32
BATCH_SIZE = 64
N_STEPS = 1500

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n.,;:!?'\"-"
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}
FALLBACK_IDX = len(ALPHABET)  # anything outside the fixed alphabet
VOCAB_SIZE = len(ALPHABET) + 1

Sample = tuple[str, dict[str, bytes]]
Window = tuple[list[int], list[int]]


def download_tiny_shakespeare() -> Path:
    """Fetch the Tiny Shakespeare corpus (cached under .downloads/)."""
    return http_download(SHAKESPEARE_URL, "tiny-shakespeare.txt")


def chunk_samples(text: str) -> Iterator[Sample]:
    for i in range(0, len(text), CHUNK_CHARS):
        chunk = text[i : i + CHUNK_CHARS]
        yield f"chunk-{i // CHUNK_CHARS:05d}", {"text": dload.codecs.text_bytes(chunk)}


def encode(text: str) -> list[int]:
    return [CHAR_TO_IDX.get(c, FALLBACK_IDX) for c in text]


def make_windows(sample: Sample) -> list[Window]:
    """One text chunk -> every (input, target) window it contains."""
    ids = encode(dload.codecs.text_from(sample[1]["text"]))
    return [
        (ids[s : s + CONTEXT_LEN], ids[s + 1 : s + CONTEXT_LEN + 1])
        for s in range(0, len(ids) - CONTEXT_LEN, STRIDE)
    ]


class CharGRU(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, h_out = self.gru(self.embed(x), h)
        return self.head(out), h_out


def generate(model: CharGRU, seed_text: str, n_chars: int, temperature: float) -> str:
    model.eval()
    ids = encode(seed_text)
    with torch.no_grad():
        _logits, h = model(torch.tensor([ids], dtype=torch.long))
    out = list(seed_text)
    last = ids[-1]
    for _ in range(n_chars):
        with torch.no_grad():
            logits, h = model(torch.tensor([[last]], dtype=torch.long), h)
        probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
        last = int(torch.multinomial(probs, 1).item())
        out.append(IDX_TO_CHAR.get(last, "?"))
    model.train()
    return "".join(out)


torch.manual_seed(0)

path = download_tiny_shakespeare()
full_text = path.read_text(encoding="utf-8")

repo = dload.Repository.open()
ds = ensure_dataset(
    repo,
    "tiny-shakespeare",
    lambda: chunk_samples(full_text),
    recipe=inspect.getsource(download_tiny_shakespeare),
    meta={"source": SHAKESPEARE_URL, "total_chars": len(full_text)},
)
print(f"dataset: {ds!r}")

window_stream = (
    ds.samples()
    .shuffle(seed=0)
    .repeat(None)
    .flat_map(make_windows)
    .shuffle(1000, seed=1)
    .batch(BATCH_SIZE)
)

model = CharGRU(VOCAB_SIZE)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
loss_fn = nn.CrossEntropyLoss()

losses: list[float] = []
t0 = time.monotonic()
for step, batch in zip(range(N_STEPS), window_stream):
    xs = torch.tensor([w[0] for w in batch], dtype=torch.long)
    ys = torch.tensor([w[1] for w in batch], dtype=torch.long)
    logits, _h = model(xs)
    loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), ys.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if (step + 1) % 300 == 0:
        recent = losses[-300:]
        print(f"step {step + 1}/{N_STEPS}: loss {sum(recent) / len(recent):.4f}")
print(f"training took {time.monotonic() - t0:.1f}s")

first_avg = sum(losses[:300]) / 300
last_avg = sum(losses[-300:]) / 300
assert_improved(first_avg, last_avg, factor=0.75, name="char LM loss")

print()
print("--- generated sample (seed 'ROMEO:', temperature 0.8) ---")
print(generate(model, "ROMEO:", 300, temperature=0.8))
