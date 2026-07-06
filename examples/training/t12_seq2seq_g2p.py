"""SEQUENCE-TO-SEQUENCE grapheme-to-phoneme conversion on the real CMU
Pronouncing Dictionary.

Ingest: 40000 (word, phoneme-sequence) pairs are the textbook case for
**columnar batching of tiny records** (same trick as t10's ratings): one
sample per word would be 40000 tiny pack entries, so instead we pack 500
words per sample as two parallel JSON arrays (`words`, `phones`), 80 samples
total.

Pipeline pattern: `.batch`/`.shuffle` only ever move whole *samples*, so a
90/10 split by hashing the *sample key* only guarantees the ratio at
500-word granularity, not word granularity -- coarser than masking each
columnar batch (t10's approach) but with 80 samples total the rounding
error is at most 500/40000 = 1.25%, which is fine here and much simpler.
Within the train split, columnar batches are flattened word-by-word with
`.flat_map(flatten_batch)`, looped forever with `.repeat(None)`, and
re-mixed with a second `.shuffle()` at word granularity -- dload's own
shard-order shuffle only ever moves whole samples -- before
`.batch(..., collate=collate)` hands the model ready-made tensors.

Model pattern: variable-length seq2seq needs its own collate -- characters
and phonemes are padded to the batch max separately, the encoder ignores
padding via `pack_padded_sequence`, and the decoder is trained with teacher
forcing (BOS + phones as input, phones + EOS as target) under a
padding-aware cross-entropy loss.
"""

# ruff: noqa: E402
from __future__ import annotations

import hashlib
import inspect
import random
import re
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

CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
MAX_ENTRIES = 40_000
WORDS_PER_BATCH = 500
BATCH_SIZE = 64
N_STEPS = 2500
HIDDEN = 128
EMBED_DIM = 64
MAX_DECODE_LEN = 20

# Standard 39-symbol ARPABET inventory used by CMUdict once stress digits are
# stripped (AH0 -> AH); known and deterministic, so it's hardcoded rather
# than recomputed on every run.
PHONEMES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER",
    "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]  # fmt: skip

CHARS = "abcdefghijklmnopqrstuvwxyz"
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 is PAD
CHAR_VOCAB = len(CHARS) + 1

PAD, BOS, EOS = 0, 1, 2
PHON_TO_IDX = {p: i + 3 for i, p in enumerate(PHONEMES)}
IDX_TO_PHON = {i: p for p, i in PHON_TO_IDX.items()}
PHON_VOCAB = len(PHONEMES) + 3

Entry = tuple[str, list[str]]
Sample = tuple[str, dict[str, bytes]]
WORD_RE = re.compile(r"[a-z]+")


def download_and_parse_cmudict() -> list[Entry]:
    """Fetch cmudict.dict (cached) and parse "word PH ON EMES [# comment]"
    lines into (word, phones) pairs: variant entries like "word(2)" are
    skipped, words are restricted to pure a-z letters of length 3-12, stress
    digits are stripped from phonemes (AH0 -> AH), capped at MAX_ENTRIES."""
    path = http_download(CMUDICT_URL, "cmudict.dict")
    entries: list[Entry] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        word, *phones_raw = line.split()
        if "(" in word or not WORD_RE.fullmatch(word) or not (3 <= len(word) <= 12):
            continue
        entries.append((word, [re.sub(r"\d", "", p) for p in phones_raw]))
        if len(entries) >= MAX_ENTRIES:
            break
    return entries


def batch_entries(entries: list[Entry]) -> Iterator[Sample]:
    for i in range(0, len(entries), WORDS_PER_BATCH):
        chunk = entries[i : i + WORDS_PER_BATCH]
        yield (
            f"words-{i // WORDS_PER_BATCH:03d}",
            {
                "words": dload.codecs.json_bytes([w for w, _p in chunk]),
                "phones": dload.codecs.json_bytes([p for _w, p in chunk]),
            },
        )


def split_bucket(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10


def is_train(sample: Sample) -> bool:
    return split_bucket(sample[0]) < 9  # ~90% of the 80 batches


def is_eval(sample: Sample) -> bool:
    return not is_train(sample)


def flatten_batch(sample: Sample) -> Iterator[Entry]:
    """flat_map fn: one columnar batch -> its (word, phones) entries."""
    _key, fields = sample
    words = dload.codecs.json_from(fields["words"])
    phones = dload.codecs.json_from(fields["phones"])
    return zip(words, phones)


def encode_words(words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [[CHAR_TO_IDX[c] for c in w] for w in words]
    lens = torch.tensor([len(i) for i in ids], dtype=torch.long)
    out = torch.zeros(len(words), int(lens.max()), dtype=torch.long)
    for i, seq in enumerate(ids):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out, lens


def collate(
    batch: list[Entry],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chars, lens = encode_words([w for w, _p in batch])
    phon_ids = [[PHON_TO_IDX[p] for p in phones] for _w, phones in batch]
    dec_in = [[BOS] + ids for ids in phon_ids]
    dec_tgt = [ids + [EOS] for ids in phon_ids]
    max_dec = max(len(d) for d in dec_in)
    in_pad = torch.zeros(len(batch), max_dec, dtype=torch.long)
    tgt_pad = torch.zeros(len(batch), max_dec, dtype=torch.long)  # 0 = PAD
    for i, (di, dt) in enumerate(zip(dec_in, dec_tgt)):
        in_pad[i, : len(di)] = torch.tensor(di, dtype=torch.long)
        tgt_pad[i, : len(dt)] = torch.tensor(dt, dtype=torch.long)
    return chars, lens, in_pad, tgt_pad


class G2P(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.char_embed = nn.Embedding(CHAR_VOCAB, EMBED_DIM, padding_idx=PAD)
        self.encoder = nn.GRU(EMBED_DIM, HIDDEN, batch_first=True)
        self.phon_embed = nn.Embedding(PHON_VOCAB, EMBED_DIM, padding_idx=PAD)
        self.decoder = nn.GRU(EMBED_DIM, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, PHON_VOCAB)

    def encode(self, chars: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            self.char_embed(chars), lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _out, h = self.encoder(packed)
        return h

    def forward(
        self, chars: torch.Tensor, lens: torch.Tensor, dec_in: torch.Tensor
    ) -> torch.Tensor:
        h = self.encode(chars, lens)
        out, _h = self.decoder(self.phon_embed(dec_in), h)
        return self.head(out)

    def greedy_decode(self, words: list[str]) -> list[list[str]]:
        """Batched greedy decode: one encoder pass, then step the decoder
        for every word in lockstep until each hits EOS or MAX_DECODE_LEN."""
        chars, lens = encode_words(words)
        with torch.no_grad():
            h = self.encode(chars, lens)
            inp = torch.full((len(words), 1), BOS, dtype=torch.long)
            done = torch.zeros(len(words), dtype=torch.bool)
            out_ids: list[list[int]] = [[] for _ in words]
            for _ in range(MAX_DECODE_LEN):
                out, h = self.decoder(self.phon_embed(inp), h)
                nxt = self.head(out[:, -1]).argmax(dim=-1)
                for i, n in enumerate(nxt.tolist()):
                    if not done[i]:
                        if n == EOS:
                            done[i] = True
                        else:
                            out_ids[i].append(n)
                if bool(done.all()):
                    break
                inp = nxt.unsqueeze(1)
        return [[IDX_TO_PHON[i] for i in ids] for ids in out_ids]


def levenshtein(a: list[str], b: list[str]) -> int:
    """Edit distance between two phoneme sequences (classic DP, one row)."""
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        row = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            row[j] = min(row[j - 1] + 1, prev_row[j] + 1, prev_row[j - 1] + (ca != cb))
        prev_row = row
    return prev_row[-1]


torch.manual_seed(0)
t0 = time.monotonic()

repo = dload.Repository.open()
ds = ensure_dataset(
    repo,
    "cmudict-g2p",
    lambda: batch_entries(download_and_parse_cmudict()),
    recipe=inspect.getsource(download_and_parse_cmudict),
    meta={
        "source": CMUDICT_URL,
        "license": "CMUdict is BSD-licensed (Carnegie Mellon University)",
        "entry_count": MAX_ENTRIES,
        "phonemes": PHONEMES,
    },
)
print(f"dataset: {ds!r}")

item_stream = (
    ds.samples()
    .filter(is_train)
    .shuffle(seed=0)
    .repeat(None)
    .flat_map(flatten_batch)
    .shuffle(2000, seed=1)
    .batch(BATCH_SIZE, collate=collate)
)

eval_entries = list(ds.samples().filter(is_eval).flat_map(flatten_batch))
print(
    f"train batches (~90%) / eval words: {sum(1 for _ in ds.samples().filter(is_train))} batches, {len(eval_entries)} eval words"
)

model = G2P()
n_params = sum(p.numel() for p in model.parameters())
print(f"model: G2P encoder-decoder GRU, {n_params} params")
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

losses: list[float] = []
for step, batch in zip(range(N_STEPS), item_stream):
    chars, lens, dec_in, dec_tgt = batch
    logits = model(chars, lens, dec_in)
    loss = loss_fn(logits.reshape(-1, PHON_VOCAB), dec_tgt.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if (step + 1) % 500 == 0:
        recent = losses[-500:]
        print(f"step {step + 1}/{N_STEPS}: loss {sum(recent) / len(recent):.4f}")
print(f"training took {time.monotonic() - t0:.1f}s")

first_avg = sum(losses[:500]) / 500
last_avg = sum(losses[-500:]) / 500
assert_improved(first_avg, last_avg, factor=0.7, name="G2P loss")

# -- eval: greedy decode every held-out word, batched for speed.
model.eval()
exact = 0
edit_total = ref_total = 0
EVAL_CHUNK = 256
for i in range(0, len(eval_entries), EVAL_CHUNK):
    chunk = eval_entries[i : i + EVAL_CHUNK]
    preds = model.greedy_decode([w for w, _p in chunk])
    for (_w, ref), pred in zip(chunk, preds):
        exact += pred == ref
        edit_total += levenshtein(pred, ref)
        ref_total += len(ref)
model.train()

exact_acc = exact / len(eval_entries)
per = edit_total / ref_total
print(f"eval ({len(eval_entries)} words): exact-match {exact_acc:.3f}, PER {per:.3f}")
print(f"total runtime: {time.monotonic() - t0:.1f}s")

assert per < 0.45, f"PER too high: {per:.3f}"
assert exact_acc > 0.05, f"exact-match too low: {exact_acc:.3f}"

print()
print("--- fun proof: greedy G2P vs reference on 8 held-out words ---")
for word, ref in random.Random(42).sample(eval_entries, 8):
    (pred,) = model.greedy_decode([word])
    mark = "OK " if pred == ref else "   "
    print(f"{mark}{word:14s} pred: {' '.join(pred):30s} ref: {' '.join(ref)}")

print(
    "PASS: character encoder -> GRU decoder learns grapheme-to-phoneme "
    "from a streamed, columnar-batched CMUdict."
)
