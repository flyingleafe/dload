# ruff: noqa: E402
"""LLM-style text ingestion + token-packing benchmark (no model training).

Ingests enwik9 (~323 MB zip -> ~1 GB of Wikipedia XML/text, the classic
enwik8/enwik9 compression-benchmark corpus) once, streaming the zip's single
entry through `zipfile` in 4 MiB chunks so the ~1 GB of decompressed text
never sits in memory whole. Each chunk becomes one raw sample; downstream
stages measure raw R2 throughput, warm-cache throughput, byte-level
tokenization, and the standard GPT-style packing recipe: tokenize -> pack
into fixed-length sequences with a rolling remainder buffer across chunk
boundaries -> shuffle (document order + a sequence-level buffer) -> batch.
Ends with a back-of-envelope H100 feasibility check for decoder-only LM
pretraining.
"""

from __future__ import annotations

import inspect
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import _env

_env.setup()
os.environ["DLOAD_CACHE_DIR"] = str(_env.REPO_ROOT / ".dload-cache-bench")

import dload
from _common import ensure_dataset, http_download  # pyright: ignore[reportMissingImports]
from dload.torch import as_iterable_dataset

QUICK = "--quick" in sys.argv
QUICK_N = 30

# mattmahoney.net (https, then http) first, then the cs.fit.edu mirror; only
# if every enwik9 source is unreachable do we fall back to enwik8 (~100 MB,
# NOT equivalent -- 1/10th the data).
ENWIK9_URLS = [
    "https://mattmahoney.net/dc/enwik9.zip",
    "http://mattmahoney.net/dc/enwik9.zip",
    "http://cs.fit.edu/~mmahoney/compression/enwik9.zip",
]
ENWIK8_FALLBACK_URL = "https://data.deepai.org/enwik8.zip"
DATASET_NAME = "enwik9"
TARGET_SHARD_SIZE = 64 * 1024 * 1024
CHUNK = 4 * 1024 * 1024

SEQ_LEN = 2048
BATCH_SIZE = 8
SHUFFLE_BUF_SEQ = 64


def _pick_and_download() -> tuple[Path, str, bool]:
    """Try enwik9 mirrors in order; only if all fail, fall back to enwik8 and
    say so loudly. Returns (zip_path, source_url, used_enwik8_fallback)."""
    for url in ENWIK9_URLS:
        try:
            return http_download(url, "enwik9.zip"), url, False
        except Exception as exc:
            print(f"  enwik9 source failed ({url}): {exc}")
    print("!" * 78)
    print("! ALL enwik9 MIRRORS UNREACHABLE -- falling back to enwik8 (~100 MB).")
    print("! enwik8 is NOT equivalent to enwik9: ~1/10th the bytes. Every")
    print("! throughput number below is measured on that smaller corpus.")
    print("!" * 78)
    return http_download(ENWIK8_FALLBACK_URL, "enwik8.zip"), ENWIK8_FALLBACK_URL, True


def download_enwik9():
    """Resolve the best available enwik9 mirror (falling back to enwik8 only
    if every enwik9 source is unreachable), then stream-decompress the
    archive's single entry in 4 MiB chunks via zipfile's file-like `.open()`
    -- `.read(CHUNK)` pulls one chunk off the DEFLATE stream at a time, so
    the ~1 GB decompressed text never sits in memory whole. Yields one
    sample per chunk: key f"chunk-{i:04d}", fields {"text": raw bytes}."""
    zip_path, _source_url, _is_fallback = _pick_and_download()
    with zipfile.ZipFile(zip_path) as zf:
        (name,) = zf.namelist()
        with zf.open(name) as f:
            i = 0
            while chunk := f.read(CHUNK):
                yield f"chunk-{i:04d}", {"text": chunk}
                i += 1


def ingest(repo: dload.Repository) -> dload.Dataset:
    try:
        ds = repo.dataset(DATASET_NAME)
        print(f"dataset {DATASET_NAME!r}: {ds!r} (already ingested)")
        return ds
    except dload.NotFoundError:
        pass
    print(f"dataset {DATASET_NAME!r} not in remote yet -- downloading + committing")
    zip_path, source_url, is_fallback = _pick_and_download()
    with zipfile.ZipFile(zip_path) as zf:
        total_uncompressed = zf.getinfo(zf.namelist()[0]).file_size
    t0 = time.monotonic()
    ds = ensure_dataset(
        repo,
        DATASET_NAME,
        download_enwik9,
        recipe=inspect.getsource(download_enwik9),
        meta={
            "source": source_url,
            "total_bytes_uncompressed": total_uncompressed,
            "is_enwik8_fallback": is_fallback,
        },
        target_shard_size=TARGET_SHARD_SIZE,
    )
    dt = time.monotonic() - t0
    mb = ds.manifest.total_bytes / 2**20
    print(f"ingest done: {dt:.1f}s, {mb:.1f} MB, {mb / dt:.2f} MB/s")
    return ds


# tokenize / pack_step / emit_packed below are module-level so the composed
# pipeline (built with `.map`/`.scan`/`.flat_map`) survives pickling into
# spawned DataLoader workers (see dload/torch.py).


def chunk_bytes(sample: tuple[str, dict[str, bytes]]) -> int:
    return len(sample[1]["text"])


def tokenize_chunk(sample: tuple[str, dict[str, bytes]]) -> tuple[str, np.ndarray]:
    """Byte-level tokenization: one token per raw byte (vocab size 256)."""
    key, fields = sample
    return key, np.frombuffer(fields["text"], dtype=np.uint8).astype(np.int64)


def tokenized_raw_bytes(item: tuple[str, np.ndarray]) -> int:
    return int(item[1].size)  # 1 byte-token == 1 original byte


def tokenized_mtokens(item: tuple[str, np.ndarray]) -> float:
    return item[1].size / 1e6


def seq_mtokens(seq: np.ndarray) -> float:
    return seq.size / 1e6


PackState = tuple[np.ndarray, list[np.ndarray]]


def pack_step(state: PackState, sample: tuple[str, np.ndarray]) -> PackState:
    """`.scan()` step for the standard GPT pretraining pattern: concatenate
    the token stream across chunk boundaries and cut fixed `SEQ_LEN` windows,
    carrying a rolling remainder buffer forward so no tokens are dropped or
    duplicated at a chunk edge. A trailing partial remainder (< SEQ_LEN) is
    dropped (never flushed by an `emitted` list). Pair with
    `.flat_map(emit_packed)` to turn each step's emitted sequences back into
    a flat stream."""
    carry, _prev_emitted = state
    _key, tokens = sample
    buf = np.concatenate([carry, tokens]) if carry.size else tokens
    emitted: list[np.ndarray] = []
    while buf.size >= SEQ_LEN:
        # .copy() matters: yielding a view would pin the whole ~32 MB chunk
        # array alive for as long as any downstream buffer holds the
        # sequence — a shuffle buffer of views OOMs a small machine.
        emitted.append(buf[:SEQ_LEN].copy())
        buf = buf[SEQ_LEN:]
    return buf, emitted


def emit_packed(state: PackState) -> list[np.ndarray]:
    return state[1]


PACK_INIT: PackState = (np.empty(0, dtype=np.int64), [])


def build_packed_pipeline(samples: dload.Pipeline) -> dload.Pipeline:
    """Tokenize -> pack, composed straight onto a pipeline of raw chunk
    samples (e.g. `ds.samples()` or `ds.samples().shuffle(...)`). Worker
    sharding flows through automatically -- no `_iterate` override needed."""
    return (
        samples.map(tokenize_chunk)
        .scan(pack_step, init=PACK_INIT)
        .flat_map(emit_packed)
    )


def bench(
    name: str,
    iterable: Any,
    n_items: int,
    bytes_per_item_fn: Any = None,
    extra_fn: Any = None,
    extra_label: str = "",
) -> float:
    """Measure wall time / items/s / approx MB/s over up to `n_items` items;
    returns items/s."""
    it = iter(iterable)
    t0 = time.monotonic()
    n = nbytes = extra_total = 0
    for item in it:
        n += 1
        if bytes_per_item_fn is not None:
            nbytes += bytes_per_item_fn(item)
        if extra_fn is not None:
            extra_total += extra_fn(item)
        if n >= n_items:
            break
    dt = max(time.monotonic() - t0, 1e-9)
    items_per_s = n / dt
    mb_per_s = (nbytes / 2**20) / dt if bytes_per_item_fn is not None else None
    extra_per_s = extra_total / dt if extra_fn is not None else None
    row = f"  {name:<26} {n:>6} it  {dt:>7.2f}s  {items_per_s:>9.1f} it/s"
    if mb_per_s is not None:
        row += f"  {mb_per_s:>9.2f} MB/s"
    if extra_per_s is not None:
        row += f"  {extra_per_s:>9.3f} {extra_label}"
    print(row)
    return items_per_s


def run_benchmarks(repo: dload.Repository, ds: dload.Dataset) -> float:
    n = QUICK_N if QUICK else len(ds)
    # Sections 4-6 yield 2048-token sequences/batches, not chunks: budget them
    # by sequence count (full corpus pass) or they'd measure a sliver of the
    # stream dominated by shuffle-buffer fill.
    n_seq = 5000 if QUICK else ds.manifest.total_bytes // SEQ_LEN
    print(
        f"\nbenchmarking with n={n} chunks / n_seq={n_seq} sequences (quick={QUICK})\n"
    )

    repo.cache.clear()  # 1-2: cold vs warm sequential raw chunk bytes (R2 vs disk)
    bench("1_cold_sequential", ds.samples(), n, bytes_per_item_fn=chunk_bytes)
    bench("2_warm_sequential", ds.samples(), n, bytes_per_item_fn=chunk_bytes)

    bench(  # 3: byte-level tokenize (np.frombuffer uint8 -> int64)
        "3_warm_tokenize",
        ds.samples().map(tokenize_chunk),
        n,
        bytes_per_item_fn=tokenized_raw_bytes,
        extra_fn=tokenized_mtokens,
        extra_label="Mtokens/s",
    )

    seq_stream = build_packed_pipeline(ds.samples())
    bench(  # 4: + rolling-remainder packing into fixed 2048-token sequences
        "4_warm_tokenpack",
        seq_stream,
        n_seq,
        extra_fn=seq_mtokens,
        extra_label="Mtokens/s",
    )

    shuffled_seq_stream = build_packed_pipeline(  # 5: + document shuffle + seq buffer
        ds.samples().shuffle(seed=0)
    ).shuffle(SHUFFLE_BUF_SEQ, seed=0)
    shuffled_seq_per_s = bench(
        "5_warm_tokenpack_shuffled",
        shuffled_seq_stream,
        n_seq,
        extra_fn=seq_mtokens,
        extra_label="Mtokens/s",
    )
    mtokens_per_s = shuffled_seq_per_s * SEQ_LEN / 1e6

    tokenpack_pipeline = shuffled_seq_stream.batch(BATCH_SIZE, collate=np.stack)
    n_batches = max(1, n_seq // BATCH_SIZE)
    # num_workers=2 would fork two extra full torch processes (~1 GB RSS each
    # on top of the parent) -- more than this 3.8 GB shared box can spare, so
    # only the in-process path runs here; b01 measures real worker scaling.
    print("  (6_num_workers=2 skipped: 3 torch processes exceed this box's RAM)")
    for nw in (0,):  # 6: torch DataLoader worker scaling, batches of [8, 2048]
        loader = torch.utils.data.DataLoader(
            as_iterable_dataset(tokenpack_pipeline), batch_size=None, num_workers=nw
        )
        bench(f"6_num_workers={nw}", loader, n_batches, extra_label="batches/s")

    print(f"\n(warm_tokenpack_shuffled: {mtokens_per_s:.3f} Mtokens/s)")
    return mtokens_per_s


def h100_feasibility(measured_mtokens_per_s: float) -> None:
    """FLOPs-based tokens/s an H100 (or a few) needs fed to it to stay busy
    pretraining a decoder-only LM, vs. what this box's warm_tokenpack_shuffled
    stage actually delivers."""
    peak_flops = 990e12  # H100 SXM, bf16 dense
    mfu = 0.40
    effective_flops = peak_flops * mfu

    print("\n" + "=" * 78)
    print("H100 FEASIBILITY: pretraining a decoder-only LM")
    print("=" * 78)
    print(
        f"assumptions: H100 bf16 peak={peak_flops:.2e} FLOPs/s, MFU={mfu:.0%}\n"
        "formula: tokens/s consumed = n_gpus * peak_flops * MFU / (6 * N_params)\n"
    )
    print(f"  {'N params':>9} {'n_gpus':>7} {'req Mtok/s':>11} {'x this box':>11}")
    worst_mult, worst_desc = 0.0, ""
    for n_params in (0.125e9, 0.35e9, 0.5e9):
        for n_gpus in (1, 2, 3, 4):
            required_mtok = (n_gpus * effective_flops / (6 * n_params)) / 1e6
            mult = required_mtok / measured_mtokens_per_s
            print(
                f"  {n_params / 1e9:>7.3f}B {n_gpus:>7} {required_mtok:>10.3f}M"
                f" {mult:>10.1f}x"
            )
            if mult > worst_mult:
                worst_mult, worst_desc = (
                    mult,
                    f"{n_params / 1e9:.3f}B params x {n_gpus} GPUs",
                )

    print(
        f"\nthis box's measured warm_tokenpack_shuffled throughput: "
        f"{measured_mtokens_per_s:.3f} Mtokens/s"
    )
    print(
        f"worst case ({worst_desc}) needs {worst_mult:.0f}x that.\n"
        "honest caveat: byte-level tokenization here is compute-trivial (np.frombuffer,\n"
        "no vocabulary lookup) vs. a real BPE tokenizer, which typically runs 5-10x\n"
        "slower per token -- scale the headroom multiples down accordingly for a\n"
        "production tokenizer. dload's SourceNode splits shards[worker::num_workers]\n"
        "with exact per-epoch coverage and no duplicate downloads, so this scales\n"
        "close to linearly with independent loader workers/processes -- horizontal\n"
        "scaling, not a faster single core, is the fix for small-model / large-GPU-\n"
        "count data starvation."
    )


def main() -> None:
    repo = dload.Repository.open()
    ds = ingest(repo)
    mtokens_per_s = run_benchmarks(repo, ds)
    h100_feasibility(mtokens_per_s)


if __name__ == "__main__":
    main()
