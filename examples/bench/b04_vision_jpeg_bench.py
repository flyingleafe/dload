# ruff: noqa: E402
"""Vision JPEG ingestion + augmentation benchmark (no model training).

Ingests Tiny-ImageNet's 100k train JPEGs (~237 MB zip, 64x64 px, 200
classes) once, streamed straight out of the zip (`namelist()` once, then
`zf.read(member)` per file -- never extracted to disk). Measures warm-cache
raw-bytes throughput, JPEG decode (the classic vision bottleneck), a full
RandomResizedCrop + hflip + ImageNet-normalize augmentation, batched
collation, and DataLoader worker scaling. JPEG bytes are stored *compressed*
in the shards and decoded only inside the pipeline (`PIL.Image.open` in
`.map`) -- the same store-compressed/decode-in-pipeline tradeoff `b01`/`b02`
make for flac/text: less shard bytes and R2 egress, traded for a decode
step every epoch. Ends with a back-of-envelope H100 feasibility check for a
ViT-style classifier.
"""

from __future__ import annotations

import inspect
import io
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import _env

_env.setup()
os.environ["DLOAD_CACHE_DIR"] = str(_env.REPO_ROOT / ".dload-cache-bench")

import dload
from _common import ensure_dataset, http_download  # pyright: ignore[reportMissingImports]
from dload.torch import as_iterable_dataset

QUICK = "--quick" in sys.argv
QUICK_N = 2000

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATASET_NAME = "tiny-imagenet-train"
TARGET_SHARD_SIZE = 32 * 1024 * 1024

IMG_SIZE = 64
BATCH_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _train_jpeg_names(zip_path: Path) -> list[str]:
    """List (without extracting) every train JPEG's member name:
    tiny-imagenet-200/train/<wnid>/images/<file>.JPEG."""
    with zipfile.ZipFile(zip_path) as zf:
        return [n for n in zf.namelist() if "/train/" in n and n.endswith(".JPEG")]


def download_tiny_imagenet():
    """Stream train JPEGs straight out of the zip (no disk extraction).
    Yields key f"{wnid}/{stem}", fields {"jpeg": raw bytes, "label": wnid}."""
    zip_path = http_download(TINY_IMAGENET_URL, "tiny-imagenet-200.zip")
    names = _train_jpeg_names(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for name in names:
            wnid = name.split("/")[2]
            stem = Path(name).stem
            yield (
                f"{wnid}/{stem}",
                {"jpeg": zf.read(name), "label": wnid.encode("utf-8")},
            )


def ingest(repo: dload.Repository) -> dload.Dataset:
    try:
        ds = repo.dataset(DATASET_NAME)
        print(f"dataset {DATASET_NAME!r}: {ds!r} (already ingested)")
        return ds
    except dload.NotFoundError:
        pass
    print(f"dataset {DATASET_NAME!r} not in remote yet -- downloading + committing")
    zip_path = http_download(TINY_IMAGENET_URL, "tiny-imagenet-200.zip")
    names = _train_jpeg_names(zip_path)
    wnids = {n.split("/")[2] for n in names}
    t0 = time.monotonic()
    ds = ensure_dataset(
        repo,
        DATASET_NAME,
        download_tiny_imagenet,
        recipe=inspect.getsource(download_tiny_imagenet),
        meta={
            "source": TINY_IMAGENET_URL,
            "n_classes": len(wnids),
            "n_images": len(names),
        },
        target_shard_size=TARGET_SHARD_SIZE,
    )
    dt = time.monotonic() - t0
    mb = ds.manifest.total_bytes / 2**20
    print(
        f"ingest done: {dt:.1f}s, {mb:.1f} MB, {len(names) / dt:.1f} images/s, "
        f"{mb / dt:.2f} MB/s"
    )
    return ds


# decode / augment / collate below are module-level so they survive pickling
# into spawned DataLoader workers (see 06_torch_dataloader.py).


def decode(sample: tuple[str, dict[str, bytes]]) -> tuple[str, np.ndarray]:
    """JPEG bytes -> RGB uint8 numpy, shape [64, 64, 3] (HWC)."""
    key, fields = sample
    img = Image.open(io.BytesIO(fields["jpeg"])).convert("RGB")
    return key, np.asarray(img, dtype=np.uint8)


def augment(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray]:
    """RandomResizedCrop-style scale(0.6-1.0) crop -> resize to 64x64 ->
    random hflip -> ImageNet-normalize to float32 CHW [3, 64, 64]."""
    key, img = item
    rng = np.random.default_rng(dload.seeded(key, "aug"))
    h, w, _c = img.shape
    scale = float(rng.uniform(0.6, 1.0))
    crop_h = max(1, int(round(h * scale)))
    crop_w = max(1, int(round(w * scale)))
    top = int(rng.integers(0, h - crop_h + 1))
    left = int(rng.integers(0, w - crop_w + 1))
    cropped = img[top : top + crop_h, left : left + crop_w]
    resized = Image.fromarray(cropped).resize(
        (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
    )
    if rng.random() < 0.5:
        resized = resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)
    return key, chw


def collate_chw(items: list[tuple[str, np.ndarray]]) -> np.ndarray:
    return np.stack([arr for _key, arr in items])


def batch_images(batch: np.ndarray) -> int:
    return batch.shape[0]


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
    row = f"  {name:<22} {n:>6} it  {dt:>7.2f}s  {items_per_s:>9.1f} it/s"
    if mb_per_s is not None:
        row += f"  {mb_per_s:>9.2f} MB/s"
    if extra_per_s is not None:
        row += f"  {extra_per_s:>9.2f} {extra_label}"
    print(row)
    return items_per_s


def run_benchmarks(repo: dload.Repository, ds: dload.Dataset) -> float:
    n = QUICK_N if QUICK else len(ds)
    print(f"\nbenchmarking with n={n} images per measurement (quick={QUICK})\n")

    # ingest's commit already writes shards into the local cache, so there's
    # no interesting "cold" R2 stage here the way b01/b02 have one -- this is
    # a warm-cache-only run.
    bench("1_warm_raw", ds.samples(), n, bytes_per_item_fn=lambda s: len(s[1]["jpeg"]))

    bench(  # 2: JPEG -> RGB uint8 numpy (the classic vision bottleneck)
        "2_warm_decode",
        ds.samples().map(decode),
        n,
        extra_fn=lambda item: item[1].shape[0] * item[1].shape[1] / 1e6,
        extra_label="MPix/s",
    )

    aug_images_s = bench(  # 3: + RandomResizedCrop-style + hflip + normalize
        "3_warm_augment", ds.samples().map(decode).map(augment), n
    )

    batch_pipe = (
        ds.samples().map(decode).map(augment).batch(BATCH_SIZE, collate=collate_chw)
    )
    n_batches = max(1, n // BATCH_SIZE)
    bench(  # 4: + batch(256, collate=np.stack) -- batches/s and images/s
        "4_batch_collate",
        batch_pipe,
        n_batches,
        extra_fn=batch_images,
        extra_label="images/s",
    )

    for nw in (0, 2):  # 5: torch DataLoader worker scaling, batches of 256
        loader = torch.utils.data.DataLoader(
            as_iterable_dataset(batch_pipe), batch_size=None, num_workers=nw
        )
        try:
            bench(
                f"5_num_workers={nw}",
                loader,
                n_batches,
                extra_fn=batch_images,
                extra_label="images/s",
            )
        except RuntimeError as e:
            # A DataLoader worker OOM-killed by the memory sandbox surfaces
            # as "worker exited unexpectedly" — report instead of crashing:
            # num_workers>0 forks full extra torch processes, which a small
            # shared box cannot always afford.
            print(f"  5_num_workers={nw}: skipped ({type(e).__name__}: {e})")

    print(f"\n(warm_augment: {aug_images_s:.1f} images/s)")
    return aug_images_s


def h100_feasibility(measured_images_per_s: float) -> None:
    """FLOPs-based images/s an H100 (or a few) needs fed to it to stay busy
    training a ViT-style classifier, vs. this box's warm_augment throughput.
    Tiny-ImageNet is natively 64px; assumes upscaling to a standard 224px/
    patch16 ViT input (196 patch tokens + 1 cls = 197 tokens/image)."""
    tokens_per_image = 197  # 224px / patch16 -> 14*14=196 patches + cls
    batch_per_gpu = 256
    peak_flops = 990e12  # H100 SXM, bf16 dense
    mfu = 0.40
    effective_flops = peak_flops * mfu

    print("\n" + "=" * 78)
    print("H100 FEASIBILITY: training a ViT-style image classifier")
    print("=" * 78)
    print(
        f"assumptions: native Tiny-ImageNet px=64 upscaled to 224 for training, "
        f"patch=16 -> tokens/image={tokens_per_image}\n"
        f"             batch/gpu={batch_per_gpu}, H100 bf16 peak={peak_flops:.2e} "
        f"FLOPs/s, MFU={mfu:.0%}"
    )
    print(
        "formulas: flops/step/gpu = 6*N*tokens_per_image*batch_per_gpu (fwd+bwd, dense)\n"
        "          step_time = flops/step/gpu / (peak_flops*MFU)\n"
        "          required_images/s = batch_per_gpu*n_gpus / step_time\n"
    )
    print(
        f"  {'N params':>9} {'n_gpus':>7} {'step_ms':>9} {'req img/s':>12} {'x this box':>11}"
    )
    worst_mult, worst_desc = 0.0, ""
    for n_params in (0.086e9, 0.3e9, 0.5e9):
        step_time = 6 * n_params * tokens_per_image * batch_per_gpu / effective_flops
        for n_gpus in (1, 2, 3, 4):
            required = batch_per_gpu * n_gpus / step_time
            mult = required / measured_images_per_s
            print(
                f"  {n_params / 1e9:>7.3f}B {n_gpus:>7} {step_time * 1000:>8.1f}ms"
                f" {required:>12.1f} {mult:>10.1f}x"
            )
            if mult > worst_mult:
                worst_mult, worst_desc = (
                    mult,
                    f"{n_params / 1e9:.3f}B params x {n_gpus} GPUs",
                )

    print(
        f"\nthis box's measured warm_augment throughput: {measured_images_per_s:.1f} images/s"
    )
    print(
        f"worst case ({worst_desc}) needs {worst_mult:.0f}x that -- e.g. ~{worst_mult:.0f} cores "
        f"of loader like this one (has 2), or a handful of typical 16-32 core dataloader hosts.\n"
        "dload's SourceNode splits shards[worker::num_workers] with exact per-epoch coverage and "
        "no duplicate downloads (see torch.py), so this scales close to linearly with independent "
        "loader processes/hosts -- horizontal, not single-box, is the fix for small-model / "
        "large-GPU-count data starvation."
    )


def main() -> None:
    repo = dload.Repository.open()
    ds = ingest(repo)
    images_per_s = run_benchmarks(repo, ds)
    h100_feasibility(images_per_s)


if __name__ == "__main__":
    main()
