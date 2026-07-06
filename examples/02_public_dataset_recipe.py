"""Ingest a public dataset with a preserved download recipe.

`download_fsdd` fetches the Free Spoken Digit Dataset from GitHub and
extracts its 3000 short WAV clips. The function's own source is stored
verbatim as the manifest's `recipe`, so anyone who later reads this dataset
back from R2 can see exactly how it was obtained — no separate README to
go stale.
"""

# ruff: noqa: E402
from __future__ import annotations

import inspect
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _env

_env.setup()

import dload
from dload.config import format_size

FSDD_URL = (
    "https://github.com/Jakobovski/free-spoken-digit-dataset/"
    "archive/refs/heads/master.zip"
)
DOWNLOAD_CACHE = Path(__file__).resolve().parent.parent / ".fsdd-download"
TARGET_SHARD_SIZE = 4 * 1024 * 1024


def download_fsdd(dest_dir: Path) -> list[Path]:
    """Download the FSDD repo zip (skipped if already cached) and extract
    its recordings/*.wav clips into dest_dir (skipped if already there).
    Returns the list of extracted wav paths, sorted."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest_dir.glob("*.wav"))
    if existing:
        return existing

    zip_path = DOWNLOAD_CACHE / "fsdd-master.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        urllib.request.urlretrieve(FSDD_URL, zip_path)  # noqa: S310

    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if "/recordings/" in name and name.endswith(".wav"):
                data = zf.read(name)
                (dest_dir / Path(name).name).write_bytes(data)
    return sorted(dest_dir.glob("*.wav"))


def samples_from_wavs(paths: list[Path]):
    for path in paths:
        digit, speaker, _index = path.stem.split("_")
        meta = dload.codecs.json_bytes({"digit": int(digit), "speaker": speaker})
        yield path.stem, {"wav": path.read_bytes(), "meta": meta}


print(f"downloading/extracting FSDD into {DOWNLOAD_CACHE / 'recordings'} ...")
t0 = time.monotonic()
wav_paths = download_fsdd(DOWNLOAD_CACHE / "recordings")
print(f"  {len(wav_paths)} wav clips ready in {time.monotonic() - t0:.2f}s")

repo = dload.Repository.open()
print("committing to R2 as 'fsdd'...")
manifest = repo.commit(
    "fsdd",
    samples_from_wavs(wav_paths),
    meta={"source": "github/Jakobovski"},
    recipe=inspect.getsource(download_fsdd),
    target_shard_size=TARGET_SHARD_SIZE,
    progress=print,
)
print()
print("manifest summary:")
print(f"  name:    {manifest.name}")
print(f"  version: {manifest.version}")
print(f"  samples: {manifest.num_samples}")
print(f"  shards:  {len(manifest.shards)}")
print(f"  size:    {format_size(manifest.total_bytes)}")

# Round-trip: read the manifest back from R2 as a fresh call and prove the
# recipe survived the trip.
print()
print("reading manifest back from R2 to verify the recipe round-tripped...")
fetched = repo.manifest("fsdd")
assert fetched.recipe == inspect.getsource(download_fsdd)
print("--- manifest.recipe ---")
print(fetched.recipe)
print("--- end recipe (matches download_fsdd source exactly) ---")
