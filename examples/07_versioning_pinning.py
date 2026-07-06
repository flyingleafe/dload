"""Versioning and pinning: a second commit under the same name, and locking
a specific version in `dload.lock` regardless of what "latest" becomes.

Regenerates the exact same 200 seeded "lab-tones" clips from
01_commit_lab_data.py, plus 50 new ones from a different seed branch, and
commits them all under the *same* dataset name — producing a new version
that shares most of its shards with the old one (content-addressed dedup
across versions, not just across re-commits).
"""

# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _env

_env.setup()

import _labtones
import dload
from dload.config import format_size

BASE_SEED = 42
NEW_SEED = 43
N_BASE = 200
N_NEW = 50
TARGET_SHARD_SIZE = 4 * 1024 * 1024
SCRATCH_LOCK_DIR = Path(".dload-tmp")

repo = dload.Repository.open()

print(
    f"regenerating the original {N_BASE} clips (seed={BASE_SEED}) plus "
    f"{N_NEW} new ones from a different seed branch (seed={NEW_SEED})..."
)
samples = _labtones.clip_samples(N_BASE, BASE_SEED) + _labtones.clip_samples(
    N_NEW, NEW_SEED, key_prefix="clip-new"
)
manifest_v2 = repo.commit(
    "lab-tones",
    samples,
    meta={
        "source": "synthetic-lab",
        "n_samples": str(N_BASE + N_NEW),
        "sample_rate": "16000",
        "seed": f"{BASE_SEED}+{NEW_SEED}",
    },
    recipe=Path(__file__).read_text(),
    target_shard_size=TARGET_SHARD_SIZE,
    progress=print,
)

print()
versions = repo.versions("lab-tones")
print(f"lab-tones now has {len(versions)} versions (newest first):")
for m in versions:
    print(
        f"  {m.version[:12]}  {m.created}  {m.num_samples} samples, {len(m.shards)} shards"
    )

new_version, old_version = versions[0], versions[1]
assert new_version.version == manifest_v2.version

shared = {s.digest for s in old_version.shards} & {s.digest for s in new_version.shards}
print()
print(
    f"shards shared between the two versions: {len(shared)} / {len(old_version.shards)} "
    f"of the old version's shards ({format_size(sum(s.size for s in old_version.shards if s.digest in shared))} "
    "reused, not re-uploaded)"
)

# Pin the old version, in a scratch lock file so the repo's own dload.lock
# stays untouched.
SCRATCH_LOCK_DIR.mkdir(exist_ok=True)
repo.lock_path = SCRATCH_LOCK_DIR / "dload.lock"
print()
print(f"pinning lab-tones@{old_version.version[:12]} in {repo.lock_path}...")
repo.pin("lab-tones", old_version.version)
resolved = repo.dataset("lab-tones")
print(
    f"repo.dataset('lab-tones') now resolves to {resolved.version[:12]} (the pinned old version)"
)
assert resolved.version == old_version.version

print()
print("unpinning...")
repo.unpin("lab-tones")
resolved = repo.dataset("lab-tones")
print(
    f"repo.dataset('lab-tones') now resolves to {resolved.version[:12]} (back to latest)"
)
assert resolved.version == new_version.version

print()
print(f"lock file contents ({repo.lock_path}):")
print(repo.lock_path.read_text().rstrip() or "(empty — no pins)")
