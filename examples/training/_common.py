"""Shared plumbing for the training examples: cached downloads, idempotent
dataset commits, and a learning-progress assertion."""

from __future__ import annotations

import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

import dload

DOWNLOADS = Path(__file__).resolve().parent.parent.parent / ".downloads"


def http_download(url: str, filename: str) -> Path:
    """Download `url` once into .downloads/ and reuse it afterwards."""
    dest = DOWNLOADS / filename
    if dest.exists():
        print(f"using cached download {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def ensure_dataset(
    repo: dload.Repository,
    name: str,
    build: Callable[[], Iterable[tuple[str, dict[str, bytes]]]],
    *,
    recipe: str,
    meta: dict | None = None,
    target_shard_size: int = 4 * 1024 * 1024,
) -> dload.Dataset:
    """Open `name` from the remote, committing it first if it isn't there.
    `build` is only called (and its downloads only happen) on first commit."""
    try:
        ds = repo.dataset(name)
        print(f"found in remote: {ds!r}")
        return ds
    except dload.NotFoundError:
        pass
    print(f"dataset {name!r} not in remote yet — building and committing")
    repo.commit(
        name,
        build(),
        recipe=recipe,
        meta=meta or {},
        target_shard_size=target_shard_size,
        progress=print,
    )
    return repo.dataset(name)


def assert_improved(
    initial: float, final: float, *, factor: float = 0.8, name: str = "loss"
) -> None:
    """The training actually learned something: final < factor * initial."""
    assert final < factor * initial, (
        f"{name} did not improve enough: {initial:.4f} -> {final:.4f}"
    )
    print(f"{name} improved: {initial:.4f} -> {final:.4f} (x{final / initial:.2f})")
