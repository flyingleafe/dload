"""Shared setup for the examples: point dload at the dload-test R2 bucket.

Loads credentials from the repo's .env and scopes everything the examples
do under the "examples/" prefix, with a cache local to the repo so the
examples never touch your real dload cache.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def setup(cache_budget: str = "unlimited") -> None:
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        raise SystemExit("examples need R2 credentials in .env at the repo root")
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
    os.environ["DLOAD_BUCKET"] = "dload-test"
    os.environ["DLOAD_PREFIX"] = "examples/"
    os.environ["DLOAD_CACHE_DIR"] = str(REPO_ROOT / ".dload-cache")
    os.environ["DLOAD_CACHE_BUDGET"] = cache_budget
