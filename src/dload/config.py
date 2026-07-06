"""Layered configuration.

Resolution order (first hit wins, per individual setting):

1. Environment variables:
     DLOAD_ENDPOINT_URL   S3 endpoint (for R2: https://<account>.r2.cloudflarestorage.com;
                          derived automatically from R2_ACCOUNT_ID when set)
     DLOAD_BUCKET         bucket name
     DLOAD_PREFIX         key prefix inside the bucket (default "")
     DLOAD_CACHE_DIR      local cache root
     DLOAD_CACHE_BUDGET   "unlimited" | "auto" | size like "50GB"
   Credentials are NOT handled here: boto3's standard chain
   (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / profiles) applies.
2. Project file: nearest `dload.toml` in cwd or its ancestors.
3. User file: ~/.config/dload/config.toml (XDG_CONFIG_HOME respected).
4. Defaults: cache_dir=~/.cache/dload (XDG_CACHE_HOME respected),
   cache_budget="auto", prefix="".

TOML shape (same for project and user files):

    [remote]
    endpoint_url = "https://....r2.cloudflarestorage.com"
    bucket = "my-data"
    prefix = ""

    [cache]
    dir = "/scratch/me/dload-cache"
    budget = "unlimited"
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigError

_DECIMAL_UNITS = {
    "": 1,
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
}
_BINARY_UNITS = {
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}
_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)\s*$")


def parse_size(text: str) -> int | None:
    """ "50GB"/"1.5TiB"/"800M"/"unlimited" -> bytes or None for unlimited.

    Decimal (KB/MB/GB/TB = powers of 1000) and binary (KiB/... = powers of
    1024) units, bare integers = bytes, case-insensitive. Raises
    dload.errors.ConfigError on garbage."""
    if text.strip().lower() == "unlimited":
        return None
    m = _SIZE_RE.match(text)
    if not m:
        raise ConfigError(f"invalid size: {text!r}")
    number, unit = m.groups()
    unit = unit.upper()
    if unit in _BINARY_UNITS:
        factor = _BINARY_UNITS[unit]
    elif unit in _DECIMAL_UNITS:
        factor = _DECIMAL_UNITS[unit]
    elif unit == "K":
        factor = _DECIMAL_UNITS["KB"]
    elif unit == "M":
        factor = _DECIMAL_UNITS["MB"]
    elif unit == "G":
        factor = _DECIMAL_UNITS["GB"]
    elif unit == "T":
        factor = _DECIMAL_UNITS["TB"]
    else:
        raise ConfigError(f"invalid size unit: {text!r}")
    return int(float(number) * factor)


def format_size(n: int | None) -> str:
    """Human-readable size ("unlimited" for None); binary units, 1 decimal."""
    if n is None:
        return "unlimited"
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(n)
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} {units[-1]}"


def _home(env: Mapping[str, str]) -> Path:
    h = env.get("HOME")
    return Path(h) if h else Path.home()


def _xdg_config_home(env: Mapping[str, str]) -> Path:
    v = env.get("XDG_CONFIG_HOME")
    return Path(v) if v else _home(env) / ".config"


def _xdg_cache_home(env: Mapping[str, str]) -> Path:
    v = env.get("XDG_CACHE_HOME")
    return Path(v) if v else _home(env) / ".cache"


def _expand_home(text: str, env: Mapping[str, str]) -> str:
    if text == "~" or text.startswith("~/"):
        return str(_home(env)) + text[1:]
    return text


def _find_project_file(cwd: Path) -> Path | None:
    d = cwd.resolve()
    for candidate_dir in (d, *d.parents):
        candidate = candidate_dir / "dload.toml"
        if candidate.exists():
            return candidate
    return None


def _load_toml_layer(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _auto_budget(cache_dir: Path) -> int:
    p = cache_dir
    while not p.exists():
        parent = p.parent
        if parent == p:
            break
        p = parent
    st = os.statvfs(p)
    return (st.f_bavail * st.f_frsize) // 2


@dataclass(frozen=True, slots=True)
class Config:
    endpoint_url: str | None
    bucket: str | None
    prefix: str
    cache_dir: Path
    cache_budget: int | None  # bytes; None = unlimited
    cache_budget_raw: str  # the setting as written ("auto"/"unlimited"/"50GB")

    @classmethod
    def load(
        cls, cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> "Config":
        """Resolve the layered configuration. `env` defaults to os.environ
        (injectable for tests). "auto" budget = half of the free space of
        the filesystem holding cache_dir (statvfs of the nearest existing
        ancestor), resolved here so `cache_budget` is always int|None."""
        env = dict(os.environ) if env is None else env
        cwd = Path.cwd() if cwd is None else Path(cwd)

        project = _load_toml_layer(_find_project_file(cwd))
        user = _load_toml_layer(_xdg_config_home(env) / "dload" / "config.toml")

        env_endpoint = env.get("DLOAD_ENDPOINT_URL")
        if not env_endpoint and env.get("R2_ACCOUNT_ID"):
            env_endpoint = f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"

        env_layer: dict[str, str | None] = {
            "endpoint_url": env_endpoint,
            "bucket": env.get("DLOAD_BUCKET"),
            "prefix": env.get("DLOAD_PREFIX"),
            "cache_dir": env.get("DLOAD_CACHE_DIR"),
            "cache_budget": env.get("DLOAD_CACHE_BUDGET"),
        }
        project_remote = project.get("remote", {})
        project_cache = project.get("cache", {})
        project_layer: dict[str, str | None] = {
            "endpoint_url": project_remote.get("endpoint_url"),
            "bucket": project_remote.get("bucket"),
            "prefix": project_remote.get("prefix"),
            "cache_dir": project_cache.get("dir"),
            "cache_budget": project_cache.get("budget"),
        }
        user_remote = user.get("remote", {})
        user_cache = user.get("cache", {})
        user_layer: dict[str, str | None] = {
            "endpoint_url": user_remote.get("endpoint_url"),
            "bucket": user_remote.get("bucket"),
            "prefix": user_remote.get("prefix"),
            "cache_dir": user_cache.get("dir"),
            "cache_budget": user_cache.get("budget"),
        }
        default_layer: dict[str, str | None] = {
            "endpoint_url": None,
            "bucket": None,
            "prefix": "",
            "cache_dir": str(_xdg_cache_home(env) / "dload"),
            "cache_budget": "auto",
        }

        def resolve(field: str) -> str | None:
            for layer in (env_layer, project_layer, user_layer, default_layer):
                v = layer[field]
                if v is not None:
                    return v
            return None

        cache_dir_raw = resolve("cache_dir")
        assert cache_dir_raw is not None  # default_layer always supplies one
        cache_dir = Path(_expand_home(cache_dir_raw, env))
        cache_budget_raw = resolve("cache_budget")
        assert cache_budget_raw is not None  # default_layer always supplies one
        cache_budget = (
            _auto_budget(cache_dir)
            if cache_budget_raw.strip().lower() == "auto"
            else parse_size(cache_budget_raw)
        )

        prefix = resolve("prefix")
        assert prefix is not None  # default_layer always supplies one

        return cls(
            endpoint_url=resolve("endpoint_url"),
            bucket=resolve("bucket"),
            prefix=prefix,
            cache_dir=cache_dir,
            cache_budget=cache_budget,
            cache_budget_raw=cache_budget_raw,
        )

    def require_remote(self) -> tuple[str, str]:
        """(endpoint_url, bucket) or raise ConfigError with a message that
        tells the user exactly how to fix it (env vars / dload init)."""
        missing = []
        if not self.endpoint_url:
            missing.append(
                "endpoint_url (set DLOAD_ENDPOINT_URL or R2_ACCOUNT_ID, "
                "or [remote] endpoint_url in dload.toml)"
            )
        if not self.bucket:
            missing.append(
                "bucket (set DLOAD_BUCKET, or [remote] bucket in dload.toml)"
            )
        if missing:
            raise ConfigError(
                "missing remote configuration: "
                + "; ".join(missing)
                + ". Run `dload init` to create a dload.toml, or export the env vars."
            )
        assert self.endpoint_url is not None and self.bucket is not None
        return self.endpoint_url, self.bucket


def write_config_file(
    path: Path,
    *,
    endpoint_url: str | None = None,
    bucket: str | None = None,
    prefix: str | None = None,
    cache_dir: str | None = None,
    cache_budget: str | None = None,
) -> None:
    """Create or update a dload TOML config file, preserving existing
    settings that are not being changed (read-modify-write; tomllib to
    read, minimal manual TOML emission to write — values are flat strings,
    no escaping headaches beyond quotes/backslashes)."""
    existing = _load_toml_layer(path)
    remote = dict(existing.get("remote", {}))
    cache = dict(existing.get("cache", {}))

    if endpoint_url is not None:
        remote["endpoint_url"] = endpoint_url
    if bucket is not None:
        remote["bucket"] = bucket
    if prefix is not None:
        remote["prefix"] = prefix
    if cache_dir is not None:
        cache["dir"] = cache_dir
    if cache_budget is not None:
        cache["budget"] = cache_budget

    lines = []
    if remote:
        lines.append("[remote]")
        lines.extend(f"{k} = {_toml_quote(v)}" for k, v in remote.items())
        lines.append("")
    if cache:
        lines.append("[cache]")
        lines.extend(f"{k} = {_toml_quote(v)}" for k, v in cache.items())
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip("\n") + "\n" if lines else "")


def _toml_quote(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
