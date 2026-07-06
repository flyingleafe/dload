"""`dload` command-line interface."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import click

from .cache import ShardCache
from .config import Config, _xdg_config_home, format_size, parse_size, write_config_file
from .errors import DloadError, NotFoundError
from .pack import Sample
from .repo import DEFAULT_SHARD_SIZE, Repository, _ref_key


def _open_repo() -> Repository:
    try:
        return Repository.open()
    except DloadError as exc:
        raise click.ClickException(str(exc)) from exc


def _version_option(f):
    return click.option(
        "--version",
        "version_prefix",
        default=None,
        metavar="PREFIX",
        help="Version id or prefix (default: pinned or latest).",
    )(f)


def _echo_config(cfg: Config) -> None:
    click.echo(f"endpoint_url: {cfg.endpoint_url or '(not set)'}")
    click.echo(f"bucket: {cfg.bucket or '(not set)'}")
    click.echo(f"prefix: {cfg.prefix or '(none)'}")
    click.echo(f"cache_dir: {cfg.cache_dir}")
    click.echo(
        f"cache_budget: {format_size(cfg.cache_budget)} ({cfg.cache_budget_raw})"
    )


def _print_table(headers: list[str], rows: list[tuple]) -> None:
    all_rows = [headers, *[[str(c) for c in row] for row in rows]]
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(headers))]
    for r in all_rows:
        click.echo("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)))


def _walk_samples(root: Path) -> Iterator[Sample]:
    """Group files under `root` into samples keyed by their relative path
    without extension; field name = extension without the dot ("data" for
    extensionless files). Only file paths are collected eagerly; contents
    are read lazily, one group at a time, as the generator is consumed."""
    groups: dict[str, dict[str, Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = PurePosixPath(path.relative_to(root).as_posix())
        suffix = rel.suffix
        field = suffix[1:] if suffix else "data"
        stem = rel.with_suffix("").as_posix() if suffix else rel.as_posix()
        groups.setdefault(stem, {})[field] = path
    for key in sorted(groups):
        yield key, {field: p.read_bytes() for field, p in groups[key].items()}


def _parse_meta(items: tuple[str, ...]) -> dict[str, str]:
    meta = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise click.BadParameter(
                f"expected KEY=VALUE, got {item!r}", param_hint="--meta"
            )
        meta[key] = value
    return meta


class _ErrorHandlingGroup(click.Group):
    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except DloadError as exc:
            raise click.ClickException(str(exc)) from exc


@click.group(cls=_ErrorHandlingGroup)
def cli() -> None:
    """Dataset management, storage, caching and streaming for ML training."""


@cli.command()
@click.option("--endpoint-url", help="S3-compatible endpoint URL.")
@click.option("--bucket", help="Bucket name.")
@click.option("--prefix", help="Key prefix inside the bucket.")
@click.option("--cache-dir", help="Local shard cache root.")
@click.option("--cache-budget", help='"unlimited" | "auto" | a size like "50GB".')
@click.option(
    "--user",
    "target",
    flag_value="user",
    default=True,
    help="Write ~/.config/dload/config.toml (default).",
)
@click.option("--project", "target", flag_value="project", help="Write ./dload.toml.")
def init(endpoint_url, bucket, prefix, cache_dir, cache_budget, target):
    """Create or update a dload config file."""
    path = (
        Path.cwd() / "dload.toml"
        if target == "project"
        else _xdg_config_home(os.environ) / "dload" / "config.toml"
    )
    write_config_file(
        path,
        endpoint_url=endpoint_url,
        bucket=bucket,
        prefix=prefix,
        cache_dir=cache_dir,
        cache_budget=cache_budget,
    )
    click.echo(f"wrote {path}")
    _echo_config(Config.load())


@cli.command()
def status():
    """Show resolved configuration, cache usage and remote reachability."""
    cfg = Config.load()
    _echo_config(cfg)
    cache = ShardCache(cfg.cache_dir, cfg.cache_budget)
    click.echo(
        f"cache: {format_size(cache.used_bytes())} / {format_size(cfg.cache_budget)} used, "
        f"{len(cache.entries())} shards"
    )
    try:
        repo = Repository.open(cfg)
        next(iter(repo.remote.list()), None)
    except DloadError as exc:
        click.echo(f"remote: error ({exc})")
    else:
        click.echo("remote: ok")


@cli.command()
def ls():
    """List datasets in the remote."""
    repo = _open_repo()
    names = repo.list_datasets()
    if not names:
        click.echo("no datasets")
        return
    rows = []
    for name in names:
        try:
            ref = repo.remote.get_bytes(_ref_key(name)).decode().strip()
        except NotFoundError:
            continue
        latest = repo.manifest(name, ref)
        rows.append(
            (
                name,
                latest.version[:12],
                latest.num_samples,
                len(latest.shards),
                format_size(latest.total_bytes),
                latest.created,
            )
        )
    _print_table(["NAME", "VERSION", "SAMPLES", "SHARDS", "SIZE", "CREATED"], rows)


@cli.command()
@click.argument("name")
@_version_option
def info(name, version_prefix):
    """Show manifest details and version history for NAME."""
    repo = _open_repo()
    manifest = repo.manifest(name, version_prefix)
    click.echo(f"name: {manifest.name}")
    click.echo(f"version: {manifest.version}")
    click.echo(f"created: {manifest.created}")
    click.echo(f"samples: {manifest.num_samples}")
    click.echo(f"shards: {len(manifest.shards)}")
    click.echo(f"size: {format_size(manifest.total_bytes)}")
    click.echo(f"meta: {manifest.meta}")
    click.echo(f"recipe: {'yes' if manifest.recipe else 'no'}")

    try:
        latest = repo.remote.get_bytes(_ref_key(name)).decode().strip()
    except NotFoundError:
        latest = None
    pinned = repo._read_lock().get(name)

    click.echo("versions:")
    for m in repo.versions(name):
        marks = [
            label
            for label, hit in (
                ("latest", m.version == latest),
                ("pinned", m.version == pinned),
            )
            if hit
        ]
        suffix = f"  [{', '.join(marks)}]" if marks else ""
        click.echo(f"  {m.version[:12]}  {m.created}{suffix}")


@cli.command()
@click.argument("name")
@click.option(
    "--from",
    "from_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory to ingest, recursively.",
)
@click.option(
    "--recipe",
    "recipe_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File whose text is stored verbatim as the dataset's recipe.",
)
@click.option(
    "--meta", "meta_items", multiple=True, metavar="KEY=VALUE", help="Repeatable."
)
@click.option(
    "--shard-size", "shard_size_text", metavar="SIZE", help="Target shard size."
)
def commit(name, from_dir, recipe_path, meta_items, shard_size_text):
    """Ingest --from DIR as a new version of NAME."""
    target_shard_size = DEFAULT_SHARD_SIZE
    if shard_size_text is not None:
        parsed = parse_size(shard_size_text)
        if parsed is None:
            raise click.BadParameter(
                "shard size cannot be 'unlimited'", param_hint="--shard-size"
            )
        target_shard_size = parsed
    recipe_text = recipe_path.read_text() if recipe_path else None

    repo = _open_repo()
    manifest = repo.commit(
        name,
        _walk_samples(from_dir),
        meta=_parse_meta(meta_items),
        recipe=recipe_text,
        target_shard_size=target_shard_size,
        progress=click.echo,
    )
    click.echo(manifest.version)


@cli.command()
@click.argument("name")
@_version_option
def recipe(name, version_prefix):
    """Print the recipe stored with a dataset version."""
    repo = _open_repo()
    manifest = repo.manifest(name, version_prefix)
    if not manifest.recipe:
        raise click.ClickException(
            f"{name}@{manifest.version[:12]} has no stored recipe"
        )
    click.echo(manifest.recipe, nl=False)


@cli.command()
@click.argument("name")
@_version_option
def pull(name, version_prefix):
    """Fetch every shard of a dataset version into the local cache."""
    repo = _open_repo()
    dataset = repo.dataset(name, version_prefix)
    dataset.fetch(progress=click.echo)
    click.echo(f"pulled {dataset!r}")


@cli.command()
@click.argument("name")
@click.argument("version", required=False)
def pin(name, version):
    """Pin NAME to VERSION (default: current latest) in dload.lock."""
    repo = _open_repo()
    resolved = repo.pin(name, version)
    click.echo(f"pinned {name} -> {resolved}")


@cli.command()
@click.argument("name")
def unpin(name):
    """Remove NAME's pin from dload.lock."""
    repo = _open_repo()
    repo.unpin(name)
    resolved = repo.resolve(name)
    click.echo(f"unpinned {name}, now resolves to {resolved}")


@cli.group()
def cache():
    """Manage the local shard cache."""


@cache.command("status")
def cache_status():
    """Show cache usage."""
    cfg = Config.load()
    c = ShardCache(cfg.cache_dir, cfg.cache_budget)
    entries = c.entries()
    click.echo(f"cache_dir: {cfg.cache_dir}")
    click.echo(f"used: {format_size(c.used_bytes())} / {format_size(cfg.cache_budget)}")
    click.echo(f"shards: {len(entries)}")


@cache.command("clear")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def cache_clear(yes):
    """Remove all unpinned shards from the local cache."""
    cfg = Config.load()
    c = ShardCache(cfg.cache_dir, cfg.cache_budget)
    if not yes:
        click.confirm(f"Clear cache at {cfg.cache_dir}?", abort=True)
    freed = c.clear()
    click.echo(f"freed {format_size(freed)}")


@cli.command()
@click.argument("name")
@_version_option
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def rm(name, version_prefix, yes):
    """Delete a dataset, or one version of it."""
    repo = _open_repo()
    label = f"{name}@{version_prefix}" if version_prefix else name
    if not yes:
        click.confirm(f"Delete {label}?", abort=True)
    repo.delete_dataset(name, version_prefix)
    click.echo(f"deleted {label}. Run `dload gc` to reclaim orphaned shards.")


@cli.command()
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def gc(yes):
    """Delete remote shards referenced by no manifest of any dataset."""
    repo = _open_repo()
    if not yes:
        click.confirm("Run garbage collection on the remote store?", abort=True)
    deleted, freed = repo.gc(progress=click.echo)
    click.echo(f"freed {deleted} shards, {format_size(freed)}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
